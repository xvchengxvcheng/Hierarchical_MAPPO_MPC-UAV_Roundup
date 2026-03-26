"""
UAV 多智能体环境（PID 控制器版本）。

该模块实现了一个“追捕/围捕”任务的仿真环境：
- `num_agents` 个追捕 UAV（索引 0..num_agents-1）
- 1 个目标 UAV（索引 num_agents）

动作语义（高层）：
- `step_still()` 输入的 `actions` 表示追捕者的“期望子目标位置”（随后会被安全可行域投影），
  再由各自控制器（PID/目标控制器）把子目标转换成控制量并推进动力学。

几何可行域：
- 由上一时刻观测得到的半空间 `planes` 定义。
- 期望子目标会先 clip 到地图边界，再投影到 `planes` 的可行区域。

可选依赖：
- 若存在 `casadi`，部分投影/规划相关函数可使用 QP 求解；否则会回退到几何投影方案。
"""

import numpy as np
import random
from control.control_mpc import Controller
from control.control_pid_1 import QuadrotorControllerXYZPos
from control.mpc_path import showmap
from envs.Quadrotor import *
import matplotlib
import os

# 通过环境变量判断是否启用实时显示（单并行环境时）
enable_realtime = os.environ.get('ENABLE_REALTIME_DISPLAY', 'False').lower() == 'true'
if enable_realtime:
    matplotlib.use('TkAgg')  # 使用交互式后端
else:
    matplotlib.use('Agg')  # 非交互式后端，只保存图片

import matplotlib.pyplot as plt

if enable_realtime:
    plt.ion()  # 开启交互模式

# Import casadi for QP solver (used in project_goal_to_planes_qp)
try:
    import casadi as ca
except ImportError:
    ca = None
    print("Warning: casadi not available, project_goal_to_planes_qp will use fallback method")

# Reuse shared geometry/visualization utilities to avoid duplicated implementations.
# The local definitions above are kept for backward compatibility, but the environment
# class below will use the imported symbols (they override the names in globals).
from envs.sim_env_common import (
    QuadSim3D,
    sellect,
    bin_dirs_into_16,
    plot_dirs_by_bucket,
    plane_from_ray_dir,
    get_collision_plane,
    is_inside_planes,
    project_goal_to_planes,
    project_goal_to_planes_qp,
    sample_start_goal_3d,
    sample_point_on_edge_3d,
    reward_tetra_shape_only,
    inter_uav_safety_penalty,
    inter_uav_safety_penalty_per_agent,
    point_in_tetrahedron_3d,
    barycentric_in_tetrahedron_3d,
    batch_points_in_tetrahedron_3d,
    one_hot_stage,
    clip_actions_to_bounds,
)


class UAVEnv_MASAC_PID:
    def __init__(self, length=2, num_obstacle=3, num_agents=4):
        """
        PID 控制器版本的追捕环境。

        环境内部包含：`num_agents` 个追捕者 + 1 个目标无人机。
        `reset_()` 负责生成随机地图/初始状态；`step_still()` 负责将追捕者期望子目标
        投影到可行域后，通过控制器推进仿真，并计算奖励/阶段标记。
        """
        # map

        self.max_episode_steps = 550
        self.map_size = 100
        self.obstacle_height = 42
        # uav
        self.num_agents = num_agents
        self.radius = 20 # lasers radius
        self.n_points = 48
        self.v_max = 4
        self.v_max_e = 2  # target
        # self.agent_plan_end_l = -8
        self.target_plan_end_l = -7
        # self.agent_plan_l = 15
        self.target_plan_l = 10
        self.last_d2target = 0
        self.last_d2targets = np.zeros(self.num_agents + 1, dtype=float)
        self.last_quad2target = np.zeros((self.num_agents, 3), dtype=float)
        self.last_action = np.zeros((self.num_agents, 3), dtype=float)

        self.post_capture_max = 10  # 围捕成功后需要额外保持的步数
        self.post_capture_step = 0  # 已保持了多少步
        self.is_holding_capture = False  # 是否已经进入“保持围捕阶段”

        self.time_step = 0.1  # update time step

        self.info = np.random.get_state()  # get seed
        self.stage = 0

        self.history_positions = [[] for _ in range(num_agents + 1)]
        self.history_goal = [[] for _ in range(num_agents)]
        self.quads_laser = np.full(self.num_agents + 1, self.radius, dtype=float)

        self.quads = []
        self.controllers = []
        self.agent_num = num_agents
        self.obs_dim = 3 * self.num_agents + 6 + 6 + 5
        self.critic_obs_dim = 15 * self.num_agents+1

        for i in range(self.num_agents + 1):
            if i != self.num_agents:  # if not target
                # self.multi_current_pos.append(np.random.uniform(low=-0.1, high=0.1, size=(2,)) + np.array([1.4, 1.9]))
                # self.multi_current_vel.append(np.zeros(2))
                self.quads.append(Quadrotor(max_vel = self.v_max))
                self.controllers.append(QuadrotorControllerXYZPos(self.quads[i]))
            else:  # for target
                self.quads.append(Quadrotor(max_vel = self.v_max_e))
                self.controllers.append(Controller(self.quads[i], self.target_plan_l, self.target_plan_end_l, dt=self.time_step))

        # collision
        self.dirs = Quadrotor._fibonacci_sphere(48)
        self.groups, _ = bin_dirs_into_16(self.dirs)

    def reset_(self, episode, target_vel=None):
        """
        初始化一个新 episode：
        - 更新随机种子（保证同 episode 可复现实验轨迹）
        - 采样追捕者与目标的初始位置/目标
        - 创建地图障碍物与可视化坐标轴（实时显示或渲染输出模式）

        Returns:
            total_obs, critic_obs, planes, start_
        """
        # SEED = random.randint(1,1000)
        episode += 75
        if target_vel is None:
            target_vel = [0, 0]

        self.last_d2target = 0
        self.last_quad2target = np.zeros((self.num_agents, 3), dtype=float)
        self.last_d2targets = np.zeros(self.num_agents + 1)
        self.history_positions = [[] for _ in range(self.num_agents + 1)]
        self.last_action = np.zeros((self.num_agents, 3), dtype=float)

        self.post_capture_step = 0
        self.is_holding_capture = False
        self.capture_ref_radius = None
        self.stage = 0

        SEED = 1024 + episode
        random.seed(SEED)
        np.random.seed(SEED)
        self.start = np.random.uniform(
            low=[0, 0, 0],
            high=[100, 100, 50],
            size=(self.num_agents, 3)
        )
        target_start, self.target_goal = sample_start_goal_3d()
        # target_start = np.array([80, 10, 50])
        start_ = np.vstack([self.start, target_start])
        
        for i in range(self.num_agents + 1):
            if i != self.num_agents:  # if not target
                # self.multi_current_pos.append(np.random.uniform(low=-0.1, high=0.1, size=(2,)) + np.array([1.4, 1.9]))
                # self.multi_current_vel.append(np.zeros(2))
                self.quads[i].reset(start_[i],max_vel = self.v_max)
                self.last_quad2target[i] = target_start - start_[i]
                self.last_d2targets[i] = np.linalg.norm(self.last_quad2target[i])
                self.last_d2target +=np.linalg.norm(self.last_quad2target[i])
            else:  # for target
                self.last_d2targets[i] = np.linalg.norm(target_start - (start_[0]+start_[1]+start_[2]+start_[3])/4)
                self.quads[i].reset(target_start,max_vel = self.v_max_e)

        bounds = np.array([0, self.map_size])
        # 如果启用实时显示，需要复用已有的 figure 避免创建多个窗口
        if enable_realtime and hasattr(self, 'ax') and self.ax is not None:
            # 复用已有的 figure 和 ax
            fig = self.ax.figure
            # 清除之前的内容
            self.ax.clear()
            # 重新绘制地图
            from envs.random_obstacle_map import generate_map
            self.mapobs, obstacles = generate_map(bounds, 2.6, self.obstacle_height, start_, self.target_goal, 250, seed_num=42+episode)
            self.ax.set_xlim(bounds[0], bounds[1])
            self.ax.set_ylim(bounds[0], bounds[1])
            self.ax.set_zlim(bounds[0], bounds[1])
            self.ax.set_box_aspect((1, 1, 1))
            self.mapobs.plotobs(self.ax, scale=1)
            self.ax.scatter(start_[:-1, 0], start_[:-1, 1], start_[:-1, 2], s=80, c='g', marker='o', label='Pursuer initial positions')
            self.ax.scatter(start_[-1, 0], start_[-1, 1], start_[-1, 2], s=80, c='r', marker='o', label='Target initial position')
            self.ax.scatter(self.target_goal[0], self.target_goal[1], self.target_goal[2], s=120, c='orange', marker='*', label='Target goal position')
            self.ax.legend(loc='upper left', bbox_to_anchor=(0, 0.90), fontsize=14)
            self.ax.zaxis.set_visible(False)
            self.ax.xaxis.pane.set_visible(False)
            self.ax.yaxis.pane.set_visible(False)
            self.ax.grid(False)
            self.ax.view_init(elev=89, azim=45)
            # 减少上方空白：直接设置 axes 位置
            self.ax.set_position([0.01, 0.05, 0.99, 0.99])
            # 重新初始化 sim3D（复用已有的 ax）
            if hasattr(self, 'sim3D') and self.sim3D is not None:
                # 清除之前的线条
                self.sim3D.agent_lines = []
            else:
                self.sim3D = QuadSim3D(self.time_step)
            self.sim3D.init_plot(self.ax)
            for _ in range(self.num_agents + 1):
                self.sim3D.add_agent_plot()
        else:
            # 非实时显示模式，正常创建新窗口（或关闭）
            display_flag = enable_realtime
            self.ax, self.mapobs = showmap(bounds, self.obstacle_height, start_, self.target_goal, episode, 2.6, display=display_flag)
            # render
            self.sim3D = QuadSim3D(self.time_step)
            self.sim3D.init_plot(self.ax)
            for _ in range(self.num_agents + 1):
                self.sim3D.add_agent_plot()
        
        # multi_obs is list of agent_obs, state is multi_obs after flattenned
        total_obs, critic_obs, planes = self.get_multi_obs(0,tag=1)
        
        # 如果启用实时显示，确保图形窗口显示（不阻塞）
        if enable_realtime and self.ax is not None:
            plt.show(block=False)
            self.ax.figure.canvas.draw()
            self.ax.figure.canvas.flush_events()

        return total_obs, critic_obs, planes, start_

    def step_still(self, raw_action, actions, planes, time_step, evaluate = False):
        """
        PID 动力学一步更新（“still”版本）。

        参数含义：
        - `raw_action`：原始动作（用于奖励中的平滑/误差项等）。
        - `actions`：追捕者期望子目标位置；会 clip 到边界并投影到可行域 `planes[i]`。
        - `planes`：上一时刻得到的可行域半空间（每个追捕者一组 planes）。
        - `time_step`：当前步数（用于 `t_norm` 归一化时间输入）。
        - `evaluate`：打开时会输出调试信息。

        Returns:
            total_obs, critic_obs, planes_, rewards, collided, agent_states, success, tag
        """
        agent_states = []
        # print(actions)
        # time.sleep(0.1)
        for i in range(self.num_agents):
            subgoal = project_goal_to_planes(
                actions[i],
                planes[i],
                bounds=(0, self.map_size),  # 或 (xmin,xmax,ymin,ymax) 视你的地图而定
                z_bounds=(0.0, 90.0),
                iters=20
            )
            thrust = self.controllers[i].step(subgoal)
            self.quads[i].update(thrust, dt=self.time_step)
            self.history_positions[i].append(self.quads[i].pos.copy())
            agent_states.append(self.quads[i].pos.copy())

        x0 = np.concatenate(self.quads[-1].get_state())
        subgoal = project_goal_to_planes(
            self.target_goal,
            planes[-1],
            bounds=(0, self.map_size),  # 或 (xmin,xmax,ymin,ymax) 视你的地图而定
            z_bounds=(0.0, 90.0),
            iters=20
        )
        x_ref = self.target_goal + 0.8 * (subgoal - self.target_goal)
        thrust = self.controllers[-1].compute_control_signal(x0, x_ref, planes[-1])
        self.quads[-1].update(thrust, dt=self.time_step)
        self.history_positions[-1].append(self.quads[-1].pos.copy())

        collided = self.update_isCollied_wrapper()

        # calculate rewards
        rewards, success, tag = self.cal_rewards_dones_(raw_action, collided, evaluate)
        self.last_action = raw_action
        total_obs, critic_obs, planes_ = self.get_multi_obs(time_step,tag)
        # sequence above can't be disrupted


        return total_obs, critic_obs, planes_, rewards, collided, agent_states, success, tag

    #1
    def get_multi_obs(self, time_step, tag):
        """
        构造观测与 critic 输入。

        对每个 UAV（追捕者 + 目标）拼接：
        - 归一化位置/速度、自身与团队/目标的相对几何
        - 追捕者的激光前半球扫描结果及其派生的“安全相关观测”
        - `t_norm`（时间归一化）和阶段 `tag` 的 one-hot

        返回：`total_obs`（每个追捕者一个观测向量列表）、`critic_obs`、以及每个追捕者对应的可行域 `planes`。
        """
        total_obs = []
        critic_obs = []
        planes = []
        pos_target = self.quads[-1].pos # 目标无人机的位置
        t_norm = time_step / self.max_episode_steps
        tag_oh = one_hot_stage(tag)
        for i in range(self.num_agents + 1):
            pos = self.quads[i].pos
            vel = self.quads[i].vel
            S_uavi = [
                pos[0] / self.map_size,
                pos[1] / self.map_size,
                pos[2] / self.map_size,
                vel[0] / self.v_max,
                vel[1] / self.v_max,
                vel[2] / self.v_max
            ]
            if i != self.num_agents:
                S_uavi.extend(self.last_action[i])

            S_team = []
            S_target = []
            S_last_target = []
            for j in range(self.num_agents + 1):
                if j != i and j != self.num_agents :
                    pos_d = self.quads[j].pos - pos_target # 其余与目标无人机的位置差
                    S_team.extend([ pos_d[0] / self.map_size,
                                    pos_d[1] / self.map_size,
                                    pos_d[2] / self.map_size] )
                elif j == self.num_agents:
                    pos_target_d = self.quads[j].pos - pos  # 与目标无人机的位置差
                    S_target.extend([pos_target_d[0] / self.map_size,
                                     pos_target_d[1] / self.map_size,
                                     pos_target_d[2] / self.map_size,
                                     ])
                    if i != self.num_agents:
                        delta = pos_target_d - self.last_quad2target[i]
                        d_S_last_target = delta / (np.linalg.norm(delta) + 1e-6)
                        S_last_target.extend(d_S_last_target.tolist())
                        # 更新“上一拍”向量缓存为当前测得的目标相对位移
                        self.last_quad2target[i] = pos_target_d

            scan = self.quads[i].scan_forward_hemisphere(self.mapobs, self.radius, self.n_points, self.dirs)
            plane, scan_ = get_collision_plane(self.groups, scan, self.radius, l = 1)
            if i != self.num_agents:
                observation = scan_["end_point"] - pos # (16, 3)

                idx_small4 = np.argsort(np.linalg.norm(observation, axis=1))[:8]
                small4 = observation[idx_small4] / self.map_size
                small1 = observation[idx_small4[0]]
                self.quads_laser[i] = np.linalg.norm(small1)
                small1 = small1 / self.map_size

                _single_obs = np.concatenate([
                    np.asarray(S_uavi).ravel(),
                    np.asarray(S_team).ravel(),
                    np.asarray(S_target).ravel(),
                    np.asarray(S_last_target).ravel(),
                    np.asarray(small4).ravel(),
                    np.asarray([t_norm]),
                    tag_oh
                ])
                total_obs.append(_single_obs)

            planes.append(plane)

        critic_obs.append(t_norm)
        critic_obs.append(tag_oh)

        return total_obs, critic_obs, planes

    def cal_rewards_dones_(self, actions, IsCollied, evaluate=False):
        """
        计算 rewards、stage tag 与成功条件。

        奖励由多部分组成（距离进展、碰撞/安全惩罚、编队几何误差等），并且包含多阶段逻辑：
        - track：向目标方向推进
        - encircle：形成期望几何关系（四面体 pairwise cos 等约束）
        - capture：接近捕获半径触发保持阶段
        - hold：保持围捕一段时间后判定 success=True

        Returns:
            rewards (np.ndarray): shape=(num_agents,)
            success (bool)
            tag (int): 1..4
        """
        success = False
        rewards = np.zeros(self.num_agents)

        mu1 = 0.3  # r_near 0.7
        mu2 = 2  # r_safe
        mu3 = 0.3  # r_multi_stage 0.1
        mu4 = 2  # r_finish
        m5 = 0.6
        d_capture = 16
        d_limit = 20
        d_long = 14
        d_safe = 6

        p = [self.quads[i].pos for i in range(self.num_agents)]
        pe = self.quads[-1].pos

        dists = np.array([np.linalg.norm(pi - pe) for pi in p], dtype=float)
        p_center = np.mean(p, axis=0)
        d_cg = np.linalg.norm(p_center - pe)
        Sum_d = dists.sum()
        d__ = np.concatenate([dists, [d_cg]])
        dist_arr = dists.copy()
        mask = (dist_arr > d_long).astype(float)

        r_near = (self.last_d2targets - d__) * 3.5

        vel_proj = np.zeros(self.num_agents)
        r_d_safe = m5 * inter_uav_safety_penalty_per_agent([self.quads[i].pos for i in range(self.num_agents + 1)], d_safe=d_safe)
        for i in range(self.num_agents):  # 计算飞行方向与目标方向的奖励
            pos = self.quads[i].pos
            vel = self.quads[i].vel
            dire_vec = pe - pos
            d = np.linalg.norm(dire_vec)  # distance to target
            vel_proj[i] = np.dot(vel, dire_vec) / (self.v_max * d + 1e-5)
            action_err = np.linalg.norm(actions[i] - self.last_action[i])

            rewards[i] += mu1 * r_near[i] * mask[i] - action_err * 0.2 - 0.2
            # approximate 1 ~ 2
            r_safe = 0
            if IsCollied[i]:
                r_safe = -10
            elif self.quads_laser[i] < 1:
                r_safe = - (1 - self.quads_laser[i])

            rewards[i] += mu2 * r_safe + r_d_safe[i]
            if evaluate:
                print(rewards[i])

        encircle, d_v = point_in_tetrahedron_3d(pe, *p)

        # 3.2 stage-1 track

        if self.is_holding_capture:
            tol = 0.2  # 5% 容忍
            # err = np.maximum(np.abs(dist_arr / d_long - 1) - tol, 0.0)
            err = np.maximum(dist_arr / d_long - 1, 0.0)
            r_dists = np.tanh(2 * err) / 2
            r_center = np.clip(- d_cg / d_capture, -1, 0)

            eps = 1e-6
            unit_vecs = []
            for i in range(self.num_agents):
                if dist_arr[i] > eps:
                    unit_vecs.append((p[i] - pe) / (dist_arr[i] + eps))
                else:
                    # 万一重合，就给一个默认值
                    unit_vecs.append(np.array([1.0, 0.0, 0.0]))
            unit_vecs = np.stack(unit_vecs, axis=0)  # (4,3)

            cos_ij_list = []
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    cos_ij = np.dot(unit_vecs[i], unit_vecs[j])
                    cos_ij_list.append(cos_ij)

            cos_ij_arr = np.array(cos_ij_list, dtype=float)
            target_cos = -1.0 / 3.0  # 规则四面体 pairwise cos
            geom_err = np.mean((cos_ij_arr - target_cos) ** 2)  # 越小越好

            # 奖励：几何误差越小越好，压到 [-1,0]
            r_geom = -geom_err
            r_geom = float(np.clip(r_geom, -1.0, 0.0))

            cos_i_pe = []
            for i in range(self.num_agents):
                eps = 1e-6
                p_others_center = (4.0 * p_center - p[i]) / 3.0  # 其余三机中心
                v = p_others_center - pe  # 目标 -> 其余三机中心
                p3_center = v / (np.linalg.norm(v) + eps)
                cos_i_pe.append(np.clip(np.dot(p3_center, pe - p[i]) / (dist_arr[i] + eps), -0.5, 0.5))
            cos_i_pe = np.array(cos_i_pe, dtype=float)

            r_capture = 3 * r_near[-1] + r_near[:self.num_agents] * mask.astype(
                float) + 1.2 + cos_i_pe - r_dists + r_center  # 100?
            # r_encircle = - np.linalg.norm((p0 + p1 + p2)/3 - pe)/2 # approximate -0.1
            rewards[0:self.num_agents] += mu3 * r_capture

            self.post_capture_step += 1
            tag = 4

            if evaluate:
                print(4)
                print("r_geom:", r_geom)
                print("cos_i_pe", cos_i_pe)
                print("r_center:", r_center)
                print(rewards[0:self.num_agents])

            if self.post_capture_step >= self.post_capture_max:
                success = True

        elif d_cg >= d_limit:
            eps = 1e-6
            cos_i_pe = []
            for i in range(self.num_agents):
                eps = 1e-6
                p_others_center = (4.0 * p_center - p[i]) / 3.0  # 其余三机中心
                v = p_others_center - pe  # 目标 -> 其余三机中心
                p3_center = v / (np.linalg.norm(v) + eps)
                cos_i_pe.append(np.clip(np.dot(p3_center, pe - p[i]) / (dist_arr[i] + eps), -0.5, 0.5))
            cos_i_pe = np.array(cos_i_pe, dtype=float)
            # r_track = (d_limit - Sum_d) / self.map_size /4  # approximate -0.5
            rewards[0:self.num_agents] += mu3 * (r_near[:self.num_agents] * mask.astype(float) + cos_i_pe)
            tag = 1
            if evaluate:
                print(1)
                print(rewards[0:self.num_agents])
        # 3.3 stage-2 encircle
        elif not encircle:
            # r_dists = np.clip(dist_arr / d_long - 1, 0, 1) / 2
            tol = 0.2  # 5% 容忍
            err = np.maximum(dist_arr / d_long - 1, 0.0)
            r_dists = np.tanh(2 * err) / 2

            eps = 1e-6
            unit_vecs = []
            for i in range(self.num_agents):
                if dist_arr[i] > eps:
                    unit_vecs.append((p[i] - pe) / (dist_arr[i] + eps))
                else:
                    # 万一重合，就给一个默认值
                    unit_vecs.append(np.array([1.0, 0.0, 0.0]))
            unit_vecs = np.stack(unit_vecs, axis=0)  # (4,3)

            cos_ij_list = []
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    cos_ij = np.dot(unit_vecs[i], unit_vecs[j])
                    cos_ij_list.append(cos_ij)

            cos_ij_arr = np.array(cos_ij_list, dtype=float)
            target_cos = -1.0 / 3.0  # 规则四面体 pairwise cos
            geom_err = np.mean((cos_ij_arr - target_cos) ** 2)  # 越小越好

            # 奖励：几何误差越小越好，压到 [-1,0]
            r_geom = -geom_err
            r_geom = float(np.clip(r_geom, -1.0, 0.0))

            cos_i_pe = []
            for i in range(self.num_agents):
                eps = 1e-6
                p_others_center = (4.0 * p_center - p[i]) / 3.0  # 其余三机中心
                v = p_others_center - pe  # 目标 -> 其余三机中心
                p3_center = v / (np.linalg.norm(v) + eps)
                cos_i_pe.append(np.clip(np.dot(p3_center, pe - p[i]) / (dist_arr[i] + eps), -0.5, 0.5))
            cos_i_pe = np.array(cos_i_pe, dtype=float)

            r_encircle = 3 * r_near[-1] + r_near[:self.num_agents] * mask.astype(
                float) + cos_i_pe + 0.2 - r_dists  # 100?
            # r_encircle = - np.linalg.norm((p0 + p1 + p2)/3 - pe)/2 # approximate -0.1
            rewards[0:self.num_agents] += mu3 * r_encircle
            if self.stage == 0:
                rewards[0:self.num_agents] += 4
                self.stage = 1
            tag = 2
            if evaluate:
                print(2)
                print("r_geom:", r_geom)
                print("cos_i_pe", cos_i_pe)
                print(rewards[0:self.num_agents])

        # 3.4 stage-3 capture
        elif (dist_arr > d_capture).any():
            tol = 0.2  # 5% 容忍
            err = np.maximum(dist_arr / d_long - 1, 0.0)
            r_dists = np.tanh(2 * err) / 2
            r_center = np.clip(- d_cg / d_capture, -1, 0)

            eps = 1e-6
            unit_vecs = []
            for i in range(self.num_agents):
                if dist_arr[i] > eps:
                    unit_vecs.append((p[i] - pe) / (dist_arr[i] + eps))
                else:
                    # 万一重合，就给一个默认值
                    unit_vecs.append(np.array([1.0, 0.0, 0.0]))
            unit_vecs = np.stack(unit_vecs, axis=0)  # (4,3)

            cos_ij_list = []
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    cos_ij = np.dot(unit_vecs[i], unit_vecs[j])
                    cos_ij_list.append(cos_ij)

            cos_ij_arr = np.array(cos_ij_list, dtype=float)
            target_cos = -1.0 / 3.0  # 规则四面体 pairwise cos
            geom_err = np.mean((cos_ij_arr - target_cos) ** 2)  # 越小越好

            # 奖励：几何误差越小越好，压到 [-1,0]
            r_geom = -geom_err
            r_geom = float(np.clip(r_geom, -1.0, 0.0))

            cos_i_pe = []
            for i in range(self.num_agents):
                eps = 1e-6
                p_others_center = (4.0 * p_center - p[i]) / 3.0  # 其余三机中心
                v = p_others_center - pe  # 目标 -> 其余三机中心
                p3_center = v / (np.linalg.norm(v) + eps)
                cos_i_pe.append(np.clip(np.dot(p3_center, pe - p[i]) / (dist_arr[i] + eps), -0.5, 0.5))
            cos_i_pe = np.array(cos_i_pe, dtype=float)

            r_capture = 3 * r_near[-1] + r_near[:self.num_agents] * mask.astype(
                float) + 1.2 + cos_i_pe - r_dists + r_center  # 100?
            # r_encircle = - np.linalg.norm((p0 + p1 + p2)/3 - pe)/2 # approximate -0.1
            rewards[0:self.num_agents] += mu3 * r_capture
            if self.stage <= 1:
                rewards[0:self.num_agents] += 4
                self.stage = 2
            tag = 3
            if evaluate:
                print(3)
                print("r_geom:", r_geom)
                print("cos_i_pe", cos_i_pe)
                print("r_center:", r_center)
                print(rewards[0:self.num_agents])

        else:
            r_center = np.clip(- d_cg / d_capture, -1, 0)


            eps = 1e-6
            unit_vecs = []
            for i in range(self.num_agents):
                if dist_arr[i] > eps:
                    unit_vecs.append((p[i] - pe) / (dist_arr[i] + eps))
                else:
                    unit_vecs.append(np.array([1.0, 0.0, 0.0]))
            unit_vecs = np.stack(unit_vecs, axis=0)

            cos_ij_list = []
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    cos_ij_list.append(np.dot(unit_vecs[i], unit_vecs[j]))
            cos_ij_arr = np.array(cos_ij_list, dtype=float)
            target_cos = -1.0 / 3.0
            geom_err = np.mean((cos_ij_arr - target_cos) ** 2)
            r_geom_hold = -geom_err
            r_geom_hold = float(np.clip(r_geom_hold, -1.0, 0.0)) * 2

            r_hold = 7 \
                     + 1.0 * r_center

            rewards[0:self.num_agents] += mu4 * r_hold
            if not self.is_holding_capture:
                # 记录第一次成功围捕时的平均半径，用作后面 5 步的参考
                self.is_holding_capture = True
                self.post_capture_step = 0

            print("capture")
            tag = 4
            if evaluate:
                print(rewards[0:self.num_agents])

        self.last_d2targets = d__
        self.last_d2target = Sum_d

        return rewards, success, tag

    def update_isCollied_wrapper(self):
        """碰撞/越界判定。

        若追捕者 UAV 越界或位于障碍物体素内，则视为碰撞（failed agent）。
        """
        inside_flags = []
        for i in range(self.num_agents):
            p = self.quads[i].pos
            # 取前三个坐标并转为 float（兼容 list / numpy / torch 标量）
            x, y, z = float(p[0]), float(p[1]), float(p[2])

            # A) 边界检查：只要有一维不在 [-10, 110] 就算碰撞
            out_of_bounds = (x < -5.0) or (x > 105.0) or \
                            (y < -5.0) or (y > 105.0) or \
                            (z < -5.0) or (z > 105.0)

            # B) 障碍物检查：沿用你现有的索引判断
            inside_obstacle = self.mapobs.idx.count((x, y, z, x, y, z)) > 0

            # C) 合并结果：在障碍物内 或 越界 都算碰撞
            inside_flags.append(inside_obstacle or out_of_bounds)

        return inside_flags


    def render(self):
        """
        返回当前三维画面的 RGBA 图像（仅画无人机与轨迹）。

        行为与 `UAVEnv_MASAC_MPC.render()` 保持一致，便于评估/采样阶段生成对比可视化。
        """
        frames = []
        for j in range(self.num_agents + 1):
            frames.append(self.quads[j].world_frame())

        # 更新无人机几何与轨迹
        self.sim3D.update_plot(frames, self.history_positions)

        # 强制重绘当前 figure，并取出 RGBA 缓冲区
        fig = self.sim3D.ax.figure
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()

        # 转成独立的 ndarray 返回
        image = np.asarray(buf).copy()  # H x W x 4 (RGBA, uint8)
        return image
    
    def render_goal(self, goal):
        """在渲染画面上叠加追捕者的期望子目标点（星形标记）。"""

        goal = np.asarray(goal, dtype=float)
        frames = []
        colors = ['b', 'g', 'y', 'c', 'r', 'm', 'k']
        ax = self.sim3D.ax
        if hasattr(self, "_goal_scatters"):
            for h in self._goal_scatters:
                h.remove()
        self._goal_scatters = []

        for j in range(self.num_agents + 1):
            frames.append(self.quads[j].world_frame())
            if j != self.num_agents:
                sc = ax.scatter(
                    goal[j, 0],
                    goal[j, 1],
                    goal[j, 2],
                    c=colors[j],
                    marker='*',
                    s=80,
                    label=f'Pid subgoal {j}'
                )
                self._goal_scatters.append(sc)

        # 确保目标点的图例可见；如已有旧图例则更新
        if len(self._goal_scatters) > 0:
            handles, labels = ax.get_legend_handles_labels()
            # 去重，避免多次调用 render_goal 堆叠旧标签
            uniq = {}
            for h, l in zip(handles, labels):
                uniq[l] = h
            ax.legend(uniq.values(), uniq.keys(), loc='upper left')

        self.sim3D.update_plot(frames, self.history_positions)
        # 强制重绘当前 figure
        fig = ax.figure
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()

        # 4) 转 ndarray（保证独立内存）
        image = np.asarray(buf).copy()  # H x W x 4 (RGBA, uint8)
        return image



