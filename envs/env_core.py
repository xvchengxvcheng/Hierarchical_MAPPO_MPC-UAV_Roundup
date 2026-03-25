import numpy as np
import time
from PIL import Image
import random
from envs.Quadrotor import *
from envs.sim_env_pid import UAVEnv_MASAC_PID
from envs.sim_env_mpc import UAVEnv_MASAC_MPC
import os
import json
# from control.control_mpc import Controller
import matplotlib
import matplotlib.pyplot as plt

# 通过环境变量判断是否启用实时显示（单并行环境时）
enable_realtime = os.environ.get('ENABLE_REALTIME_DISPLAY', 'False').lower() == 'true'
if enable_realtime:
    matplotlib.use('TkAgg')  # 使用交互式后端
    plt.ion()  # 开启交互模式
else:
    matplotlib.use('Agg')  # 非交互式后端，只保存图片

test_time = 81


def save_image(env_render, filename):
    # Convert the RGBA buffer to an RGB image
    image = Image.fromarray(env_render, 'RGBA')  # Use 'RGBA' mode since the buffer includes transparency
    image = image.convert('RGB')  # Convert to 'RGB' if you don't need transparency
    image.save(filename)


class EnvCore_(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        self.agent_num = 4  # 设置智能体(小飞机)的个数，这里设置为两个 # set the number of agents(aircrafts), here set to two
        self.obs_dim = 14  # 设置智能体的观测维度 # set the observation dimension of agents
        self.action_dim = 3  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(14,))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            sub_agent_obs.append(np.random.random(size=(14,)))
            sub_agent_reward.append([np.random.rand()])
            sub_agent_done.append(False)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]


class EnvCore(object):
    """
    MAPPO 用的"核心环境"包装：

    - 对外暴露接口和 on-policy 代码里的 EnvCore 完全一致：
        * 属性: agent_num, obs_dim, action_dim
        * 方法: reset(), step(actions)
    - 内部可以用 UAVEnv_MASAC_PID 或 UAVEnv_MASAC_MPC 做仿真和奖励
    """

    def __init__(self, num_agents: int = 4, episode_offset: int = 0, enable_realtime_display: bool = None, controller_type: str = 'PID'):
        """
        Args:
            num_agents: 智能体数量
            episode_offset: episode起始偏移
            enable_realtime_display: 是否启用实时显示
            controller_type: 控制器类型，'pid' 或 'mpc'，默认为 'pid'
        """
        # 根据controller_type选择底层环境
        self.controller_type = controller_type.lower()
        if self.controller_type == 'pid':
            self._env = UAVEnv_MASAC_PID(num_agents=num_agents)
            self._use_step_still = True  # PID使用step_still方法
        elif self.controller_type == 'mpc':
            self._env = UAVEnv_MASAC_MPC(num_agents=num_agents)
            self._use_step_still = False  # MPC使用step方法
        else:
            raise ValueError(f"controller_type must be 'pid' or 'mpc', got '{controller_type}'")

        # 判断是否启用实时显示：优先使用参数，否则使用环境变量
        if enable_realtime_display is None:
            self._enable_realtime_display = enable_realtime
        else:
            self._enable_realtime_display = enable_realtime_display

        # 给 ContinuousActionEnv / Runner 用的三个关键属性
        self.agent_num = self._env.num_agents  # 智能体数量
        self.action_dim = 3  # 每个智能体 3 维连续动作（x,y,z 方向的子目标）

        # 为了得到 obs_dim，做一次"哑初始化"，episode 从 offset 开始
        self._episode = episode_offset
        total_obs, _, planes, _ = self._env.reset_(episode=self._episode)
        # total_obs 是长度为 agent_num 的 list，每个元素是一维观测向量
        self.obs_dim = len(total_obs[0])
        self.action_scale = np.array([self._env.map_size / 4,
                                      self._env.map_size / 4,
                                      self._env.map_size / 4], dtype=np.float32)

        # 记录当前几何约束平面 & 时间步，用于后续 step_still 调用
        self._planes = planes
        self._t = 0

        # 记录当前episode是否撞毁（失败） / Record whether current episode has collision (failure)
        self._episode_collided = False

        # 撞毁记录文件路径（JSON格式） / Collision record file path (JSON format)
        controller_suffix = self.controller_type.upper()
        self._collision_record_file = f"collision_records_{controller_suffix}_{test_time}.json"
        self._collision_record_format = "json"  # 支持 "json" 或 "csv" / Support "json" or "csv"

        # 实时显示渲染计数器：每 N 步渲染一次（默认每2步）
        self._render_step_counter = 0
        self._render_interval = 2  # 每2步渲染一次

        # 存储成功时绘制的平面句柄（用于清除旧平面）
        self._success_planes = []
        
        # 存储subgoal散点句柄（用于实时显示时绘制和清除）
        self._subgoal_scatters = []

        # 如果启用实时显示，确保matplotlib处于交互模式
        if self._enable_realtime_display:
            if not plt.isinteractive():
                plt.ion()
            # 确保图形窗口显示（不阻塞）
            if self._env.sim3D.ax is not None:
                plt.show(block=False)
                self._env.sim3D.ax.figure.canvas.draw()
                self._env.sim3D.ax.figure.canvas.flush_events()
                # 首次显示画面后暂停，便于观察初始场景（秒；0 关闭）
                try:
                    _pause = float(
                        os.environ.get("REALTIME_DISPLAY_START_PAUSE_SEC", "5")
                    )
                except ValueError:
                    _pause = 5.0
                if _pause > 0:
                    time.sleep(_pause)

    # 提供和原 EnvCore 相同签名的 reset()：
    #   返回： [obs_agent_0, obs_agent_1, ..., obs_agent_{N-1}]
    def reset(self, episode=None):
        # 每次 reset 认为是一个新 episode
        if episode is not None:
            self._episode = int(episode)
        else:
            self._episode += 1

        self._t = 0

        # 重置episode撞毁标志 / Reset episode collision flag
        self._episode_collided = False

        # 重置渲染计数器
        self._render_step_counter = 0

        # 清除之前绘制的成功平面和subgoal
        self._clear_success_planes()
        self._clear_subgoals()

        total_obs, critic_obs, planes, _ = self._env.reset_(episode=self._episode)
        self._planes = planes

        # 如果启用实时显示，确保图形窗口正确显示
        if self._enable_realtime_display and self._env.sim3D.ax is not None:
            if not plt.isinteractive():
                plt.ion()
            plt.show(block=False)
            self._env.sim3D.ax.figure.canvas.draw()
            self._env.sim3D.ax.figure.canvas.flush_events()

        # 转成 list[np.ndarray]，shape = (obs_dim,)
        obs_list = [np.asarray(o, dtype=np.float32) for o in total_obs]

        return obs_list

    # 提供和原 EnvCore 相同签名的 step(actions)：
    #   输入：actions 是 [agent_num, action_dim] 或等价结构
    #   输出： [obs_list, rew_list, done_list, info_list]
    def step(self, actions):
        # 转成 (agent_num, action_dim) 的 float32 数组
        actions = np.asarray(actions, dtype=np.float32)
        assert actions.shape[0] == self.agent_num, \
            f"expected actions for {self.agent_num} agents, got {actions.shape[0]}"

        # 原 MASAC 里，奖励里用的是“原始动作 raw_action (3 维)”，
        # 这里直接把 MAPPO 的动作当作 raw_action，用于动作平滑惩罚等。
        # raw_action = actions.copy()
        actions_clipped = np.clip(actions, -1.0, 1.0)
        raw_action = actions_clipped.copy()

        subgoals = []
        target_z = float(self._env.quads[-1].pos[2])

        for i in range(self.agent_num):
            cur_pos = np.asarray(self._env.quads[i].pos, dtype=np.float32).copy()  # 关键：copy，避免改写 env 内部状态
            # cur_pos[2] = target_z  # 让子目标的参考 z 与目标机对齐（绝对坐标）

            delta = actions_clipped[i] * self.action_scale
            subgoals.append(cur_pos + delta)

        # 调用底层环境的step方法（PID使用step_still，MPC使用step）：
        #   - raw_action: 3 维动作（用于奖励中的动作变化项）
        #   - actions:    3 维子目标位置（会被 project_goal_to_planes 做安全投影）
        #   - planes:     上一步 get_multi_obs 计算的半空间约束（长度 = agent_num+1）
        #   - time_step:  当前时间步（用于 t_norm）
        cond_episode = (self._episode % 20 == 0)  # 全局 episode 是 5 的倍数
        cond_step = ((self._t + 1) % 40 == 0)  # 第 10, 20, 30... 步

        eval_flag = cond_episode and cond_step
        # ==========================================================

        # 根据控制器类型调用不同的方法
        if self._use_step_still:
            # PID控制器使用step_still方法
            total_obs, _, planes_new, rewards, collided, _, success, tag = \
                self._env.step_still(
                    raw_action=raw_action,
                    actions=subgoals,
                    planes=self._planes,
                    time_step=self._t,
                    evaluate=eval_flag,  # 使用上述规则
                )
        else:
            # MPC控制器使用step方法
            total_obs, _, planes_new, rewards, collided, _, success, tag = \
                self._env.step(
                    raw_action=raw_action,
                    actions=subgoals,
                    planes=self._planes,
                    time_step=self._t,
                    evaluate=eval_flag,  # 使用上述规则
                )

        # 实时显示或保存图片
        if self._enable_realtime_display:
            # 实时显示模式：每 N 步更新一次3D图形（默认每2步）
            self._render_step_counter += 1
            should_render = (self._render_step_counter >= self._render_interval) or success
            
            if should_render:
                if self._render_step_counter >= self._render_interval:
                    self._render_step_counter = 0  # 重置计数器
                
                frames = []
                for j in range(self.agent_num + 1):
                    frames.append(self._env.quads[j].world_frame())
                
                self._env.sim3D.update_plot(frames, self._env.history_positions)
                
                # 如果成功，绘制四个围捕平面
                if tag == 4:
                    self._draw_success_planes()
                
                self._draw_subgoals(subgoals)
                
                # 强制重绘当前 figure
                if self._env.sim3D.ax is not None:
                    self._env.sim3D.ax.figure.canvas.draw()
                    self._env.sim3D.ax.figure.canvas.flush_events()
                    
                    # 给 UI 一点时间渲染
                    # 实时显示模式下使用更小的暂停时间，让动画更快
                    pause_time = self._env.sim3D.animation_rate
                    if self._enable_realtime_display:
                        # 实时显示时使用更小的暂停时间（最快0.001秒），让动画更流畅
                        pause_time = min(pause_time, 0.001)
                    plt.pause(pause_time)
            
            # 如果成功，保存图片（即使在实时显示模式下）
            if success:
                self._clear_subgoals()
                if self._env.sim3D.ax is not None:
                    self._env.sim3D.ax.figure.canvas.draw()
                    self._env.sim3D.ax.figure.canvas.flush_events()
                
                # 保存图片
                controller_suffix = self.controller_type.lower()
                os.makedirs(f"images_{test_time}_ppo_{controller_suffix}_capture", exist_ok=True)
                env_render = self._env.render()
                filename = f"images_{test_time}_ppo_{controller_suffix}_capture/episode_{self._episode}_step_{self._t}.png"
                save_image(env_render, filename)
        
        elif eval_flag or success:
            # 保存图片模式：只在特定条件下保存
            print(self._episode)
            print(self._t)
            controller_suffix = self.controller_type.lower()
            os.makedirs(f"images_{test_time}_ppo_{controller_suffix}", exist_ok=True)
            
            # 如果成功且tag == 4，绘制平面
            if success:
                self._draw_success_planes()
                # 重新绘制以确保平面在图像中
                if self._env.sim3D.ax is not None:
                    self._env.sim3D.ax.figure.canvas.draw()
                    self._env.sim3D.ax.figure.canvas.flush_events()
                env_render = self._env.render()
                filename = f"images_{test_time}_ppo_{controller_suffix}/episode_{self._episode}_step_{self._t}_success.png"
                save_image(env_render, filename)
            else:
                env_render = self._env.render_goal(subgoals)
                filename = f"images_{test_time}_ppo_{controller_suffix}/episode_{self._episode}_step_{self._t}.png"
                save_image(env_render, filename)

        # 更新内部状态
        self._planes = planes_new
        self._t += 1

        # 检测是否有任何无人机撞毁，如果有则标记episode为失败 / Check if any UAV collided, mark episode as failure if so
        if any(collided):
            self._episode_collided = True

        # 1) 观测：list, 每个元素 shape=(obs_dim,)
        obs_list = [np.asarray(o, dtype=np.float32) for o in total_obs]

        # 2) 奖励：保持和旧 EnvCore 一样，每个智能体是 [reward] 这种一维 list
        rew_list = [[float(r)] for r in rewards]

        # 3) done：这里简单处理成"所有智能体共用一个 done 标志"
        #    - 成功围捕 success=True
        #    - 或时间步达到最大步数
        done_flag = success or any(collided) or self._t >= self._env.max_episode_steps
        done_list = [done_flag] * self.agent_num

        # 4) info：可按需塞额外信息（碰撞/成功标志等）
        # 如果episode结束，保存撞毁记录 / Save collision record if episode ends
        if done_flag:
            self._save_collision_record(success)

        info_list = []
        for i in range(self.agent_num):
            info = {
                "success": bool(success),
                "step": self._t,
                # "episode_collided": self._episode_collided,  # Episode是否撞毁（失败） / Whether episode has collision (failure)
            }
            info_list.append(info)

        return [obs_list, rew_list, done_list, info_list]

    # 供 env_continuous 调用的 seed 接口
    def seed(self, seed: int):
        np.random.seed(seed)
        random.seed(seed)

    def _draw_success_planes(self):
        """
        绘制四架围捕无人机中任意三架所构成的四个平面
        四个组合：(0,1,2), (0,1,3), (0,2,3), (1,2,3)
        """
        if not hasattr(self._env, 'sim3D') or self._env.sim3D is None or self._env.sim3D.ax is None:
            return
        
        # 清除之前绘制的平面
        self._clear_success_planes()
        
        # 获取四架围捕无人机的位置（索引0-3）
        positions = []
        for i in range(self.agent_num):  # agent_num = 4
            pos = np.array(self._env.quads[i].pos)
            positions.append(pos)
        
        # 四个组合：C(4,3) = 4
        combinations = [
            (0, 1, 2),  # 排除第3架
            (0, 1, 3),  # 排除第2架
            (0, 2, 3),  # 排除第1架
            (1, 2, 3),  # 排除第0架
        ]
        
        # 平面颜色（半透明）
        plane_colors = ['red', 'red', 'red', 'red']
        plane_alpha = 0.3
        
        for idx, (i, j, k) in enumerate(combinations):
            p1 = positions[i]
            p2 = positions[j]
            p3 = positions[k]
            
            # 使用 plot_trisurf 绘制三角形平面
            # 创建三个点的坐标数组
            x = np.array([p1[0], p2[0], p3[0]])
            y = np.array([p1[1], p2[1], p3[1]])
            z = np.array([p1[2], p2[2], p3[2]])
            
            # 绘制三角形平面
            tri_surf = self._env.sim3D.ax.plot_trisurf(
                x, y, z,
                color=plane_colors[idx % len(plane_colors)],
                alpha=plane_alpha,
                linewidth=0,
                edgecolor='none'
            )
            self._success_planes.append(tri_surf)
    
    def _clear_success_planes(self):
        """清除之前绘制的成功平面"""
        if not hasattr(self._env, 'sim3D') or self._env.sim3D is None or self._env.sim3D.ax is None:
            return
        
        for plane in self._success_planes:
            try:
                plane.remove()
            except:
                pass
        
        self._success_planes = []
    
    def _draw_subgoals(self, subgoals):
        """
        绘制各无人机的subgoal（子目标）
        Args:
            subgoals: 列表，包含每个智能体的子目标位置（不包括目标无人机）
        """
        if not hasattr(self._env, 'sim3D') or self._env.sim3D is None or self._env.sim3D.ax is None:
            return
        
        # 清除之前的subgoal散点
        self._clear_subgoals()
        
        # 颜色列表，与render_goal保持一致
        colors = ['b', 'g', 'y', 'c', 'r', 'm', 'k']
        ax = self._env.sim3D.ax
        
        # 为每个智能体绘制subgoal（不包括目标无人机）
        for i in range(len(subgoals)):
            if i < len(subgoals):
                subgoal = np.asarray(subgoals[i], dtype=float)
                sc = ax.scatter(
                    subgoal[0],
                    subgoal[1],
                    subgoal[2],
                    c=colors[i % len(colors)],
                    marker='*',  
                    s=80,  
                    label=f'Subgoal {i}',
                    alpha=0.8
                )
                self._subgoal_scatters.append(sc)
    
    def _clear_subgoals(self):
        """清除之前绘制的subgoal散点"""
        if not hasattr(self._env, 'sim3D') or self._env.sim3D is None or self._env.sim3D.ax is None:
            return
        
        for scatter in self._subgoal_scatters:
            try:
                scatter.remove()
            except:
                pass
        
        self._subgoal_scatters = []

    def _save_collision_record(self, success: bool):
        """
        保存episode撞毁记录到文件（JSON格式） / Save episode collision record to file (JSON format)
        Args:
            success: 是否成功完成任务 / Whether task completed successfully
        """
        try:
            # 创建results目录（如果不存在） / Create results directory if not exists
            results_dir = "results_collision"
            os.makedirs(results_dir, exist_ok=True)

            # 文件路径 / File path
            filepath = os.path.join(results_dir, self._collision_record_file)

            # 构建记录字典 / Build record dictionary
            record = {
                "episode": int(self._episode),
                "collided": bool(self._episode_collided),
                "success": bool(success),
                "steps": int(self._t),
                "timestamp": time.time()  # 添加时间戳便于分析 / Add timestamp for analysis
            }

            # JSON格式：每行一个JSON对象（JSONL格式），便于追加写入和读取 / JSON format: one JSON object per line (JSONL), easy to append and read
            record_line = json.dumps(record, ensure_ascii=False) + "\n"

            # 追加写入文件 / Append to file
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(record_line)
        except Exception as e:
            # 如果保存失败，打印错误但不中断训练 / If save fails, print error but don't interrupt training
            print(f"Warning: Failed to save collision record: {e}")

