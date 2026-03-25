"""
绘制四个阶段（track / encircle / capture / hold）的静态 3D 围捕示意图。
- 障碍物：使用 generate_map(bounds, 2.5, obstacle_height, start_, goal_, 250, seed_num=42+episode) 生成“城市柱状体”障碍
- 无人机：给定 4 架 UAV 的位置/速度（可替换为你环境中的 quads[i].pos / quads[i].vel）
- 目标：给定 target_pos
- 输出：每个阶段一张图（PNG），并可选生成 2x2 汇总图

依赖：numpy, matplotlib
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from envs.random_obstacle_map import generate_map
from envs.Quadrotor import Quadrotor
from utils_ import euler_to_quaternion

# -----------------------------
# 1) 辅助函数：判断点是否在四面体内（用于验证阶段）
# -----------------------------
def rot_mat_to_quaternion(R):
    """
    将旋转矩阵转换为四元数 (w, x, y, z)。
    """
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s=4*qw
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s=4*qx
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s=4*qy
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s=4*qz
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])
def _vol6(a, b, c, d):
    """返回 6 倍有向体积 det([b-a, c-a, d-a])."""
    return np.linalg.det(np.stack([b - a, c - a, d - a], axis=1))

def point_in_tetrahedron_3d(P, A, B, C, D, tol=1e-9):
    """判断点 P 是否在四面体 ABCD 内（含边界）。"""
    P, A, B, C, D = map(lambda x: np.asarray(x, float).reshape(3), [P, A, B, C, D])
    V = np.abs(_vol6(A, B, C, D))
    if V < tol:
        return False, V / 6.0
    v0 = np.abs(_vol6(P, B, C, D))
    v1 = np.abs(_vol6(A, P, C, D))
    v2 = np.abs(_vol6(A, B, P, D))
    v3 = np.abs(_vol6(A, B, C, P))
    s = v0 + v1 + v2 + v3 - V
    close_sum = s <= tol * (V + 1.0)
    return close_sum, s / 6

# -----------------------------
# 2) 绘图工具：障碍物 / UAV / 速度箭头 / 围捕四面体
# -----------------------------


def set_axes_equal(ax):
    """让 3D 坐标轴等比例显示。"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



def compute_quaternion_from_velocity(pos, vel, target_pos=None):
    """
    根据速度方向计算四元数姿态。
    如果速度为零或很小，则使用指向目标的方向。
    """
    eps = 1e-6
    v_norm = np.linalg.norm(vel)
    
    if v_norm > eps:
        # 使用速度方向作为前进方向
        forward = vel / v_norm
    elif target_pos is not None:
        # 如果速度为零，使用指向目标的方向
        to_target = target_pos - pos
        to_target_norm = np.linalg.norm(to_target)
        if to_target_norm > eps:
            forward = to_target / to_target_norm
        else:
            forward = np.array([1.0, 0.0, 0.0])  # 默认方向
    else:
        forward = np.array([1.0, 0.0, 0.0])  # 默认方向
    
    # 将forward向量转换为四元数（假设机体的+x轴对齐到forward）
    # 计算一个合理的姿态：forward指向速度方向，up尽量保持为z方向
    up_world = np.array([0.0, 0.0, 1.0])
    
    # 如果forward和up平行，调整up
    if abs(np.dot(forward, up_world)) > 0.99:
        up_world = np.array([0.0, 1.0, 0.0])
    
    # 计算右向量和上向量
    right = np.cross(forward, up_world)
    right = right / (np.linalg.norm(right) + eps)
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + eps)
    
    # 构建旋转矩阵：body的+x -> forward, body的+y -> right, body的+z -> up
    R = np.array([
        [forward[0], right[0], up[0]],
        [forward[1], right[1], up[1]],
        [forward[2], right[2], up[2]]
    ])
    
    # 转换为四元数
    quat = rot_mat_to_quaternion(R)
    return quat


def draw_uavs_and_target(ax, uav_pos, uav_vel, target_pos,
                         uav_colors=("tab:blue", "tab:green", "tab:cyan", "tab:red"),
                         show_text=True, vel_scale=6.0, show_dist=False, stage_tag=None):
    """
    uav_pos: (4,3)
    uav_vel: (4,3)
    target_pos: (3,)
    show_dist: 是否显示到目标的距离
    stage_tag: 阶段标签 (1, 2, 3, 4)，用于确定电机散点大小
    使用四旋翼几何表示而不是圆点
    """
    uav_pos = np.asarray(uav_pos, float)
    uav_vel = np.asarray(uav_vel, float)
    target_pos = np.asarray(target_pos, float).reshape(3)

    # 目标
    ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]],
               marker="*", s=180, color="orange", edgecolor="k", linewidths=0.5, zorder=50)
    ax.text(target_pos[0], target_pos[1], target_pos[2] + 1.0, "Target",
            fontsize=10, zorder=60)

    # UAV - 使用四旋翼几何表示
    for i in range(4):
        p = uav_pos[i]
        v = uav_vel[i]
        
        # 根据速度和位置计算姿态
        quat = compute_quaternion_from_velocity(p, v, target_pos)
        
        # 创建临时的Quadrotor对象以获取world_frame
        quad = Quadrotor(pos=p, quat=quat, vel=v)
        frame = quad.world_frame()  # (3, 6)
        
        # 增大无人机尺寸：缩放因子
        size_scale = 2.5  # 增大1.5倍
        center_pos = frame[:, 4:5]  # 机体中心位置
        # 相对于中心缩放所有点
        frame_scaled = center_pos + (frame - center_pos) * size_scale
        
        # 绘制四旋翼几何：
        # 列 0,2 -> 机臂1 (m1-m3)
        # 列 1,3 -> 机臂2 (m2-m4)
        # 列 4 -> 机体中心
        # 列 5 -> 朝向箭头
        
        # 机臂1: m1 -> m3 (对角线)
        ax.plot([frame_scaled[0, 0], frame_scaled[0, 2]], 
                [frame_scaled[1, 0], frame_scaled[1, 2]], 
                [frame_scaled[2, 0], frame_scaled[2, 2]],
                '-', c=uav_colors[i], linewidth=2.5, zorder=40)
        
        # 机臂2: m2 -> m4 (对角线)
        ax.plot([frame_scaled[0, 1], frame_scaled[0, 3]], 
                [frame_scaled[1, 1], frame_scaled[1, 3]], 
                [frame_scaled[2, 1], frame_scaled[2, 3]],
                '-', c=uav_colors[i], linewidth=2.5, zorder=40)
        
        # 朝向箭头: 机体中心 -> h
        ax.plot([frame_scaled[0, 4], frame_scaled[0, 5]], 
                [frame_scaled[1, 4], frame_scaled[1, 5]], 
                [frame_scaled[2, 4], frame_scaled[2, 5]],
                '-', c='green', linewidth=2.0, zorder=41)
        
        # 在四个电机位置绘制小点
        # 根据阶段设置散点大小：阶段1、2时 s=10，阶段3、4时 s=20
        if stage_tag in [1, 2]:
            motor_size = 70
        elif stage_tag in [3]:
            motor_size = 100
        elif stage_tag in [4]:
            motor_size = 180
        else:
            motor_size = 100  # 默认值
        
        for m_idx in range(4):
            ax.scatter([frame_scaled[0, m_idx]], [frame_scaled[1, m_idx]], [frame_scaled[2, m_idx]],
                      s=motor_size, color=uav_colors[i], edgecolor='k', linewidths=0.3, zorder=42)

        if show_text:
            speed = float(np.linalg.norm(v))
            dist_to_target = float(np.linalg.norm(p - target_pos))
            text_lines = [f"UAV{i+1}"]
            if show_dist:
                text_lines.append(f"d={dist_to_target:.1f}")
            text_lines.extend([
                f"|v|={speed:.2f}"
            ])
            ax.text(p[0], p[1], p[2] + 1.2,
                    "\n".join(text_lines),
                    fontsize=8, zorder=60)


def draw_tetrahedron(ax, pts4, face_alpha=0.18, edge_color="k", face_color="red"):
    """
    用 4 个 UAV 位置绘制四面体（四个三角面），用于 encircle/capture/hold 阶段示意。
    pts4: (4,3)
    """
    P = np.asarray(pts4, float).reshape(4, 3)
    faces = [
        [P[0], P[1], P[2]],
        [P[0], P[1], P[3]],
        [P[0], P[2], P[3]],
        [P[1], P[2], P[3]],
    ]
    poly = Poly3DCollection(faces, alpha=face_alpha, facecolor=face_color, edgecolor=edge_color, linewidths=0.8)
    ax.add_collection3d(poly)

    # 边
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for i,j in edges:
        ax.plot([P[i,0],P[j,0]],[P[i,1],P[j,1]],[P[i,2],P[j,2]],
                color=edge_color, linewidth=1.0, alpha=0.9, zorder=30)


def draw_track_lines(ax, uav_pos, target_pos, ls="--", lw=1.2):
    """Track 阶段：用虚线表示追踪方向（UAV->Target）。"""
    uav_pos = np.asarray(uav_pos, float)
    target_pos = np.asarray(target_pos, float).reshape(3)
    for i in range(4):
        p = uav_pos[i]
        ax.plot([p[0], target_pos[0]], [p[1], target_pos[1]], [p[2], target_pos[2]],
                linestyle=ls, linewidth=lw, color="k", alpha=0.6, zorder=10)


def draw_center_of_gravity(ax, uav_pos, target_pos, color="purple", marker="o", size=100):
    """
    绘制四架无人机的重心（质心）以及重心到目标的连线。
    uav_pos: (4,3) - 四架无人机的位置
    target_pos: (3,) - 目标位置
    """
    uav_pos = np.asarray(uav_pos, float)
    target_pos = np.asarray(target_pos, float).reshape(3)
    
    # 计算重心（四架无人机的平均位置）
    center = np.mean(uav_pos, axis=0)
    
    # 绘制重心点（无边框，使用洋红色）
    ax.scatter([center[0]], [center[1]], [center[2]],
               marker=marker, s=size, color="magenta", edgecolor="none", linewidths=0, zorder=45)
    # 文字放在点上方，增大z偏移量和zorder值，确保不被遮挡
    ax.text(center[0]-1.5, center[1]-2, center[2] + 1.5, "Cen",
            fontsize=7, color=color, fontweight='bold', zorder=100, ha='center', va='bottom')
    
    # 绘制重心到目标的连线（实线）
    ax.plot([center[0], target_pos[0]], [center[1], target_pos[1]], [center[2], target_pos[2]],
            linestyle="-", linewidth=2.0, color=color, alpha=0.8, zorder=15, label="CG to Target")
    
    # 显示重心到目标的距离
    d_cg = np.linalg.norm(center - target_pos)
    mid_point = (center + target_pos) / 2
    # ax.text(mid_point[0], mid_point[1], mid_point[2] + 1.0,
    #         f"d_cg={d_cg:.1f}", fontsize=9, color=color, zorder=60)


# -----------------------------
# 3) 四阶段示例数据（你可以替换为环境中的真实状态）
# -----------------------------
def make_stage_snapshots():
    """
    根据奖励函数的阶段定义生成四个阶段的示例数据。
    阶段判断逻辑（参考 sim_env_mpc.py cal_rewards_dones_）:
    - Tag 1 (Track): d_cg >= d_limit (20)
    - Tag 2 (Encircle): d_cg < d_limit (20) 且 not encircle
    - Tag 3 (Capture): encircle 为真 且 (dist_arr > d_capture).any() (d_capture=16)
    - Tag 4 (Hold): encircle 为真 且 (dist_arr <= d_capture).all() (d_capture=16)
    """
    stage = {}
    d_limit = 20
    d_capture = 16
    d_long = 14
    
    # 目标位置（固定示意）
    target = np.array([60.0, 55.0, 18.0])

    # ========== Stage 1: Track (追踪阶段) ==========
    # 条件：d_cg >= d_limit (20)，即四架无人机中心到目标的距离 >= 20
    # 特征：无人机分散，朝向目标移动，目标不在四面体内
    uav_pos_track = np.array([
        [35, 30, 16],  # 距离目标约 30
        [40, 70, 15],  # 距离目标约 20
        [25, 45, 19],  # 距离目标约 25
        [30, 50, 17],  # 距离目标约 22
    ], float)
    p_center_track = np.mean(uav_pos_track, axis=0)
    d_cg_track = np.linalg.norm(p_center_track - target)
    print(f"Track阶段: d_cg = {d_cg_track:.2f}, 应 >= {d_limit}")
    
    stage["track"] = dict(
        title="Track (追踪)",
        target=target,
        uav_pos=uav_pos_track,
        uav_vel=np.array([
            [ 3.5,  2.8,  0.2],  # 朝向目标方向
            [ 2.5, -1.8,  0.1],
            [ 3.0,  2.5,  0.3],
            [ 2.8,  1.5,  0.15],
        ], float),
        show_tetra=False,
        show_track_lines=True,
        tag=1
    )

    # ========== Stage 2: Encircle (围绕阶段) ==========
    # 条件：d_cg < d_limit (20) 且 not encircle（目标不在四面体内）
    # 特征：四架无人机分散在目标周围较远的位置，形成包围态势但还未完全包围
    # 模仿参考图片：无人机离得远一点，不绘制四面体，显示追踪虚线
    uav_pos_encircle = np.array([
        [45, 40, 20],  # 距离目标约 25.5（左上）
        [75, 50, 22],  # 距离目标约 19.9（右上）
        [55, 75, 18],  # 距离目标约 20.2（前）
        [35, 35, 16],  # 距离目标约 30.4（左下）
    ], float)
    p_center_encircle = np.mean(uav_pos_encircle, axis=0)
    d_cg_encircle = np.linalg.norm(p_center_encircle - target)
    encircle_flag, _ = point_in_tetrahedron_3d(target, *uav_pos_encircle)
    print(f"Encircle阶段: d_cg = {d_cg_encircle:.2f} (< {d_limit}), encircle = {encircle_flag} (应为False)")
    
    stage["encircle"] = dict(
        title="Encircle (围绕)",
        target=target,
        uav_pos=uav_pos_encircle,
        uav_vel=np.array([
            [ 1.2,  1.5,  0.1],  # 朝向目标移动
            [-1.5,  0.5,  0.2],
            [ 0.5, -2.0,  0.0],
            [ 1.8,  1.2,  0.15],
        ], float),
        show_tetra=False,  # 不绘制四面体，模仿参考图
        show_track_lines=True,  # 显示追踪虚线，模仿参考图
        tag=2
    )

    # ========== Stage 3: Capture (收缩阶段) ==========
    # 条件：encircle 为真（目标在四面体内）且至少有一个无人机距离 > d_capture (16)
    # 特征：目标已被包围，但距离还在收缩中
    # 在目标周围形成四面体：目标位置 (60, 55, 18)
    # 使用规则四面体顶点，距离约 17-18（部分 > 16）
    uav_pos_capture = np.array([
        [60, 55, 35],   # 上方，距离约 17
        [73, 55, 18],   # 右方，距离约 13
        [55, 68, 18],   # 前方，距离约 13
        [52, 47, 18],   # 左方，距离约 8.5
    ], float)
    dist_arr_capture = np.array([np.linalg.norm(p - target) for p in uav_pos_capture])
    encircle_flag_capture, _ = point_in_tetrahedron_3d(target, *uav_pos_capture)
    p_center_capture = np.mean(uav_pos_capture, axis=0)
    d_cg_capture = np.linalg.norm(p_center_capture - target)
    print(f"Capture阶段: encircle = {encircle_flag_capture} (应为True), "
          f"dist_arr = {dist_arr_capture}, d_cg = {d_cg_capture:.2f}, "
          f"max_dist = {dist_arr_capture.max():.2f} (应有一些 > {d_capture})")
    
    stage["capture"] = dict(
        title="Capture (收缩)",
        target=target,
        uav_pos=uav_pos_capture,
        uav_vel=np.array([
            [ 0.4,  0.1, -0.3],  # 向目标收缩
            [-0.3,  0.2, -0.08],
            [ 0.05, -0.4,  0.0],
            [ 0.3,  0.25,  0.05],
        ], float),
        show_tetra=True,
        show_track_lines=False,
        tag=3
    )

    # ========== Stage 4: Hold (保持阶段) ==========
    # 条件：encircle 为真 且 所有无人机距离 <= d_capture (16)
    # 特征：紧密包围，构型稳定，速度很小
    # 在目标周围形成更紧密的四面体，所有距离 <= 16
    uav_pos_hold = np.array([
        [60, 55, 28],   # 上方，距离约 10
        [68, 55, 18],   # 右方，距离约 8
        [60, 63, 18],   # 前方，距离约 8
        [52, 55, 18],   # 左方，距离约 8
    ], float)
    dist_arr_hold = np.array([np.linalg.norm(p - target) for p in uav_pos_hold])
    encircle_flag_hold, _ = point_in_tetrahedron_3d(target, *uav_pos_hold)
    p_center_hold = np.mean(uav_pos_hold, axis=0)
    d_cg_hold = np.linalg.norm(p_center_hold - target)
    print(f"Hold阶段: encircle = {encircle_flag_hold} (应为True), "
          f"dist_arr = {dist_arr_hold}, d_cg = {d_cg_hold:.2f}, "
          f"max_dist = {dist_arr_hold.max():.2f} (应 <= {d_capture})")
    
    stage["hold"] = dict(
        title="Hold (保持)",
        target=target,
        uav_pos=uav_pos_hold,
        uav_vel=np.array([
            [0.1, 0.03, -0.05],  # 几乎静止，微调
            [-0.08, 0.04, -0.02],
            [0.01, -0.1, 0.005],
            [0.05, 0.06, 0.01],
        ], float),
        show_tetra=True,
        show_track_lines=False,
        tag=4
    )

    return stage


# -----------------------------
# 4) 主绘图流程：生成四张阶段图
# -----------------------------
def plot_one_stage(mapobs, bounds, snapshot, out_path, elev=28, azim=-55, obstacle_height=60.0):
    # 计算并显示阶段状态信息（用于确定视角和范围）
    uav_pos = np.asarray(snapshot["uav_pos"], float)
    target = np.asarray(snapshot["target"], float)
    p_center = np.mean(uav_pos, axis=0)
    tag = snapshot.get("tag", 0)
    
    # 根据阶段设置合适的视角和显示范围
    # 计算所有关键点的范围
    all_points = np.vstack([uav_pos, target.reshape(1, -1), p_center.reshape(1, -1)])
    x_all = all_points[:, 0]
    y_all = all_points[:, 1]
    z_all = all_points[:, 2]
    
    # 根据阶段调整视角和范围
    margin = 15.0  # 基础边距
    if tag == 1:  # Track阶段：显示较广范围，俯视角度
        x_center, y_center = np.mean(x_all), np.mean(y_all)
        x_range = np.max(x_all) - np.min(x_all) + margin * 0.5
        y_range = np.max(y_all) - np.min(y_all) + margin * 0.5
        xmin, xmax = x_center - x_range/2, x_center + x_range/2
        ymin, ymax = y_center - y_range/2, y_center + y_range/2
        zmin, zmax = max(0, np.min(z_all) - 5), min(obstacle_height, np.max(z_all) + 10)
        elev, azim = 45, 45  # 俯视斜角
    elif tag == 2:  # Encircle阶段：突出重心和目标关系，侧视角度
        x_center, y_center = (target[0] + p_center[0])/2, (target[1] + p_center[1])/2
        x_range = max(np.max(x_all) - np.min(x_all), 35) + margin * 0.5
        y_range = max(np.max(y_all) - np.min(y_all), 35) + margin * 0.5
        xmin, xmax = x_center - x_range/2, x_center + x_range/2
        ymin, ymax = y_center - y_range/2, y_center + y_range/2
        zmin, zmax = max(0, np.min(z_all) - 5), min(obstacle_height, np.max(z_all) + 10)
        elev, azim = 25, -60  # 较低俯角，侧视
    elif tag == 3:  # Capture阶段：突出四面体和目标，中等角度
        x_center, y_center = target[0], target[1]
        x_range = max(np.max(x_all) - np.min(x_all), 25) + margin * 0.5
        y_range = max(np.max(y_all) - np.min(y_all), 25) + margin * 0.5
        xmin, xmax = x_center - x_range/2, x_center + x_range/2
        ymin, ymax = y_center - y_range/2, y_center + y_range/2
        zmin, zmax = max(0, np.min(z_all) - 3), min(obstacle_height, np.max(z_all) + 8)
        elev, azim = 30, -50  # 中等俯角
    elif tag == 4:  # Hold阶段：紧密包围，近视角，突出细节
        x_center, y_center = target[0], target[1]
        x_range = max(np.max(x_all) - np.min(x_all), 15) + margin * 0.5
        y_range = max(np.max(y_all) - np.min(y_all), 15) + margin * 0.5
        xmin, xmax = x_center - x_range/2, x_center + x_range/2
        ymin, ymax = y_center - y_range/2, y_center + y_range/2
        zmin, zmax = max(0, np.min(z_all) - 2), min(obstacle_height, np.max(z_all) + 5)
        elev, azim = 35, -55  # 稍近的视角，突出细节
    else:  # 默认全景
        bounds_arr = np.asarray(bounds, float)
        if bounds_arr.ndim == 1:
            xmin, xmax = bounds_arr[0], bounds_arr[1]
            ymin, ymax = bounds_arr[0], bounds_arr[1]
            zmin, zmax = 0.0, obstacle_height
        else:
            bounds_2d = bounds_arr.reshape(3, 2)
            (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds_2d

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # 障碍物：使用 Map 对象的 plotobs 方法绘制（已经包含了）
    mapobs.plotobs(ax, scale=1)

    # UAV + target
    draw_uavs_and_target(ax, snapshot["uav_pos"], snapshot["uav_vel"], snapshot["target"],
                         show_text=True, vel_scale=6.0, show_dist=True, stage_tag=snapshot.get("tag", None))

    # Track 阶段虚线
    if snapshot.get("show_track_lines", False):
        draw_track_lines(ax, snapshot["uav_pos"], snapshot["target"])

    # Encircle/Capture/Hold 四面体/围捕面
    if snapshot.get("show_tetra", False):
        draw_tetrahedron(ax, snapshot["uav_pos"], face_alpha=0.18, face_color="red", edge_color="k")
    
    # 阶段二（Encircle）：绘制重心和重心到目标的连线
    if snapshot.get("tag", None) == 2:
        draw_center_of_gravity(ax, uav_pos, target)

    # 计算阶段状态信息（用于标题显示）
    d_cg = np.linalg.norm(p_center - target)
    dist_arr = np.array([np.linalg.norm(p - target) for p in uav_pos])
    encircle, _ = point_in_tetrahedron_3d(target, *uav_pos)
    
    # 视角与坐标范围
    title_with_info = f"{snapshot['title']} (Tag={snapshot.get('tag', '?')})\n"
    title_with_info += f"d_cg={d_cg:.1f}, encircle={encircle}, max_dist={dist_arr.max():.1f}"
    ax.set_title(title_with_info, fontsize=14, pad=10)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    set_axes_equal(ax)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_four_stages():
    # 你环境里的示例参数（按需替换）
    episode = 0
    obstacle_height = 60.0
    # bounds 应该是一维数组 [min, max]，用于 x, y 坐标范围
    bounds = np.array([0, 100], float)

    # start 应该是 (n, 3) 形状的数组，goal 是 (3,) 形状
    # 注意：start 应该是二维数组，每个元素是一维数组 [x, y, z]
    start_ = np.array([[0.0, 0.0, 5.0]], float)  # 至少一个起点，形状为 (1, 3)
    target_goal = np.array([60.0, 55.0, 18.0], float)  # 与目标位置一致，形状为 (3,)

    # 按你要求的调用形式生成障碍物
    # 注意：generate_map 中的 bounds 是一维数组，用于 x, y 坐标范围
    mapobs, obstacles = generate_map(bounds, 2.8, obstacle_height, start_, target_goal, 250, seed_num=42 + episode)

    # 四阶段状态（可替换为真实 quads 状态）
    stage = make_stage_snapshots()

    os.makedirs("fig_stages", exist_ok=True)
    for key in ["track", "encircle", "capture", "hold"]:
        out_path = os.path.join("fig_stages", f"{key}.png")
        plot_one_stage(mapobs, bounds, stage[key], out_path, obstacle_height=obstacle_height)

    # 可选：再生成一个 2x2 汇总图（便于论文排版）
    fig = plt.figure(figsize=(12, 10))
    keys = ["track", "encircle", "capture", "hold"]
    axes_list = []  # 保存所有子图的axes，用于单独保存
    
    for idx, key in enumerate(keys, 1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        # 障碍物：使用 Map 对象的 plotobs 方法绘制
        mapobs.plotobs(ax, scale=1)
        snap = stage[key]
        draw_uavs_and_target(ax, snap["uav_pos"], snap["uav_vel"], snap["target"], show_text=False, vel_scale=6.0, stage_tag=snap.get("tag", None))
        if snap.get("show_track_lines", False):
            draw_track_lines(ax, snap["uav_pos"], snap["target"], lw=1.0)
        if snap.get("show_tetra", False):
            draw_tetrahedron(ax, snap["uav_pos"], face_alpha=0.16, face_color="red", edge_color="k")
        # 阶段二（Encircle）：绘制重心和重心到目标的连线
        if snap.get("tag", None) == 2:
            draw_center_of_gravity(ax, snap["uav_pos"], snap["target"], color="purple", marker="o", size=80)
        
        # 为汇总图也应用与单张图相同的视角和范围逻辑
        uav_pos_sum = np.asarray(snap["uav_pos"], float)
        target_sum = np.asarray(snap["target"], float)
        p_center_sum = np.mean(uav_pos_sum, axis=0)
        tag_sum = snap.get("tag", 0)
        
        # 计算范围（与单张图相同逻辑）
        all_points_sum = np.vstack([uav_pos_sum, target_sum.reshape(1, -1), p_center_sum.reshape(1, -1)])
        x_all_sum = all_points_sum[:, 0]
        y_all_sum = all_points_sum[:, 1]
        z_all_sum = all_points_sum[:, 2]
        
        margin_sum = 15.0
        if tag_sum == 1:  # Track
            x_center_sum, y_center_sum = np.mean(x_all_sum), np.mean(y_all_sum)
            x_range_sum = np.max(x_all_sum) - np.min(x_all_sum) + margin_sum * 0.5
            y_range_sum = np.max(y_all_sum) - np.min(y_all_sum) + margin_sum * 0.5
            xmin, xmax = x_center_sum - x_range_sum/2, x_center_sum + x_range_sum/2
            ymin, ymax = y_center_sum - y_range_sum/2, y_center_sum + y_range_sum/2
            zmin, zmax = max(0, np.min(z_all_sum) - 5), min(obstacle_height, np.max(z_all_sum) + 10)
            elev_sum, azim_sum = 45, 45
        elif tag_sum == 2:  # Encircle
            x_center_sum, y_center_sum = (target_sum[0] + p_center_sum[0])/2, (target_sum[1] + p_center_sum[1])/2
            x_range_sum = max(np.max(x_all_sum) - np.min(x_all_sum), 35) + margin_sum * 0.5
            y_range_sum = max(np.max(y_all_sum) - np.min(y_all_sum), 35) + margin_sum * 0.5
            xmin, xmax = x_center_sum - x_range_sum/2, x_center_sum + x_range_sum/2
            ymin, ymax = y_center_sum - y_range_sum/2, y_center_sum + y_range_sum/2
            zmin, zmax = max(0, np.min(z_all_sum) - 5), min(obstacle_height, np.max(z_all_sum) + 10)
            elev_sum, azim_sum = 25, -60
        elif tag_sum == 3:  # Capture
            x_center_sum, y_center_sum = target_sum[0], target_sum[1]
            x_range_sum = max(np.max(x_all_sum) - np.min(x_all_sum), 25) + margin_sum * 0.5
            y_range_sum = max(np.max(y_all_sum) - np.min(y_all_sum), 25) + margin_sum * 0.5
            xmin, xmax = x_center_sum - x_range_sum/2, x_center_sum + x_range_sum/2
            ymin, ymax = y_center_sum - y_range_sum/2, y_center_sum + y_range_sum/2
            zmin, zmax = max(0, np.min(z_all_sum) - 3), min(obstacle_height, np.max(z_all_sum) + 8)
            elev_sum, azim_sum = 30, -50
        elif tag_sum == 4:  # Hold
            x_center_sum, y_center_sum = target_sum[0], target_sum[1]
            x_range_sum = max(np.max(x_all_sum) - np.min(x_all_sum), 15) + margin_sum * 0.5
            y_range_sum = max(np.max(y_all_sum) - np.min(y_all_sum), 15) + margin_sum * 0.5
            xmin, xmax = x_center_sum - x_range_sum/2, x_center_sum + x_range_sum/2
            ymin, ymax = y_center_sum - y_range_sum/2, y_center_sum + y_range_sum/2
            zmin, zmax = max(0, np.min(z_all_sum) - 2), min(obstacle_height, np.max(z_all_sum) + 5)
            elev_sum, azim_sum = 35, -55
        else:  # 默认
            bounds_arr = np.asarray(bounds, float)
            if bounds_arr.ndim == 1:
                xmin, xmax = bounds_arr[0], bounds_arr[1]
                ymin, ymax = bounds_arr[0], bounds_arr[1]
                zmin, zmax = 0.0, obstacle_height
            else:
                xmin, xmax = bounds_arr[0, 0], bounds_arr[0, 1]
                ymin, ymax = bounds_arr[1, 0], bounds_arr[1, 1]
                zmin, zmax = bounds_arr[2, 0], bounds_arr[2, 1]
            elev_sum, azim_sum = 28, -55
        
        title_sum = f"{snap['title']} (Tag={tag_sum})"
        ax.set_title(title_sum, fontsize=14, pad=6)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.view_init(elev=elev_sum, azim=azim_sum)
        ax.set_axis_off()
        set_axes_equal(ax)
        
        # 保存当前子图的信息，用于单独保存
        axes_list.append((ax, idx))

    fig.tight_layout()
    fig.savefig(os.path.join("fig_stages", "four_stages_grid.png"), dpi=300)
    
    # 将每个子图单独保存为 1.png, 2.png, 3.png, 4.png
    for ax, idx in axes_list:
        # 创建新的figure用于单独保存
        fig_single = plt.figure(figsize=(9, 7))
        ax_single = fig_single.add_subplot(111, projection="3d")
        
        # 复制当前子图的所有内容
        # 获取原始axes的视角和范围
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
        elev = ax.elev
        azim = ax.azim
        
        # 重新绘制内容
        snap = stage[keys[idx-1]]
        mapobs.plotobs(ax_single, scale=1)
        draw_uavs_and_target(ax_single, snap["uav_pos"], snap["uav_vel"], snap["target"], show_text=False, vel_scale=6.0, stage_tag=snap.get("tag", None))
        if snap.get("show_track_lines", False):
            draw_track_lines(ax_single, snap["uav_pos"], snap["target"], lw=1.0)
        if snap.get("show_tetra", False):
            draw_tetrahedron(ax_single, snap["uav_pos"], face_alpha=0.16, face_color="red", edge_color="k")
        if snap.get("tag", None) == 2:
            draw_center_of_gravity(ax_single, snap["uav_pos"], snap["target"], color="purple", marker="o", size=80)
        
        # 设置相同的视角和范围
        ax_single.set_xlim(xlim)
        ax_single.set_ylim(ylim)
        ax_single.set_zlim(zlim)
        ax_single.view_init(elev=elev, azim=azim)
        ax_single.set_axis_off()
        set_axes_equal(ax_single)
        
        # 保存单独的子图
        fig_single.tight_layout()
        fig_single.savefig(os.path.join("fig_stages", f"{idx}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_single)
    
    plt.close(fig)

    print("Saved figures to: fig_stages/")
    print("  - Individual stages: track.png, encircle.png, capture.png, hold.png")
    print("  - Grid summary: four_stages_grid.png")
    print("  - Grid subplots: 1.png (Track), 2.png (Encircle), 3.png (Capture), 4.png (Hold)")


if __name__ == "__main__":
    plot_four_stages()
