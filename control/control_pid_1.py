import numpy as np

# ---------- 一些四元数/旋转工具 ----------

def quat_conj(q):
    """共轭四元数"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mul(q1, q2):
    """四元数乘法 q = q1 ⊗ q2"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rot_mat_to_quat(R):
    """
    旋转矩阵 (body->world) -> 四元数 (w,x,y,z)，右手系，R @ v_body = v_world
    """
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)

def thrust_to_attitude(thrust_world, yaw_sp):
    """
    根据期望推力方向（世界系向量）和期望 yaw（绕世界 z 轴）生成期望四元数 (body->world)

    thrust_world: (3,) 世界系推力向量
    yaw_sp: 标量, 期望偏航角 (rad), 在世界系绕 +Z
    """
    tw = np.array(thrust_world, dtype=float)
    Tn = np.linalg.norm(tw)
    if Tn < 1e-6:
        # 推力太小，就保持水平朝向，只用 yaw
        c, s = np.cos(yaw_sp), np.sin(yaw_sp)
        # 世界系下，body x 对应 yaw 旋转
        xb = np.array([c, s, 0.0])
        zb = np.array([0.0, 0.0, 1.0])
        yb = np.cross(zb, xb)
        yb /= np.linalg.norm(yb) + 1e-12
    else:
        # 期望机体 z 轴在世界系下的方向 = thrust_world 归一化
        zb = tw / Tn

        # 参考世界 x 轴方向 (由 yaw_sp 决定)
        c, s = np.cos(yaw_sp), np.sin(yaw_sp)
        xC = np.array([c, s, 0.0])   # 世界水平面上的参考前向

        # 构造 body y 轴，使其垂直 zb 和 xC
        yb = np.cross(zb, xC)
        yn = np.linalg.norm(yb)
        if yn < 1e-6:
            # thrust 跟参考方向几乎平行，选一个正交向量
            yb = np.array([0.0, 1.0, 0.0])
        else:
            yb /= yn

        # body x 轴由 y×z 得到
        xb = np.cross(yb, zb)

    xb /= np.linalg.norm(xb) + 1e-12
    yb /= np.linalg.norm(yb) + 1e-12
    zb /= np.linalg.norm(zb) + 1e-12

    # 组装 R(b->w)：列为 body 轴在世界系中的表示
    R = np.column_stack((xb, yb, zb))
    return rot_mat_to_quat(R)


# ---------- 控制器 ----------

class QuadrotorControllerXYZPos:
    """
    只实现 xyz_pos 模式：
    pos_sp -> vel_sp -> thrust_world -> attitude -> rate -> [T,M] -> 4 个电机输入 u (0~1)
    """

    def __init__(self, quad: "Quadrotor"):
        self.quad = quad
        self.mass = quad.mass
        self.g_vec = quad.g.flatten()   # [0,0,9.81]

        # 根据几何构造 [T,M] <-> 4 个推力 的混控矩阵
        # T  = sum(Fi)
        # Mx = sum(Fi * y_f[i])
        # My = -sum(Fi * x_f[i])
        # Mz = sum(Fi * z_l_tau[i])
        A = np.vstack([
            np.ones(4),
            quad.y_f,
            -quad.x_f,
            quad.z_l_tau
        ])           # A shape: (4,4)
        self.A_inv = np.linalg.inv(A)

        # 一些简单的增益（你可以根据需要调）
        self.Kp_pos  = np.array([1.0, 1.0, 1.0])   # 位置 P
        self.Kp_vel  = np.array([1.5, 1.5, 2.0])   # 速度 P
        self.Kp_att  = np.array([5.0, 5.0, 3.0])   # 姿态 P (roll, pitch, yaw)
        self.Kp_rate = np.array([0.1, 0.1, 0.1])   # 角速度 P
        self.Kd_rate = np.array([0.01, 0.01, 0.01])# 角速度 D

        self.vel_max = quad.max_vel

    # ---- 内部：从 [T,M] 算 4 个推力输入 u（0~1） ----
    def mix_TM_to_u(self, T, M):
        """
        T: 标量，总推力(N)
        M: (3,) 力矩 (Nm)
        返回: u_norm (4,) in [0,1]
        """
        t_vec = np.array([T, M[0], M[1], M[2]])
        F = self.A_inv @ t_vec   # 每个电机推力 Fi (N)

        # 限幅到电机能力范围 [0, max_thrust]
        F = np.clip(F, 0.0, self.quad.max_thrust)

        # Quadrotor.update 里会 self.u = u * max_thrust
        # 所以这里归一化到 0~1
        u_norm = F / self.quad.max_thrust
        return u_norm

    # ---- 外部接口：一步控制（对应 xyz_pos 那条链） ----
    def step(self, pos_sp, vel_sp=None, acc_sp=None, yaw_sp=0.0):
        """
        计算 xyz_pos 控制下的 4 个电机输入 u (0~1)

        pos_sp: (3,)  期望位置 (x,y,z)
        vel_sp: (3,)  期望速度, 默认为 0
        acc_sp: (3,)  期望加速度前馈, 默认为 0
        yaw_sp: 标量，期望 yaw（世界系绕 +Z）

        返回:
            u (4,) in [0,1]
        """
        quad = self.quad
        pos   = quad.pos
        vel   = quad.vel
        quat  = quad.quat
        omega = quad.a_rate

        if vel_sp is None:
            vel_sp = np.zeros(3)
        if acc_sp is None:
            acc_sp = np.zeros(3)

        # 1) 位置环: pos -> vel_sp (类似 z_pos_control + xy_pos_control)
        pos_err = pos_sp - pos
        vel_sp = vel_sp + self.Kp_pos * pos_err

        # saturateVel: 限制期望速度模长
        v_norm = np.linalg.norm(vel_sp)
        if v_norm > self.vel_max:
            vel_sp = vel_sp * (self.vel_max / (v_norm + 1e-12))

        # 2) 速度环: vel -> acc_sp_cmd (类似 z_vel_control + xy_vel_control)
        vel_err = vel_sp - vel
        acc_sp_cmd = acc_sp + self.Kp_vel * vel_err   # 世界系中期望加速度

        # 加上重力补偿：希望 net_acc = acc_sp_cmd
        # 动力学里: v_dot = -g + R * [0,0,sum(u)]/m + ...，
        # 所以需要 thrust_world/mass ≈ acc_sp_cmd + g
        acc_cmd_world = acc_sp_cmd + self.g_vec  # (3,)

        # 3) thrustToAttitude: 世界系期望推力 -> 期望姿态四元数 q_d
        thrust_world = self.mass * acc_cmd_world
        q_d = thrust_to_attitude(thrust_world, yaw_sp)

        # 4) attitude_control: q 与 q_d -> 期望角速度 omega_sp
        q = quat
        q_conj = quat_conj(q)
        q_e = quat_mul(q_conj, q_d)   # 误差四元数 (当前->期望)

        # 把虚部当成姿态误差，生成角速度期望
        # 注意这里简单用对角增益，实际可分开调 roll/pitch/yaw
        omega_sp = 2.0 * np.sign(q_e[0]) * q_e[1:4] * self.Kp_att

        # 5) rate_control: 角速度闭环 -> 力矩 M
        rate_err = omega_sp - omega
        M = self.Kp_rate * rate_err - self.Kd_rate * omega   # (3,)

        # 6) 总推力: 用 thrust_world 的模长
        T = np.linalg.norm(thrust_world)

        # 7) 混控器: [T,M] -> 4 个电机推力 -> 归一化 u(0~1)
        u = self.mix_TM_to_u(T, M)
        return u
