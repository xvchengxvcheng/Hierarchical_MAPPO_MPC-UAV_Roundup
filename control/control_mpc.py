import os
import casadi as ca
import numpy as np
# from rbfnet import RBFNet
from utils_ import skew_symmetric, v_dot_q, quaternion_inverse
from typing import TYPE_CHECKING
from envs.Quadrotor import Quadrotor
# if TYPE_CHECKING:
#     from env.sim_env import Quadrotor
from numpy.linalg import norm


# velMaxAll = 6.0
class Controller:
    def __init__(self, quad:Quadrotor, n_nodes, plan_end_l, pos_max = 100, dt=0.1, alpha = 0.005):#alpha太大姿态变化小？
        """
        :param quad: quadrotor object
        :type quad: Quadrotor3D
        :param n_nodes: number of optimization nodes until time horizon
        """
        self.alpha = alpha
        self.qr = 0.0005
        self.ru_d = 0.0005
        self.rho_v = 1e4
        self.rho_obs = 1e5
        self.M = 16
        self.SLACK_CAP = 1e3
        self.plan_end_l = plan_end_l # long
        self.rho_pos = 1e5  # 位置软约束权重（可按需要调大/调小）
        self.POS_MIN = -10.0
        self.POS_MAX = pos_max

        self.N = n_nodes    # number of control nodes within horizon
        self.dt = dt        # time step
        self.x_dim = 13
        self.u_dim = 4
        # self.use_rbf = use_rbf

        # if self.use_rbf:
        #     self.rbfnet = RBFNet(num_inputs1=3, num_inputs2=3, num_rbfs=40, output_dim=3, learning_rate=0.2)

        self.opti = ca.Opti()
        self.opt_states = self.opti.variable(self.x_dim, self.N+1)
        self.opt_controls = self.opti.variable(self.u_dim, self.N)
        # 速度修正量
        self.s_v = self.opti.variable(1, self.N + 1)
        # 平面参数：A @ p + b <= s   其中 A 形状 (M,3), b 形状 (M,1)
        self.plane_A = self.opti.parameter(self.M, 3)
        self.plane_b = self.opti.parameter(self.M, 1)

        # 每个时刻、每个平面的松弛：s_obs >= 0，形状 (M, N+1)
        self.s_obs = self.opti.variable(self.M, self.N + 1)
        self.opti.subject_to(self.opti.bounded(0, self.s_obs, self.SLACK_CAP))

        self.quad = quad

        self.max_u = quad.max_input_value
        self.min_u = quad.min_input_value

        g = 9.81
        u_hover = (self.quad.mass * g) / (4.0 * self.quad.max_thrust)
        u_hover = float(np.clip(u_hover, self.min_u, self.max_u))
        self.u_hover = u_hover

        self.u_prev = np.full((self.u_dim, self.N), self.u_hover, dtype=float)
        self.x_prev = np.zeros((self.x_dim, self.N+1))

        # Declare model variables
        self.p = self.opt_states[:3,:]      # position
        self.q = self.opt_states[3:7,:]     # angle quaternion (wxyz)
        self.v = self.opt_states[7:10,:]    # velocity
        self.r = self.opt_states[10:13,:]   # angle rate

        f = lambda x_, u_, f_d_: ca.vertcat(*[
            self.p_dynamics(x_),
            self.q_dynamics(x_),
            self.v_dynamics(x_, u_, f_d_),
            self.w_dynamics(x_, u_)
        ])

        # Noise variables
        self.f_d = self.opti.parameter(3, 1)
        self.f_t = self.opti.parameter(3, 1)

        # Initial condition
        self.opt_x_0 = self.opti.parameter(self.x_dim, 1)
        self.opt_x_end = self.opti.parameter(3, 1)
        self.opti.subject_to(self.opt_states[:, 0] == self.opt_x_0)
        for i in range(self.N):
            x_next = self.opt_states[:,i] + f(self.opt_states[:,i], self.opt_controls[:,i], self.f_d)*self.dt
            self.opti.subject_to(self.opt_states[:,i+1] == x_next)

        obj = 0

        planning_h = 1
        obj += 5 * ca.sumsqr(self.opt_states[:3, self.plan_end_l:] - self.opt_x_end) + 0.08 * ca.sumsqr(
            self.opt_controls - self.u_hover) + ca.sum2(
            self.alpha * (self.opt_states[3, planning_h:] ** 2 + self.opt_states[6, planning_h:] ** 2) * (
                    self.opt_states[4, planning_h:] ** 2 + self.opt_states[5, planning_h:] ** 2))

        for k in range(self.N + 1):
            v_k = self.v[:, k]  # 3×1 向量
            slack_k = self.s_v[0, k]  # 标量松弛
            v_sq = ca.sumsqr(v_k)
            vmax_eff = self.quad.max_vel + slack_k
            self.opti.subject_to(
                v_sq <= vmax_eff * vmax_eff
            )

        # 速度约束
        obj += self.rho_v * ca.sumsqr(self.s_v)
        # 障碍物约束
        obj += self.rho_obs * ca.sumsqr(self.s_obs)
        # 位置软约束
        # obj += self.rho_pos * (ca.sumsqr(self.s_p_lo) + ca.sumsqr(self.s_p_hi))

        self.opti.minimize(obj)
        for k in range(self.N + 1):
            gk = self.plane_A @ self.p[:, k] + self.plane_b - self.s_obs[:, k]  # (M,1)
            self.opti.subject_to(self.opti.bounded(-self.SLACK_CAP, gk, 0))

        self.opti.subject_to(self.opti.bounded(self.min_u, self.opt_controls, self.max_u))

        Z = ca.DM.zeros(1, self.N + 1)
        self.opti.subject_to(self.opti.bounded(Z, self.s_v, 1e6))

        opts_setting = {
            'ipopt.max_iter': 2000,
            'ipopt.tol': 1e-4,  # 主要收敛容差（这才是真正起作用的参数）
            'ipopt.acceptable_tol': 1e-4,  # 可接受解容差
            'ipopt.acceptable_obj_change_tol': 1e-5,
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.warm_start_init_point': 'yes',
            # 可选：如果需要更详细的诊断信息，可以临时开启
            # 'ipopt.print_level': 5,  # 0=无输出, 5=详细输出
        }

        self.opti.solver('ipopt', opts_setting)

    def p_dynamics(self, x):
        v = x[7:10]
        return v

    def q_dynamics(self, x):
        q = x[3:7]
        r = x[10:13]
        return 1 / 2 * ca.mtimes(skew_symmetric(r), q)

    def v_dynamics(self, x, u, f_d):
        q = x[3:7]
        f_thrust = u * self.quad.max_thrust
        g = ca.vertcat(0.0, 0.0, 9.81)
        a_thrust = ca.vertcat(0.0, 0.0, f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3]) / self.quad.mass +\
                   f_d/self.quad.mass

        v_dynamics = v_dot_q(a_thrust, q) - g
        return v_dynamics

    def w_dynamics(self, x, u):
        r = x[10:13]
        f_thrust = u * self.quad.max_thrust

        y_f = ca.MX(self.quad.y_f)
        x_f = ca.MX(self.quad.x_f)
        c_f = ca.MX(self.quad.z_l_tau)
        return ca.vertcat(
            ( ca.mtimes(f_thrust.T, y_f) + (self.quad.J[1] - self.quad.J[2]) * r[1] * r[2]) / self.quad.J[0],
            (-ca.mtimes(f_thrust.T, x_f) + (self.quad.J[2] - self.quad.J[0]) * r[2] * r[0]) / self.quad.J[1],
            ( ca.mtimes(f_thrust.T, c_f) + (self.quad.J[0] - self.quad.J[1]) * r[0] * r[1]) / self.quad.J[2])

    # def saturateVel(self):
    #
    #     # Saturate Velocity Setpoint
    #     # ---------------------------
    #     # Either saturate each velocity axis separately, or total velocity (prefered)
    #     totalVel_sp = norm(self.quad.vel)
    #     if (totalVel_sp > velMaxAll):
    #         self.vel_sp = self.quad.vel / totalVel_sp * velMaxAll

    def compute_control_signal(self, x_0, x_end, planes_list):
        self.opti.set_initial(self.s_obs, 0)
        self.opti.set_initial(self.s_v, 0)

        f_d = np.zeros((3, 1))
        if len(planes_list) == 0:
            P = np.zeros((0, 4), dtype=float)
        else:
            P = np.vstack([np.asarray(pi, dtype=float) for pi in planes_list if len(pi) > 0])

        K = min(len(P), self.M)
        A = np.zeros((self.M, 3), dtype=float)
        b = np.zeros((self.M, 1), dtype=float)  # 失效行：A=0, b=0 => 恒满足
        if K > 0:
            A[:K, :] = P[:K, :3]
            b[:K, 0] = P[:K, 3]

        self.opti.set_value(self.plane_A, A)
        self.opti.set_value(self.plane_b, b)

        self.opti.set_value(self.opt_x_0, x_0)
        self.opti.set_value(self.f_d, f_d)
        self.opti.set_value(self.opt_x_end, x_end)

        if np.all(self.x_prev == 0):
            self.x_prev = np.tile(x_0.reshape(-1, 1), (1, self.N + 1)).copy()
        self.opti.set_initial(self.opt_controls, self.u_prev)
        self.opti.set_initial(self.opt_states, self.x_prev)

        try:
            sol = self.opti.solve()
            U = sol.value(self.opt_controls)  # (4, N)
            X = sol.value(self.opt_states)
            self.u_prev = U.copy()
            self.x_prev = X.copy()
            return U[:, 0]
        except RuntimeError as e:
            print("[WARN] MPC solve failed:", str(e))

            u_fallback = None
            try:
                Udbg = self.opti.debug.value(self.opt_controls)
                if isinstance(Udbg, np.ndarray) and Udbg.size >= self.u_dim:
                    print(1)
                    cand = Udbg[:, 0].astype(float).flatten()
                    if np.all(np.isfinite(cand)):
                        print(2)
                        u_fallback = cand
            except Exception:
                pass

            if u_fallback is None:
                print(3)
                u_fallback = self.u_prev[:, 0].astype(float).flatten().copy()

            u_fallback = np.clip(u_fallback, self.min_u, self.max_u)

            return u_fallback

