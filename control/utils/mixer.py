# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np



def mixerFM(quad, thr, moment):
    t = np.array([thr, moment[0], moment[1], moment[2]])
    w_cmd = np.sqrt(np.clip(np.dot(quad.params["mixerFMinv"], t), quad.params["minWmotor"]**2, quad.params["maxWmotor"]**2))

    return w_cmd

def mixerThrust(quad, thr, moment):
    """
    输入:
        thr    : 标量，总推力 T
        moment : np.array([Mx, My, Mz])
    输出:
        F_cmd : np.array([F1, F2, F3, F4]) 每个电机的推力 (N)
    """
    T  = thr
    Mx, My, Mz = moment

    dxm = quad.params["dxm"]
    dym = quad.params["dym"]
    kTh = quad.params["kTh"]
    kTo = quad.params["kTo"]
    kappa = kTo / kTh   # 比值 kTo/kTh

    # 直接用上面推导好的 A^{-1}
    A_inv = np.array([
        [0.25,  1.0/(4*dym),  1.0/(4*dxm), -1.0/(4*kappa)],
        [0.25, -1.0/(4*dym),  1.0/(4*dxm),  1.0/(4*kappa)],
        [0.25, -1.0/(4*dym), -1.0/(4*dxm), -1.0/(4*kappa)],
        [0.25,  1.0/(4*dym), -1.0/(4*dxm),  1.0/(4*kappa)],
    ])

    t = np.array([T, Mx, My, Mz])
    F_cmd = A_inv @ t

    # 这里可以加推力限幅（根据你设定的单电机推力范围）
    # 比如：
    # F_cmd = np.clip(F_cmd, quad.params["minFmotor"], quad.params["maxFmotor"])

    return F_cmd

## Under here is the conventional type of mixer

# def mixer(throttle, pCmd, qCmd, rCmd, quad):
#     maxCmd = quad.params["maxCmd"]
#     minCmd = quad.params["minCmd"]

#     cmd = np.zeros([4, 1])
#     cmd[0] = throttle + pCmd + qCmd - rCmd
#     cmd[1] = throttle - pCmd + qCmd + rCmd
#     cmd[2] = throttle - pCmd - qCmd - rCmd
#     cmd[3] = throttle + pCmd - qCmd + rCmd
    
#     cmd[0] = min(max(cmd[0], minCmd), maxCmd)
#     cmd[1] = min(max(cmd[1], minCmd), maxCmd)
#     cmd[2] = min(max(cmd[2], minCmd), maxCmd)
#     cmd[3] = min(max(cmd[3], minCmd), maxCmd)
    
#     # Add Exponential to command
#     # ---------------------------
#     cmd = expoCmd(quad.params, cmd)

#     return cmd

# def expoCmd(params, cmd):
#     if params["ifexpo"]:
#         cmd = np.sqrt(cmd)*10
    
#     return cmd

# def expoCmdInv(params, cmd):
#     if params["ifexpo"]:
#         cmd = (cmd/10)**2
    
#     return cmd
