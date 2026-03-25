# from env.sim_env import Quadrotor,QuadSim3D
from envs.random_obstacle_map import generate_map,Map
# from control.control_mpc import Controller
# from parameters import *
import numpy as np
import pickle
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
# ❌ from env.sim_env import Quadrotor
if TYPE_CHECKING:
    from envs import Quadrotor,QuadSim3D

DT = 0.1
N = 10
USE_RBF = True
FILE_NAME="results/data_{}.txt".format(USE_RBF)

def rotate(quat):
    q = np.asarray(quat, dtype=float).reshape(-1, 4)  # 支持(4,)或(N,4)
    w, x, y, z = q.T
    z_axis = np.column_stack([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    ])

    return z_axis

def showmap_(bounds, height, start, goal):
    # bounds = np.array([0, 100])
    size = 250
    density = 2.5

    mapobs, obstacles = generate_map(bounds, density, height, start, goal, size)

    colls = []
    # cube_ids, local_idx = mapobs.obstacles_in_cube(p0, r=20.0)

    # 开启交互模式，确保不阻塞
    plt.ion()
    fig = plt.figure(figsize=(13, 13))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_zlim(bounds[0], bounds[1])

    mapobs.plotobs(ax, scale=1)
    # ax.scatter(p0[0], p0[1], p0[2], s=60, c='g', marker='o')  # 起点 绿色

    ax.zaxis.set_visible(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.grid(False)

    ax.view_init(elev=60, azim=45)
    # plt.tight_layout()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)

    plt.show(block=False)  # 关键：不阻塞
    fig.canvas.draw()
    fig.canvas.flush_events()

    return ax, mapobs

def showmap(bounds, height, start, goal, episode, density, show_points=True, display=False):
    # bounds = np.array([0, 100])
    size = 250
    density = density

    # create map with selected obstacles
    mapobs, obstacles = generate_map(bounds, density, height, start, goal, size,seed_num=42+episode)

    # 开启交互模式，确保不阻塞
    # 调整 figsize 高度，减少上方空白
    fig = plt.figure(figsize=(13, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    # 直接设置 axes 位置，减少上方空白
    # [left, bottom, width, height] 范围都是 0-1
    ax.set_position([0.01, 0.05, 0.99, 0.99])

    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_zlim(bounds[0], bounds[1])
    ax.set_box_aspect((1, 1, 1))

    mapobs.plotobs(ax, scale=1)
    if show_points:
        ax.scatter(start[:-1, 0], start[:-1, 1], start[:-1, 2], s=80, c='g', marker='o', label='start')
        ax.scatter(start[-1, 0], start[-1, 1], start[-1, 2], s=80, c='r', marker='o', label='start_target')
        ax.scatter(goal[0], goal[1], goal[2], s=120, c='orange', marker='*', label='goal')
        ax.legend(loc='upper left', bbox_to_anchor=(0, 0.90), fontsize=14)

    ax.zaxis.set_visible(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.grid(False)

    # ax.view_init(elev=89, azim=45)
    ax.view_init(elev=45, azim=225)

    if not display:
        plt.close(fig)

    return ax, mapobs

if __name__ == "__main__":
    from control.control_mpc import Controller
    from envs.sim_env_mpc import Quadrotor, QuadSim3D
    quad = Quadrotor()
    sim3D = QuadSim3D()
    controller = Controller(quad, n_nodes=N, plan_end_l = -6, dt=DT)
    path = []
    ref = np.array([[3.0, 4.0, 2.0],[4, 6, -1.0],[-3,0,3],[0,0,0]])

    times = []
    ref_log = []
    quat = []
    err = []

    cur_time = 0
    total_time = [10,30,50,70]
    iter = 0
    map_size = 100
    bounds = np.array([0, map_size])

    height = 50

    start = np.array([[0, 0, 5]])
    goal = np.array([65, 65, 5])

    ax, mapobs  = showmap(bounds, height, start, goal, episode=1, display=True,density=5)

    sim3D.init_plot(ax)
    plt.ion()
    plt.show(block=False)

    for i in range(4):
        while (total_time[i] > cur_time):
            x0 = np.concatenate(quad.get_state())
            x_ref = ref[i, :]
            ref_log.append(x_ref)
            start = time.time()
            thrust = controller.compute_control_signal(x0, x_ref,[])
            times.append(time.time() - start)

            # logging.info("Thrust value [0,1]: {}\t{}\t{}\t{}".format(thrust[0], thrust[1], thrust[2], thrust[3]))
            quad.update(thrust, dt=DT)

            # print(quad.pos)
            path.append(quad.pos)
            quat.append(quad.quat)
            err.append(np.linalg.norm(quad.pos - x_ref))

            frame = quad.world_frame()
            sim3D.update_plot(frame, path)

            # 强制重绘当前 figure
            sim3D.ax.figure.canvas.draw()
            sim3D.ax.figure.canvas.flush_events()

            # 给 UI 一点时间渲染
            plt.pause(sim3D.animation_rate)

            cur_time += DT

    plt.ioff()
    plt.show()


    with open(FILE_NAME, 'wb') as file:
        path = np.array(path)
        pose = rotate(quat)
        ref = np.array(ref_log)
        times = np.array(times)
        print("Max processing time: {:.4f}s".format(times.max()))
        print("Min processing time: {:.4f}s".format(times.min()))
        print("Mean processing time: {:.4f}s".format(times.mean()))
        data = dict()
        data['path'] = path
        data['ref'] = ref
        data['times'] = times
        data['pose'] = pose
        data['err'] = np.array(err)
        pickle.dump(data, file)