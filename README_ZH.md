[English](README.md) | **中文**

# Hierarchical MPC-MAPPO（多无人机围捕）

## 概述

这是一个 3D 多无人机围捕系统，在高层协同规划中结合了 **MAPPO**，在底层安全控制中结合了 **MPC**。该框架使用**基于 LiDAR 的凸可行域构建方法**，以在密集环境中实现高效避障和稳定的协同包围。

## 项目结构

- `algorithms/`：MAPPO 实现与神经网络模块
- `control/`：底层控制器（MPC / PID）以及动力学/几何工具
- `envs/`：仿真环境（包括连续动作环境与无人机交互逻辑）
- `runner/`：训练与评估运行器（buffer、rollout、日志）
- `train/`：训练、评估与可视化脚本
- `pic/`：演示/论文使用的图片与视频
- `results/`：训练输出结果（模型、日志等）

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练

```bash
# 注意：建议将 n_rollout_threads 设为 18（会带来更高的计算/内存开销）。
# 减少并行环境数量可能会轻微降低性能。
python train/train.py --method MPC
python train/train.py --method PID
```

### 评估已训练模型

> 下方路径是本仓库中的示例。请根据你的本地结果目录调整 `--model_dir`。

```bash
python train/evaluate.py --episode_length 550 --model_dir results/MyEnv/MyEnv/mappo/check/run63/models --method MPC --n_eval_rollout_threads 1
python train/evaluate.py --episode_length 550 --model_dir results/MyEnv/MyEnv/mappo/check/run58/models --method PID --n_eval_rollout_threads 1
```

### 绘制训练曲线

```bash
python train/reward_graph.py --log_dirs results/MyEnv/MyEnv/mappo/check/run58/logs results/MyEnv/MyEnv/mappo/check/run63/logs
```

## 结果展示

### 示例快照

- MPC 结果：

![](pic/episode_76_step_191_success_mpc.png)

- PID 结果：

![](pic/episode_76_step_244_success_pid.png)

## 视频

- 稀疏障碍物：[`sparse_obstacle.mp4`](pic/sparse_obstacle.mp4)
- 稠密障碍物：[`dense_obstacle.mp4`](pic/dense_obstacle.mp4)

## 联系方式

- `2381755917@qq.com`
- `xvchengxvcheng@proton.me`（不常用）

## 论文状态

本仓库包含以下论文稿件对应的实验实现与结果：

**Drone Capture in Complex Obstacle Environments: A Hierarchical Framework Combining MPC and RL**

论文当前处于投稿/评审阶段，尚未正式发表。
待论文公开后，我们会在此补充发表 venue、DOI 和/或 arXiv 链接。

## 引用（暂定）

如果你使用了本项目代码，目前请先引用该论文标题：

```bibtex
@misc{drone_capture_hierarchical_mpc_rl,
  title={Drone Capture in Complex Obstacle Environments: A Hierarchical Framework Combining MPC and RL},
  author={xv cheng},
  year={2026},
  note={Manuscript under review}
}
```

## 参考项目

本项目的实现参考并借鉴了以下开源仓库：

- [bobzwik/Quadcopter_SimCon](https://github.com/bobzwik/Quadcopter_SimCon)
- [duynamrcv/quadrotor_mpc_casadi](https://github.com/duynamrcv/quadrotor_mpc_casadi)
- [tinyzqh/light_mappo](https://github.com/tinyzqh/light_mappo)
- [Bharath2/Quadrotor-Simulation](https://github.com/Bharath2/Quadrotor-Simulation)
