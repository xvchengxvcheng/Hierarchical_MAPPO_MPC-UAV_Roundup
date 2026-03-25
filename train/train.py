"""
# @Time    : 2021/6/30 10:07 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : train1.py
"""

# !/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

# 获取当前文件的父目录
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))

# 将父目录加入 sys.path，否则下面的 import 会失败
sys.path.append(parent_dir)

from config import get_config
from envs.env_wrappers import DummyVecEnv

"""备用训练脚本，用于并行跑第二个实验。"""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.

            from envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv(method=all_args.method)

            # from envs.env_discrete import DiscreteActionEnv

            # env = DiscreteActionEnv(method=all_args.method)

            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.
            from envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv(method=all_args.method)
            # from envs.env_discrete import DiscreteActionEnv
            # env = DiscreteActionEnv(method=all_args.method)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="MyEnv", help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    # 如果提供了自定义 MLP 结构，则用最后一层维度作为 hidden_size，层数为 len-1
    if getattr(all_args, "mlp_hidden_sizes", None):
        all_args.hidden_size = all_args.mlp_hidden_sizes[-1]
        all_args.layer_N = max(len(all_args.mlp_hidden_sizes) - 1, 1)

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (
            all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False
        ), "check recurrent policy!"
    else:
        raise NotImplementedError

    assert (
        all_args.share_policy == True and all_args.scenario_name == "simple_speaker_listener"
    ) == False, "The simple_speaker_listener scenario can not use shared policy. Please check the config.py."

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir（单独的结果目录，避免与 train.py 冲突）
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results_train1")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # 设置不同的进程标题便于区分
    setproctitle.setproctitle(
        "train1-"
        + str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    # 以环境实际的智能体数量为准，避免命令行参数与环境默认值不一致
    actual_num_agents = envs.envs[0].num_agent
    if all_args.num_agents != actual_num_agents:
        print(
            f"[Warn] 覆盖 num_agents: 命令行为 {all_args.num_agents}, "
            f"环境实际为 {actual_num_agents}"
        )
    all_args.num_agents = actual_num_agents
    num_agents = actual_num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
