"""
评估脚本：加载已训练的网络参数并进行评估
Evaluation script: Load trained network parameters and evaluate
"""
import sys
import os
import numpy as np
from pathlib import Path
import torch

# 添加父目录到路径
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
sys.path.append(parent_dir)

from config import get_config
from envs.env_wrappers import DummyVecEnv

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


def make_eval_env(all_args):
    """创建评估环境"""
    def get_env_fn(rank):
        def init_env():
            from envs.env_continuous import ContinuousActionEnv
            env = ContinuousActionEnv(method=all_args.method)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    # 评估时禁用自动reset，由评估循环手动控制reset
    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)], auto_reset=False)


def parse_args(args, parser):
    """解析评估相关的参数"""
    parser.add_argument("--scenario_name", type=str, default="MyEnv", help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3) #MPE环境
    parser.add_argument("--num_agents", type=int, default=4, help="number of players")
    
    all_args = parser.parse_known_args(args)[0]
    
    # 如果提供了自定义 MLP 结构
    if getattr(all_args, "mlp_hidden_sizes", None):
        all_args.hidden_size = all_args.mlp_hidden_sizes[-1]
        all_args.layer_N = max(len(all_args.mlp_hidden_sizes) - 1, 1)
    
    return all_args


def evaluate_policy(policy, envs, num_episodes=10, deterministic=True, max_steps=None):
    """
    评估策略性能
    
    Args:
        policy: 策略网络对象
        envs: 环境对象
        num_episodes: 评估的回合数
        deterministic: 是否使用确定性策略（True=取均值/模式，False=采样）
        max_steps: 每个episode的最大步数，如果为None则使用默认值550
    
    Returns:
        episode_rewards: 每个回合的总奖励列表（长度为 num_episodes）
        episode_lengths: 每个回合的长度列表（长度为 num_episodes）
        episode_successes: 每个环境的成功状态列表（长度为 num_episodes * num_envs）
    """
    episode_rewards = []
    episode_lengths = []
    episode_successes = []  # 每个环境的成功状态列表

    # 从第一个子环境中获取智能体数量（DummyVecEnv 自身没有 num_agents 属性）
    num_agents = getattr(envs.envs[0], "num_agent", None)
    if num_agents is None:
        raise AttributeError("评估环境 envs.envs[0] 未找到 num_agent 属性，请检查环境封装。")
    
    for episode in range(num_episodes):
        obs = envs.reset()
        episode_reward = 0
        episode_length = 0
        # 为每个环境单独记录成功状态
        env_successes = [False] * envs.num_envs
        
        # 初始化 RNN 状态
        # 注意：即使不使用RNN，也需要初始化 rnn_states 为全零数组（不能为None）
        # 因为 actor 的 forward 方法会尝试对 rnn_states 调用 .to() 方法
        # 参考 runner/shared/env_runner.py 中的实现
        if hasattr(policy, 'actor') and hasattr(policy.actor, '_use_recurrent_policy') and policy.actor._use_recurrent_policy:
            recurrent_N = policy.actor._recurrent_N
            hidden_size = policy.actor.hidden_size
        elif hasattr(policy, 'actor') and hasattr(policy.actor, '_use_naive_recurrent_policy') and policy.actor._use_naive_recurrent_policy:
            recurrent_N = policy.actor._recurrent_N
            hidden_size = policy.actor.hidden_size
        else:
            # 非RNN情况下，使用默认值（这些值不会被实际使用，但需要提供正确的形状）
            recurrent_N = 1
            hidden_size = getattr(policy.actor, 'hidden_size', 64)  # 尝试获取，否则使用默认值
        
        rnn_states = np.zeros(
            (envs.num_envs, num_agents, recurrent_N, hidden_size),
            dtype=np.float32
        )
        
        masks = np.ones((envs.num_envs, num_agents, 1), dtype=np.float32)
        done = False
        
        # 获取最大步数：只从参数获取，如果未提供则使用默认值550
        if max_steps is None:
            max_steps = 550
        
        # 确保 max_steps 是整数
        max_steps = int(max_steps)
        
        step = 0
        while not done and step < max_steps:
            # 准备输入
            # obs 形状: (num_envs, num_agents, obs_dim) - numpy array
            # np.concatenate(obs) 将其展平为 (num_envs * num_agents, obs_dim) 供 policy.act 使用
            # rnn_states 形状: (num_envs, num_agents, recurrent_N, hidden_size)
            # masks 形状: (num_envs, num_agents, 1)
            
            # 参考 runner/shared/env_runner.py: 在调用 policy.act() 之前需要调用 prep_rollout()
            if hasattr(policy, 'prep_rollout'):
                policy.prep_rollout()
            
            actions, rnn_states = policy.act(
                np.concatenate(obs),  # 展平为 (num_envs * num_agents, obs_dim)
                np.concatenate(rnn_states),  # 展平为 (num_envs * num_agents, recurrent_N, hidden_size)
                np.concatenate(masks),  # 展平为 (num_envs * num_agents, 1)
                deterministic=deterministic
            )
            # 将返回的 actions 和 rnn_states 转换为 numpy 并重新分割为每个环境
            # 参考 runner/shared/env_runner.py: 使用 _t2n() 转换 torch tensor
            actions = np.array(np.split(_t2n(actions), envs.num_envs))  # (num_envs, num_agents, action_dim)
            rnn_states = np.array(np.split(_t2n(rnn_states), envs.num_envs))  # (num_envs, num_agents, recurrent_N, hidden_size)
            
            # 处理动作格式（根据环境类型）
            # 参考 runner/shared/env_runner.py 中的动作处理逻辑
            if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                actions_env = []
                for i in range(envs.action_space[0].shape):
                    uc_actions_env = np.eye(envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                    if i == 0:
                        actions_env = uc_actions_env
                    else:
                        actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
            elif envs.action_space[0].__class__.__name__ == "Discrete":
                actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
            else:
                # 连续动作空间，直接使用 actions
                actions_env = actions
            
            # 执行动作
            obs, rewards, dones, infos = envs.step(actions_env)
            
            # 检查 success 信息
            # infos 结构: list[list[dict]]，外层是环境，内层是智能体
            # 为每个环境单独检查成功标志
            for env_idx, env_infos in enumerate(infos):
                if len(env_infos) > 0:
                    # 检查第一个智能体的 success 标志（所有智能体共享同一个 success）
                    # 因为 success 是全局的（围捕成功），所以只需要检查第一个智能体
                    if "success" in env_infos[0] and env_infos[0]["success"]:
                        env_successes[env_idx] = True
            
            episode_reward += np.sum(rewards)
            episode_length += 1
            step += 1
            
            # 更新 masks 和重置 RNN 状态
            # 参考 runner/shared/env_runner.py: dones 的形状是 (num_envs, num_agents)
            # 但需要处理可能的形状差异
            if dones.ndim == 1:
                # 如果 dones 是 (num_envs,)，扩展为 (num_envs, num_agents)
                dones_reshaped = np.tile(dones[:, np.newaxis], (1, num_agents))
            else:
                dones_reshaped = dones
            
            # 重置 RNN 状态：done 的环境-智能体对的 RNN 状态重置为 0
            # 参考 runner/shared/env_runner.py: eval_rnn_states[eval_dones == True] = np.zeros(...)
            rnn_states[dones_reshaped == True] = np.zeros(
                ((dones_reshaped == True).sum(), recurrent_N, hidden_size),
                dtype=np.float32
            )
            
            # 更新 masks：done 的环境-智能体对的 mask 设为 0
            # 参考 runner/shared/env_runner.py: eval_masks[eval_dones == True] = np.zeros(...)
            masks = np.ones((envs.num_envs, num_agents, 1), dtype=np.float32)
            masks[dones_reshaped == True] = np.zeros(((dones_reshaped == True).sum(), 1), dtype=np.float32)
            
            # 检查是否所有环境都结束
            # 参考 runner/shared/env_runner.py: 每个环境的所有智能体都done才算done
            if dones.ndim == 2:
                # 每个环境的所有智能体都done才算done
                env_dones = np.all(dones, axis=1)
                if np.all(env_dones):
                    done = True
            else:
                # dones 是 (num_envs,)，直接检查所有环境
                if np.all(dones):
                    done = True
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        # 将每个环境的成功状态添加到列表中
        episode_successes.extend(env_successes)
        
        # 显示每个环境的成功状态
        success_count_this_episode = sum(env_successes)
        success_status = f"{success_count_this_episode}/{envs.num_envs} 成功"
        print(f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}, Status = {success_status}")
    
    return episode_rewards, episode_lengths, episode_successes


def main(args):
    """主函数：加载模型并评估"""
    parser = get_config()
    all_args = parse_args(args, parser)
    
    # 若评估使用单环境，则自动打开实时三维可视化（支持 PID 和 MPC 环境）
    # 依赖 env_core.py 中对 ENABLE_REALTIME_DISPLAY 的支持
    if getattr(all_args, "n_eval_rollout_threads", 1) == 1:
        # 必须在 import 环境模块之前设置；此处在创建 env 之前执行即可
        os.environ["ENABLE_REALTIME_DISPLAY"] = "True"
        print("[Info] 单评估环境：已设置 ENABLE_REALTIME_DISPLAY=True（实时三维显示）")
    else:
        # 多环境评估时不启用实时显示，以免卡死/刷屏
        os.environ.pop("ENABLE_REALTIME_DISPLAY", None)
        print(f"[Info] n_eval_rollout_threads={all_args.n_eval_rollout_threads}，不启用实时三维显示")
    
    # 检查模型路径
    if all_args.model_dir is None:
        raise ValueError("必须提供 --model_dir 参数来指定模型路径！")
    
    if not os.path.exists(all_args.model_dir):
        raise ValueError(f"模型路径不存在: {all_args.model_dir}")
    
    if not os.path.exists(os.path.join(all_args.model_dir, 'actor.pt')):
        raise ValueError(f"在 {all_args.model_dir} 中找不到 actor.pt 文件！")
    
    print(f"加载模型从: {all_args.model_dir}")
    
    # 设置设备
    if all_args.cuda and torch.cuda.is_available():
        print("使用 GPU...")
        device = torch.device("cuda:0")
    else:
        print("使用 CPU...")
        device = torch.device("cpu")
    
    # 设置随机种子
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    
    # 创建评估环境
    eval_envs = make_eval_env(all_args)
    
    # 获取实际智能体数量
    actual_num_agents = eval_envs.envs[0].num_agent
    if all_args.num_agents != actual_num_agents:
        print(f"[警告] 覆盖 num_agents: 命令行为 {all_args.num_agents}, 环境实际为 {actual_num_agents}")
    all_args.num_agents = actual_num_agents
    
    # 创建策略网络
    from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy
    
    share_observation_space = (
        eval_envs.share_observation_space[0] 
        if all_args.use_centralized_V 
        else eval_envs.observation_space[0]
    )
    
    policy = Policy(
        all_args,
        eval_envs.observation_space[0],
        share_observation_space,
        eval_envs.action_space[0],
        device=device
    )
    
    # 加载模型参数
    print("加载 Actor 网络参数...")
    actor_path = os.path.join(all_args.model_dir, 'actor.pt')
    actor_state_dict = torch.load(actor_path, map_location=device)
    policy.actor.load_state_dict(actor_state_dict)
    print("✓ Actor 网络加载成功")
    
    if not all_args.use_render:
        print("加载 Critic 网络参数...")
        critic_path = os.path.join(all_args.model_dir, 'critic.pt')
        if os.path.exists(critic_path):
            critic_state_dict = torch.load(critic_path, map_location=device)
            policy.critic.load_state_dict(critic_state_dict)
            print("✓ Critic 网络加载成功")
        else:
            print("⚠ Critic 网络文件不存在，跳过加载")
    
    # 设置策略为评估模式
    policy.actor.eval()
    if hasattr(policy, 'critic'):
        policy.critic.eval()
    
    # 执行评估
    print("\n开始评估...")
    print("=" * 50)
    
    num_eval_episodes = getattr(all_args, 'eval_episodes', 10)
    max_episode_steps = getattr(all_args, 'episode_length', 550)  # 从配置获取最大步数
    episode_rewards, episode_lengths, episode_successes = evaluate_policy(
        policy, 
        eval_envs, 
        num_episodes=num_eval_episodes,
        deterministic=True,  # 评估时使用确定性策略
        max_steps=max_episode_steps  # 传入最大步数
    )
    
    # 计算成功率
    # episode_successes 现在包含所有环境的成功状态
    total_env_runs = len(episode_successes)  # 总环境运行次数 = num_episodes * num_envs
    success_count = sum(episode_successes)  # 成功环境数
    success_rate = success_count / total_env_runs * 100.0 if total_env_runs > 0 else 0.0
    
    # 打印统计结果
    print("=" * 50)
    print("\n评估结果统计:")
    print(f"评估回合数: {num_eval_episodes}")
    print(f"并行环境数: {eval_envs.num_envs}")
    print(f"总环境运行次数: {total_env_runs}")
    print(f"成功环境数: {success_count}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"平均奖励: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}")
    print(f"最大奖励: {np.max(episode_rewards):.4f}")
    print(f"最小奖励: {np.min(episode_rewards):.4f}")
    print(f"平均回合长度: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"总奖励: {np.sum(episode_rewards):.4f}")
    
    # 关闭环境
    eval_envs.close()
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_count': success_count,
        'success_rate': success_rate,
        'episode_successes': episode_successes
    }


if __name__ == "__main__":
    main(sys.argv[1:])

