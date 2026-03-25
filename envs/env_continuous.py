import gym
from gym import spaces
import numpy as np


class ContinuousActionEnv(object):
    """
    对于连续动作环境的封装
    Wrapper for continuous action environment.
    """

    def __init__(self, method="MPC", **kwargs):
        """
        Args:
            method: 控制器类型，'PID' 或 'MPC'，默认为 'MPC'（与 config.py 中的默认值一致）
            **kwargs: 传递给 EnvCore 的其他参数（如 num_agents, episode_offset, enable_realtime_display）
        """
        # 统一使用 env_core.py，通过 controller_type 参数切换
        from envs.env_core import EnvCore
        
        # 将 method 转换为 controller_type（支持大小写）
        controller_type = method.upper() if method else "MPC"
        if controller_type not in ["PID", "MPC"]:
            raise ValueError(f"Unsupported method: {method}. Supported methods are 'PID' and 'MPC'.")
        
        # 传递 controller_type 和其他参数给 EnvCore
        self.env = EnvCore(controller_type=controller_type, **kwargs)
        self.num_agent = self.env.agent_num

        self.signal_obs_dim = self.env.obs_dim
        self.signal_action_dim = self.env.action_dim

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False

        self.movable = True

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        share_obs_dim = 0
        total_action_space = []
        for agent in range(self.num_agent):
            # physical action space
            u_action_space = spaces.Box(
                low=-np.inf,
                high=+np.inf,
                shape=(self.signal_action_dim,),
                dtype=np.float32,
            )

            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            self.action_space.append(total_action_space[0])

            # observation space
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.signal_obs_dim,),
                    dtype=np.float32,
                )
            )  # [-inf,inf]

        self.share_observation_space = [
            spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32
            )
            for _ in range(self.num_agent)
        ]

    def step(self, actions):
        """
        输入actions维度假设：
        # actions shape = (5, 2, 5)
        # 5个线程的环境，里面有2个智能体，每个智能体的动作是一个one_hot的5维编码

        Input actions dimension assumption:
        # actions shape = (5, 2, 5)
        # 5 threads of environment, there are 2 agents inside, and each agent's action is a 5-dimensional one_hot encoding
        """

        results = self.env.step(actions)
        obs, rews, dones, infos = results
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, episode=None):
        obs = self.env.reset(episode=episode)
        return np.stack(obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass

    def seed(self, seed):
        self.env.seed(seed)
