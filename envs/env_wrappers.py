"""
# @Time    : 2021/7/1 8:44 上午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_wrappers.py
Modified from OpenAI Baselines code to work with multi-agent envs
"""

from typing import Any


import numpy as np
import multiprocessing as mp
import cloudpickle


def _worker(remote, parent_remote, env_fn_wrapper):
    """
    子进程工作函数，在每个独立的进程中运行一个环境实例。
    """
    parent_remote.close()
    env = env_fn_wrapper.var()
    
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, rew, done, info = env.step(data)
                # 注意：不在子进程中自动 reset，让主进程统一管理 episode 计数
                remote.send((obs, rew, done, info))
            elif cmd == 'reset':
                if data is not None:
                    obs = env.reset(episode=data)
                else:
                    obs = env.reset()
                remote.send(obs)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.share_observation_space, env.action_space))
            elif cmd == 'get_attr':
                # 获取环境对象的属性
                attr_name = data
                if hasattr(env, attr_name):
                    remote.send(getattr(env, attr_name))
                else:
                    remote.send(None)
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('Worker: got KeyboardInterrupt')
    finally:
        env.close()


class CloudpickleWrapper(object):
    """
    使用 cloudpickle 来序列化环境创建函数，以便在子进程中使用。
    """
    def __init__(self, var):
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = cloudpickle.loads(obs)


class _EnvProxy(object):
    """
    环境代理对象，用于在多进程环境下访问环境属性。
    通过管道查询子进程中的环境属性，保持向后兼容性。
    """
    def __init__(self, remote):
        self._remote = remote
        self._cache = {}
    
    def __getattr__(self, name):
        # 如果属性已缓存，直接返回
        if name in self._cache:
            return self._cache[name]
        
        # 通过管道查询属性
        self._remote.send(('get_attr', name))
        value = self._remote.recv()
        
        # 缓存属性值（如果值不为 None）
        if value is not None:
            self._cache[name] = value
        
        return value


# single env
class DummyVecEnv():
    def __init__(self, env_fns, auto_reset=True):
        """
        使用多进程创建向量化环境。
        每个环境在独立的进程中运行。
        使用 spawn context 确保跨平台兼容性（特别是 Windows）。
        
        Args:
            env_fns: 环境工厂函数列表
            auto_reset: 是否在环境结束时自动reset（默认True，评估时设为False）
        """
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False
        self.auto_reset = auto_reset  # 控制是否自动reset
        
        # 使用 spawn context 创建进程和管道，确保跨平台兼容性
        # spawn 方法在 Windows 上是默认的，在 Linux/Mac 上也能正常工作
        ctx = mp.get_context('spawn')
        
        # 使用 context 创建进程间通信的管道
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.ps = [
            ctx.Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        
        # 启动所有子进程
        for p in self.ps:
            p.daemon = True  # 如果主进程退出，子进程也会退出
            p.start()
        
        # 关闭主进程端的工作队列（只保留子进程端）
        for remote in self.work_remotes:
            remote.close()
        
        # 获取第一个环境的空间信息（所有环境应该相同）
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space
        
        # 创建代理环境对象列表，保持向后兼容性
        # 这样 train.py 中的 envs.envs[0].num_agent 仍然可以工作
        self.envs = [_EnvProxy(self.remotes[i]) for i in range(self.num_envs)]
        
        self.actions = None
        self.global_episode = 0

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        """
        异步发送动作到各个子进程。
        """
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
        self.actions = actions

    def step_wait(self):
        """
        等待所有子进程完成 step 操作并收集结果。
        
        注意：虽然子进程完成时间可能不同，但由于我们按顺序从 self.remotes 接收结果，
        所以 results 的顺序与环境的顺序一致（results[i] 对应第 i 个环境的结果）。
        这保证了返回的 obs, rews, dones, infos 的顺序正确。
        """
        self._assert_not_closed()
        # 按顺序接收结果，保证顺序与 self.remotes 的顺序一致
        # 即使某个环境完成得慢，也会按顺序等待，确保结果顺序正确
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        
        obs, rews, dones, infos = zip(*results)
        obs = np.array(obs)
        rews = np.array(rews)
        dones = np.array(dones)
        # infos 保持为列表，因为它是 list[list[dict]] 结构，不能转换为 numpy 数组
        infos = list(infos)
        
        # reset 逻辑：如果环境结束，需要重置（仅在auto_reset=True时）
        if self.auto_reset:
            for i, done in enumerate(dones):
                if "bool" in done.__class__.__name__:
                    # 标量 done
                    if done:
                        obs[i] = self._reset_env_at_index(i)
                else:
                    # 多智能体 done 数组，所有 agent 都 done 才 reset
                    if np.all(done):
                        obs[i] = self._reset_env_at_index(i)
        
        self.actions = None
        return obs, rews, dones, infos

    def _reset_env_at_index(self, i):
        """
        重置第 i 个子 env：
        - 全局 episode 计数 +1
        - 调用子 env.reset(episode=global_episode)
        """
        self.global_episode += 1
        ep = self.global_episode
        
        self.remotes[i].send(('reset', ep))
        obs_i = self.remotes[i].recv()
        return obs_i

    def reset(self):
        """
        为每个子 env 分配一个全局 episode 号，从 1 开始递增：
        env0 -> ep1, env1 -> ep2, ...，之后再根据 done 情况继续递增。
        """
        self._assert_not_closed()
        obs = []
        for i in range(self.num_envs):
            obs_i = self._reset_env_at_index(i)
            obs.append(obs_i)
        return np.array(obs)

    def close(self):
        """
        关闭所有子进程和通信管道。
        """
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def _assert_not_closed(self):
        """
        检查环境是否已关闭。
        """
        assert not self.closed, "Trying to operate on a closed environment"

    def render(self, mode="human"):
        """
        注意：多进程环境下 render 功能受限，因为环境在子进程中。
        如果需要 render，建议使用单进程版本。
        """
        if mode == "rgb_array":
            # 多进程环境下无法直接获取渲染结果
            raise NotImplementedError("Render in multiprocessing mode is not supported. Use single process mode for rendering.")
        elif mode == "human":
            raise NotImplementedError("Render in multiprocessing mode is not supported. Use single process mode for rendering.")
        else:
            raise NotImplementedError