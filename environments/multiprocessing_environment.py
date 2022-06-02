"""Pytorch vector environment"""

from typing import Union, Tuple, List

import gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnvWrapper, SubprocVecEnv, DummyVecEnv


class PyTorchVecEnv(VecEnvWrapper):

    def __init__(self, venv: Union[DummyVecEnv, SubprocVecEnv], device: Union[str, torch.device]) -> None:
        super().__init__(venv)
        self.device = device

    def reset(self) -> torch.Tensor:
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)

        return obs

    def step_async(self, actions: torch.Tensor) -> None:
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[dict]]:
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()

        return obs, reward, done, info


def make_env(env: Union[str, gym.Env], seed: int, rank: int) -> callable:
    def _make():
        if isinstance(env, gym.Env):
            env.seed(seed + rank)
        elif isinstance(env, str):
            raise NotImplementedError

        return env

    return _make


def make_vec_envs(env: Union[str, gym.Env], num_envs: int, seed: int,
                  device: Union[str, torch.device]) -> PyTorchVecEnv:

    envs = [make_env(env, seed, i) for i in range(num_envs)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = PyTorchVecEnv(envs, device)

    return envs
