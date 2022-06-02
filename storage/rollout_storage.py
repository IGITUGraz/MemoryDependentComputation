"""Rollout storage"""

from typing import Tuple, Union, Optional

import gym
import torch
import torch.utils.data


class RolloutStorage(object):

    def __init__(self, num_steps: int, num_environments: int, observations_size: Tuple[int, ...],
                 action_space: gym.Space, recurrent_state_size: Tuple[Tuple[int, ...]]) -> None:
        super().__init__()
        self.observations = torch.zeros(num_steps + 1, num_environments, *observations_size)
        self.action_log_probs = torch.zeros(num_steps, num_environments, 1)

        if isinstance(action_space, gym.spaces.Discrete):
            self.actions = torch.zeros(num_steps, num_environments, 1).long()
        else:
            self.actions = torch.zeros(num_steps, num_environments, action_space.shape[0])

        self.values = torch.zeros(num_steps + 1, num_environments, 1)
        self.rewards = torch.zeros(num_steps, num_environments, 1)
        self.returns = torch.zeros(num_steps + 1, num_environments, 1)

        self.masks = torch.ones(num_steps + 1, num_environments, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_environments, 1)

        self.recurrent_states = [torch.zeros(num_steps + 1, num_environments, *size) for size in recurrent_state_size]

        self.num_steps = num_steps
        self.step = 0

    @staticmethod
    def _flatten(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(t * n, *tensor.size()[2:])

    def to(self, device: Union[str, torch.device]) -> None:
        self.observations = self.observations.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.values = self.values.to(device)
        self.rewards = self.rewards.to(device)
        self.returns = self.returns.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.recurrent_states = [state.to(device) for state in self.recurrent_states]

    def insert(self, observation, action_log_prob, action, value, reward, mask, bad_mask, recurrent_states) -> None:
        self.observations[self.step + 1].copy_(observation)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.actions[self.step].copy_(action)
        self.values[self.step].copy_(value)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)
        self.bad_masks[self.step + 1].copy_(bad_mask)

        for i, state in enumerate(recurrent_states):
            self.recurrent_states[i][self.step + 1].copy_(state)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self) -> None:
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

        for i, state in enumerate(self.recurrent_states):
            self.recurrent_states[i][0].copy_(state[-1])

    def compute_returns(self, next_value: torch.Tensor, gamma: float, use_proper_time_limits: Optional[bool] = True) \
            -> None:
        if use_proper_time_limits:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size()[0])):
                self.returns[step] = (self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]) * \
                                     self.bad_masks[step + 1] + (1 - self.bad_masks[step + 1]) * self.values[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size()[0])):
                self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages: torch.Tensor, num_mini_batches: Optional[int] = None,
                               mini_batch_size: Optional[int] = None) -> Tuple:
        num_steps, num_environments = self.rewards.size()[0:2]
        batch_size = num_environments * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batches, (
                "PPO requires the number of processes ({}) * number of steps ({}) = {} to be greater than or equal to "
                "the number of PPO mini batches ({}).".format(num_environments, num_steps, num_environments *
                                                              num_steps, num_mini_batches))
            mini_batch_size = batch_size // num_mini_batches

        sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)

        for indices in sampler:
            observations_batch = self.observations[:-1].view(-1, *self.observations.size()[2:])[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            actions_batch = self.actions.view(-1, self.actions.size()[-1])[indices]
            values_batch = self.values[:-1].view(-1, 1)[indices]
            returns_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            recurrent_states_batch = [state[:-1].view(-1, state.size()[-1])[indices] for state in self.recurrent_states]

            if advantages is None:
                advantages_target = None
            else:
                advantages_target = advantages.view(-1, 1)[indices]

            yield observations_batch, old_action_log_probs_batch, actions_batch, values_batch, returns_batch,\
                masks_batch, advantages_target, recurrent_states_batch

    def recurrent_generator(self, advantages: torch.Tensor, num_mini_batches: int) -> Tuple:
        num_environments = self.rewards.size()[1]

        assert num_environments >= num_mini_batches, (
            "PPO requires the number of processes ({}) to be greater than or equal to the number of PPO mini batches "
            "({}).".format(num_environments, num_mini_batches))

        perm = torch.randperm(num_environments)
        num_envs_per_batch = num_environments // num_mini_batches
        for start_ind in range(0, num_environments, num_envs_per_batch):
            observations_batch = []
            old_action_log_probs_batch = []
            actions_batch = []
            values_batch = []
            returns_batch = []
            masks_batch = []
            advantages_target = []
            recurrent_states_batch = [[] for _ in range(len(self.recurrent_states))]

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                observations_batch.append(self.observations[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                actions_batch.append(self.actions[:, ind])
                values_batch.append(self.values[:-1, ind])
                returns_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                advantages_target.append(advantages[:, ind])

                for i, state in enumerate(self.recurrent_states):
                    recurrent_states_batch[i].append(state[0:1, ind])

            t, n = self.num_steps, num_envs_per_batch

            # These are all tensors of size (t, n, -1)
            observations_batch = torch.stack(observations_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            values_batch = torch.stack(values_batch, 1)
            returns_batch = torch.stack(returns_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            advantages_target = torch.stack(advantages_target, 1)

            # Each state is just a (n, -1) tensor
            recurrent_states_batch = [torch.cat(states, dim=0) for states in recurrent_states_batch]

            # Flatten the (t, n, ...) tensors to (t * n, ...)
            observations_batch = self._flatten(t, n, observations_batch)
            old_action_log_probs_batch = self._flatten(t, n, old_action_log_probs_batch)
            actions_batch = self._flatten(t, n, actions_batch)
            values_batch = self._flatten(t, n, values_batch)
            returns_batch = self._flatten(t, n, returns_batch)
            masks_batch = self._flatten(t, n, masks_batch)
            advantages_target = self._flatten(t, n, advantages_target)

            yield observations_batch, old_action_log_probs_batch, actions_batch, values_batch, returns_batch,\
                masks_batch, advantages_target, recurrent_states_batch
