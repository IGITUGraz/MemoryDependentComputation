"""Proximal Policy Optimization (PPO)"""

from typing import Optional, Tuple

import torch

from functions.utility_functions import variance_explained
from storage import rollout_storage


class PPO(object):

    def __init__(self, actor_critic: torch.nn.Module, value_coefficient: float, entropy_coefficient: float,
                 max_grad_norm: float, ppo_clip_param: float, ppo_batches, ppo_epochs: int, learning_rate: float,
                 adam_eps: float, use_clipped_value_loss: Optional[bool] = True) -> None:
        super().__init__()

        self.actor_critic = actor_critic
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm
        self.ppo_clip_param = ppo_clip_param
        self.ppo_batches = ppo_batches
        self.ppo_epochs = ppo_epochs
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate, eps=adam_eps)

    def update(self, rollouts: rollout_storage.RolloutStorage) -> Tuple[float, float, float, float, float]:
        advantages = rollouts.returns[:-1] - rollouts.values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        loss_epoch = 0.0
        value_loss_epoch = 0.0
        policy_loss_epoch = 0.0
        entropy_epoch = 0.0
        var_explained = 0.0

        for epoch in range(self.ppo_epochs):

            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(advantages, self.ppo_batches)
            else:
                data_generator = rollouts.feed_forward_generator(advantages, self.ppo_batches)

            for sample in data_generator:
                observations_batch, old_action_log_probs_batch, actions_batch, values_batch, returns_batch, \
                    masks_batch, advantages_target, recurrent_states_batch = sample

                values, action_log_probs, entropy, _ = self.actor_critic.evaluate_action(
                    action=actions_batch,
                    inputs=observations_batch,
                    states=recurrent_states_batch,
                    mask=masks_batch
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * advantages_target
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_param, 1.0 + self.ppo_clip_param) * advantages_target
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    values_clipped = values_batch + (values - values_batch).\
                        clamp(-self.ppo_clip_param, self.ppo_clip_param)
                    value_losses = (values - returns_batch).pow(2)
                    value_losses_clipped = (values_clipped - returns_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (returns_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                loss = self.value_coefficient * value_loss + policy_loss - self.entropy_coefficient * entropy
                loss.backward()

                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

                self.optimizer.step()

                loss_epoch += loss.item()
                value_loss_epoch += value_loss.item()
                policy_loss_epoch += policy_loss.item()
                entropy_epoch += entropy.item()

                var_explained += variance_explained(values.detach().cpu().numpy(), returns_batch.cpu().numpy())

        num_updates = self.ppo_epochs * self.ppo_batches

        loss_epoch /= num_updates
        value_loss_epoch /= num_updates
        policy_loss_epoch /= num_updates
        entropy_epoch /= num_updates
        var_explained /= num_updates

        return loss_epoch, value_loss_epoch, policy_loss_epoch, entropy_epoch, var_explained
