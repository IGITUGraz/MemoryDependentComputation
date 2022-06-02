"""Policy class"""

from typing import Optional, Tuple, List

import gym
import torch

from layers.distributions import Categorical


class Policy(torch.nn.Module):
    def __init__(self, action_space: gym.Space, net: torch.nn.Module) -> None:
        super().__init__()

        if isinstance(action_space, gym.spaces.Discrete):
            output_size = action_space.n
            self.dist = Categorical(net.output_size, output_size)
        else:
            raise NotImplementedError

        self.net = net

    @property
    def is_recurrent(self) -> bool:
        return self.net.is_recurrent

    @property
    def state_size(self) -> Tuple[Tuple[int, ...]]:
        if self.is_recurrent:
            return self.net.state_size
        return (1,),

    def forward(self, inputs: torch.Tensor, states: List[torch.Tensor], mask: torch.Tensor) -> NotImplementedError:
        raise NotImplementedError

    def get_action(self, inputs: torch.Tensor, states: List[torch.Tensor], mask: torch.Tensor,
                   deterministic: Optional[bool] = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                   List[torch.Tensor]]:
        value, actor_features, states = self.net(inputs, states, mask)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, states

    def get_value(self, inputs: torch.Tensor, states: List[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        value, _, _ = self.net(inputs, states, mask)

        return value

    def evaluate_action(self, inputs: torch.Tensor, states: List[torch.Tensor], mask: torch.Tensor,
                        action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        value, actor_features, states = self.net(inputs, states, mask)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, states


class RNNBase(torch.nn.Module):

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    @property
    def state_size(self) -> Tuple[Tuple[int, ...]]:
        return self.net.state_size

    @property
    def output_size(self) -> int:
        return self.net.output_size

    def _forward(self, inputs: torch.Tensor, states: List[torch.Tensor], mask: torch.Tensor) -> Tuple[
            torch.Tensor, List[torch.Tensor]]:

        if inputs.size()[0] == states[0].size()[0]:
            states = [state * mask.view(-1, *(1,) * (state.dim()-1)) for state in states]
            outputs, states = self.net(inputs.unsqueeze(0), states)
            outputs = outputs.squeeze(0)
        else:
            # Inputs is a (t, n, -1) tensor that has been flatten to (t * n, -1)
            n = states[0].size()[0]
            t = int(inputs.size()[0] / n)

            inputs = inputs.view(t, n, inputs.size()[1])
            mask = mask.view(t, n)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((mask[1:] == 0.0).any(dim=-1).nonzero(as_tuple=False).squeeze().cpu())

            # +1 to correct the mask[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # Add t=0 and t=t to the list
            has_zeros = [0] + has_zeros + [t]

            output_sequence, layer_outputs_sequence = [], []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in mask together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                states = [state * mask[start_idx].view(-1, *(1,) * (state.dim() - 1)) for state in states]
                outputs, states = self.net(inputs[start_idx:end_idx], states)

                output_sequence.append(outputs)

            outputs = torch.cat(output_sequence, dim=0)
            outputs = outputs.view(t * n, -1)

        return outputs, states
