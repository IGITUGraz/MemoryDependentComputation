"""Encoding layer"""

from typing import Optional, Tuple, List

import torch

from models.neuron_models import NeuronModel


class EncodingLayer(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, mask_time_words: bool, learn_encoding: bool,
                 num_time_steps: int, dynamics: NeuronModel) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_time_steps = num_time_steps
        self.dynamics = dynamics

        self.encoding = torch.nn.Parameter(torch.Tensor(input_size, hidden_size), requires_grad=learn_encoding)

        if mask_time_words and learn_encoding:
            # Set gradient of time-words encoding to zero to avoid modifying them
            def mask_time_words_hook(grad):
                grad_modified = grad.clone()
                grad_modified[-1] = 0.0
                return grad_modified

            self.encoding.register_hook(mask_time_words_hook)

        self.reset_parameters()

    def forward(self, x: torch.Tensor, states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor,
                                                                                             List[torch.Tensor]]:
        batch_size, sequence_length, _, _ = x.size()

        if states is None:
            states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)

        i = torch.sum(self.encoding * x, dim=2)

        output_sequence = []
        for n in range(sequence_length):
            for t in range(self.num_time_steps):
                output, states = self.dynamics(i.select(1, n), states)

                output_sequence.append(output)

        return torch.stack(output_sequence, dim=1), states

    def reset_parameters(self) -> None:
        self.encoding.data.fill_(1.0)
