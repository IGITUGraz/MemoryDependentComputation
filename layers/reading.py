"""Reading layer"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn.functional

from models.neuron_models import NeuronModel


class ReadingLayer(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, feedback_delay: int, dynamics: NeuronModel) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.feedback_delay = feedback_delay
        self.dynamics = dynamics

        self.W = torch.nn.Parameter(torch.Tensor(hidden_size, input_size + hidden_size))

        self.reset_parameters()

    def forward(self, x: torch.Tensor, mem: torch.Tensor, states: Optional[Tuple[List[torch.Tensor],
                List[torch.Tensor], torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        batch_size, sequence_length, _ = x.size()

        if states is None:
            key_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            val_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            val_buffer = torch.zeros(batch_size, self.feedback_delay, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            key_states, val_states, val_buffer = states

        key_output_sequence = []
        val_output_sequence = []
        for t in range(sequence_length):
            # Compute current from input and previous value to key-layer
            i = torch.nn.functional.linear(torch.cat([x.select(1, t),
                                                      val_buffer.select(1, t % self.feedback_delay)], dim=-1), self.W)

            # Key-layer
            key, key_states = self.dynamics(i, key_states)

            # Current from key-layer to value-layer ('bij,bj->bi', mem, key)
            ikv_t = (key.unsqueeze(1) * mem).sum(2)

            # Value-layer
            val, val_states = self.dynamics(ikv_t, val_states)

            # Update value buffer
            val_buffer[:, t % self.feedback_delay, :] = val

            key_output_sequence.append(key)
            val_output_sequence.append(val)

        states = [key_states, val_states, val_buffer]

        return torch.stack(key_output_sequence, dim=1), torch.stack(val_output_sequence, dim=1), states

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.W, gain=math.sqrt(2))
