"""Writing layer"""

import math
from typing import Optional, Tuple, Callable

import torch
import torch.nn.functional

from functions.utility_functions import exp_convolve
from models.neuron_models import NeuronModel


class WritingLayer(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, plasticity_rule: Callable, tau_trace: float,
                 dynamics: NeuronModel) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.plasticity_rule = plasticity_rule
        self.dynamics = dynamics

        self.decay_trace = math.exp(-1.0 / tau_trace)
        self.W = torch.nn.Parameter(torch.Tensor(hidden_size + hidden_size, input_size))

        self.reset_parameters()

    def forward(self, x: torch.Tensor, states: Optional[Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...],
                torch.Tensor, torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, _ = x.size()

        if states is None:
            key_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            val_states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            key_trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            val_trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
            mem = torch.zeros(batch_size, self.hidden_size, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            key_states, val_states, key_trace, val_trace, mem = states

        i = torch.nn.functional.linear(x, self.W)
        ik, iv = i.chunk(2, dim=2)

        key_output_sequence = []
        val_output_sequence = []
        for t in range(sequence_length):

            # Key-layer
            key, key_states = self.dynamics(ik.select(1, t), key_states)

            # Current from key-layer to value-layer ('bij,bj->bi', mem, key)
            ikv_t = 0.2 * (key.unsqueeze(1) * mem).sum(2)

            # Value-layer
            val, val_states = self.dynamics(iv.select(1, t) + ikv_t, val_states)

            # Update traces
            key_trace = exp_convolve(key, key_trace, self.decay_trace)
            val_trace = exp_convolve(val, val_trace, self.decay_trace)

            # Update memory
            delta_mem = self.plasticity_rule(key_trace, val_trace, mem)
            mem = mem + delta_mem

            key_output_sequence.append(key)
            val_output_sequence.append(val)

        return mem, torch.stack(key_output_sequence, dim=1), torch.stack(val_output_sequence, dim=1)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.W, gain=math.sqrt(2))
