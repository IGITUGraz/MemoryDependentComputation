"""Spiking dense layer"""

import math
from typing import Tuple, Optional, List, Union

import torch
import torch.nn.functional

from functions.utility_functions import exp_convolve
from models.neuron_models import NeuronModel


class DenseLayer(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, dynamics: NeuronModel,
                 tau_trace: Optional[float] = 20.0) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dynamics = dynamics

        self.decay_trace = math.exp(-1.0 / tau_trace)
        self.W = torch.nn.Parameter(torch.Tensor(hidden_size, input_size))

        self.reset_parameters()

    def forward(self, x: torch.Tensor, states: Optional[List[Union[Tuple[torch.Tensor, ...], torch.Tensor]]] = None) \
            -> Tuple[torch.Tensor, torch.Tensor, List[Union[Tuple[torch.Tensor, ...], torch.Tensor]]]:
        batch_size, sequence_length, *dims = x.size()

        if states is None:
            states = self.dynamics.initial_states(batch_size, self.hidden_size, x.dtype, x.device)
            trace = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            states, trace = states

        i = torch.nn.functional.linear(x, self.W)

        output_sequence, trace_sequence = [], []
        for t in range(sequence_length):
            output, states = self.dynamics(i.select(1, t), states)

            trace = exp_convolve(output, trace, self.decay_trace)

            output_sequence.append(output)
            trace_sequence.append(trace)

        states = [states, trace]

        return torch.stack(output_sequence, dim=1), torch.stack(trace_sequence, dim=1), states

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.W, gain=math.sqrt(2))
