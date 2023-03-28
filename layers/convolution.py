"""Spiking 2D convolution and pooling layers"""

import math
from typing import Tuple, Optional

import torch
import torch.nn.functional

from functions.autograd_functions import SpikeFunction
from models.neuron_models import NeuronModel, NonLeakyIafPscDelta


class Conv2DLayer(torch.nn.Module):

    def __init__(self, fan_in: int, fan_out: int, k_size: int, padding: int, stride: int,
                 dynamics: NeuronModel, use_bias: bool = False) -> None:
        super().__init__()
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.k_size = k_size
        self.padding = padding
        self.conv2d = torch.nn.Conv2d(fan_in, fan_out, (k_size, k_size), stride=(stride, stride),
                                      padding=(padding, padding), bias=use_bias)
        self.dynamics = dynamics
        self.reset_parameters()

    def forward(self, x: torch.Tensor, states: Optional[Tuple[torch.Tensor, ...]] = None) -> Tuple[torch.Tensor, ...]:
        batch_size, sequence_length, c, h, w = x.size()
        hidden_size = self.fan_out * h * w
        assert self.fan_in == c

        if states is None:
            states = self.dynamics.initial_states(batch_size, hidden_size, x.dtype, x.device)

        output_sequence, max_activation = [], [-float('inf')]
        for t in range(sequence_length):
            output = torch.flatten(self.conv2d(x.select(1, t)), -3, -1)
            max_activation.append(torch.max(output))
            output, states = self.dynamics(output, states)
            output_sequence.append(output)

        output = torch.reshape(torch.stack(output_sequence, dim=1), [batch_size, sequence_length, self.fan_out, h, w])

        return output, max(max_activation)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.conv2d.weight, gain=math.sqrt(2))


class MaxPool2DLayer(torch.nn.Module):

    def __init__(self, k_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        self.max_pool = torch.nn.MaxPool2d(kernel_size=self.k_size, stride=self.stride, padding=self.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[0] if isinstance(x, Tuple) else x
        batch_size, sequence_length, c, h, w = x.size()

        output_sequence = []
        for t in range(sequence_length):
            output_sequence.append(self.max_pool(x.select(1, t)))

        return torch.stack(output_sequence, dim=1)
