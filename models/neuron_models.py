"""Spiking neuron models"""

import math
from abc import ABC
from abc import abstractmethod
from typing import Tuple, Type

import torch
import torch.nn.functional

from functions.autograd_functions import SpikeFunction


class NeuronModel(ABC):

    @abstractmethod
    def __call__(self, x: torch.Tensor, states: Tuple[torch.Tensor, ...]) -> Tuple[
            torch.Tensor, Tuple[torch.Tensor, ...]]:
        pass

    @staticmethod
    @abstractmethod
    def initial_states(batch_size: int, hidden_size: int, dtype: Type[torch.dtype],
                       device: Type[torch.device]) -> Tuple[torch.Tensor, ...]:
        pass


class IafPscDelta(NeuronModel):

    def __init__(self, thr: float = 1.0, perfect_reset: bool = False, refractory_time_steps: int = 1,
                 tau_mem: float = 20.0, spike_function: Type[torch.autograd.Function] = SpikeFunction,
                 dampening_factor: float = 1.0) -> None:
        super().__init__()
        self.thr = thr
        self.perfect_reset = perfect_reset
        self.refractory_time_steps = refractory_time_steps

        self.decay_mem = math.exp(-1.0 / tau_mem)

        self.spike_function = lambda x: spike_function.apply(x, dampening_factor)

    def __call__(self, x_t: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[
            torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        z_t, v_t, r_t = states

        is_refractory = torch.greater(r_t, 0.0)

        # Integrate membrane.
        v_t, v_scaled = self.integrate(x_t, z_t, v_t, self.thr, self.perfect_reset, self.decay_mem)

        # Spike generation.
        z_t = self.spike_function(v_scaled)
        z_t = torch.where(is_refractory, torch.zeros_like(z_t), z_t)

        # Update refractory period.
        r_t = torch.where(is_refractory, (r_t - 1.0).clamp(0.0, self.refractory_time_steps),
                          self.refractory_time_steps * z_t)

        return z_t, (z_t, v_t, r_t)

    @staticmethod
    @torch.jit.script
    def integrate(x_t: torch.Tensor, z_t: torch.Tensor, v_t: torch.Tensor, thr: float, perfect_reset: bool,
                  decay_mem: float) -> Tuple[torch.Tensor, torch.Tensor]:
        if perfect_reset:
            v_t = decay_mem * v_t * (1.0 - z_t) + (1.0 - decay_mem) * x_t
        else:
            v_t = decay_mem * v_t + (1.0 - decay_mem) * x_t - z_t * thr

        v_scaled = (v_t - thr) / thr

        return v_t, v_scaled

    @staticmethod
    def initial_states(batch_size: int, hidden_size: int, dtype: torch.dtype,
                       device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
                torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
                torch.zeros(batch_size, hidden_size, dtype=dtype, device=device))
