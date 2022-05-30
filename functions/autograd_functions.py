"""Custom autograd functions"""

from typing import Tuple, Any

import torch


class SpikeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, factor: float) -> torch.Tensor: # noqa
        ctx.save_for_backward(inputs)
        ctx.factor = factor

        return torch.greater(inputs, 0.0).type(inputs.dtype)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Tuple[torch.Tensor, None]: # noqa
        inputs, = ctx.saved_tensors
        grad_inputs = grad_outputs.clone()
        return grad_inputs * ctx.factor * torch.maximum(torch.zeros_like(inputs), 1.0 - inputs.abs()), None
