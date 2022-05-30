"""Some utility functions"""

import torch


@torch.jit.script
def exp_convolve(x: torch.Tensor, trace: torch.Tensor, decay: float) -> torch.Tensor:
    return (1.0 - decay) * x + decay * trace
