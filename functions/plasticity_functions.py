"""Plasticity functions"""

import torch


@torch.jit.script
class InvertedOjaWithSoftUpperBound(object):

    def __init__(self, w_max: float, gamma_pos: float, gamma_neg: float) -> None:
        self.w_max = w_max
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def __call__(self, pre: torch.Tensor, post: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """gamma_pos * (w_max - mem) * ('bi,bj->bij', post, pre) - gamma_neg * ('bij,bj->bij', weights, pre^2)"""

        return self.gamma_pos * (self.w_max - weights) * post.unsqueeze(2) * pre.unsqueeze(1) - \
            self.gamma_neg * pre.unsqueeze(1)**2 * weights
