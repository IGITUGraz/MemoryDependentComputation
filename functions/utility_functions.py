"""Some utility functions"""
from typing import List

import numpy as np
import torch


@torch.jit.script
def exp_convolve(x: torch.Tensor, trace: torch.Tensor, decay: float) -> torch.Tensor:
    return (1.0 - decay) * x + decay * trace


def euclidean_distance(x: torch.Tensor, y: torch.Tensor):
    n = x.size()[0]
    m = y.size()[0]
    d = x.size()[1]
    if d != y.size()[1]:
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class TrimSilence(object):
    """Removes the silence at the beginning and end of the passed audio data.

        This transformation assumes that the audio is normalized.
    """
    def __init__(self, threshold: float) -> None:
        assert 0. <= threshold <= 1.
        self.threshold = threshold

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        start, end = 0, 0

        for i, sample in enumerate(x):
            if abs(sample) > self.threshold:
                start = i
                break

        # Reverse the array for trimming the end
        for i, sample in enumerate(x.flip(dims=(0,))):
            if abs(sample) > self.threshold:
                end = len(x) - i
                break

        return x[start:end]


def variance_explained(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """ Computes fraction of variance that y_pred explains about y.

    Interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


def to_one_hot(x: int, num_values: int) -> List[int]:
    y = [0] * num_values
    y[x] = 1

    return y
