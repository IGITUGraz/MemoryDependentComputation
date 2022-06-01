"""Some utility functions"""

import torch


@torch.jit.script
def exp_convolve(x: torch.Tensor, trace: torch.Tensor, decay: float) -> torch.Tensor:
    return (1.0 - decay) * x + decay * trace


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
