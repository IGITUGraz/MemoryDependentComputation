"""Utilities for model checkpoints"""

import os
import shutil
from typing import Union, Optional

import torch


def save_checkpoint(state: dict, is_best: bool, filename: str) -> None:
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


def load_checkpoint(path: str, device: Optional[Union[torch.device, str]] = None) -> dict:
    if device is None:
        return torch.load(path)
    else:
        # Map model to be loaded to specified single device.
        return torch.load(path, map_location=device)
