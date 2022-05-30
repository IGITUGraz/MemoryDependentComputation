"""Utilities for training and evaluation metrics"""

from typing import Tuple, Union, List

import torch


def accuracy(output: torch.Tensor, target: torch.Tensor, top_k: Tuple = (1,)) -> Union[float, List[float]]:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size()[0]

        _, pred = output.topk(k=max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res if len(top_k) > 1 else res[0]
