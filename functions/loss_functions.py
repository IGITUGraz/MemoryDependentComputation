"""Loss functions"""

from typing import Tuple

import torch
from torch.nn import functional

from functions.utility_functions import euclidean_distance


class PrototypicalLoss(torch.nn.Module):

    def __init__(self, num_support: int) -> None:
        super().__init__()
        self.num_support = num_support

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return prototypical_loss(inputs, target, self.num_support)


def prototypical_loss(inputs: torch.Tensor, target: torch.Tensor, num_support: int) -> Tuple[
        torch.Tensor, torch.Tensor]:
    target_cpu = target.to('cpu')
    input_cpu = inputs.to('cpu')

    def support_idc(c):
        return target_cpu.eq(c).nonzero()[:num_support].squeeze(1)

    classes = torch.unique(target_cpu)
    n_classes = len(classes)

    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - num_support

    support_idc = list(map(support_idc, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idc])
    query_idc = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[num_support:], classes))).view(-1)

    query_samples = inputs.to('cpu')[query_idc]
    dists = euclidean_distance(query_samples, prototypes)

    log_p_y = torch.nn.functional.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_idc = torch.arange(0, n_classes)
    target_idc = target_idc.view(n_classes, 1, 1)
    target_idc = target_idc.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_idc).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_idc.squeeze()).float().mean()

    return loss_val, acc_val
