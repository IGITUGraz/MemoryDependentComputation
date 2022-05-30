"""Embedding layers"""

import math
from typing import Optional

import torch
import torch.nn.functional


class EmbeddingLayer(torch.nn.Module):

    def __init__(self, input_size: int, hidden_size: int, padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx

        self.embedding = torch.nn.Embedding(input_size, hidden_size, padding_idx=padding_idx)

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.embedding.weight.data, gain=math.sqrt(2))

        if self.padding_idx is not None:
            self.embedding.weight.data[self.padding_idx].fill_(0.0)
