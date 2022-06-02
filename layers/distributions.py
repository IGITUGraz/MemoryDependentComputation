"""Distribution interfaces"""

from typing import Optional, Any

import torch


class FixedCategorical(torch.distributions.Categorical):

    def rsample(self, sample_shape: Optional[torch.Size] = torch.Size()) -> torch.Tensor:
        return super().rsample(sample_shape)

    def cdf(self, value: torch.Tensor) -> torch.Tensor:
        return super().cdf(value)

    def icdf(self, value: torch.Tensor) -> torch.Tensor:
        return super().icdf(value)

    def sample(self, sample_shape: Optional[torch.Size] = torch.Size()) -> torch.Tensor:
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions: torch.Tensor) -> torch.Tensor:
        return (
            super().log_prob(actions.squeeze(-1))
            .view(actions.size()[0], -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self) -> torch.Tensor:
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(torch.nn.Module):

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        self.linear = torch.nn.Linear(input_size, output_size)

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> Any:
        x = self.linear(x)

        return FixedCategorical(logits=x)

    def reset_parameters(self) -> None:
        torch.nn.init.orthogonal_(self.linear.weight.data, gain=0.01)
        torch.nn.init.constant_(self.linear.bias.data, val=0.0)
