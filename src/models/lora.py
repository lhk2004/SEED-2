"""Low-rank adapters that leave the wrapped base layers untouched."""

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """``base(x) + (alpha / rank) * B(A(dropout(x)))``, with ``base`` frozen."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        # Keep optimizer state and adapter master weights in float32 under AMP.
        self.lora_A = nn.Parameter(
            torch.empty(
                rank, base.in_features, device=base.weight.device, dtype=torch.float32
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features, rank, device=base.weight.device, dtype=torch.float32
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        update = nn.functional.linear(
            nn.functional.linear(
                self.dropout(inputs.to(self.lora_A.dtype)), self.lora_A
            ),
            self.lora_B,
        )
        return base_output + (update * self.scaling).to(base_output.dtype)


def add_lora_adapters(
    module: nn.Module, rank: int, alpha: float, dropout: float
) -> None:
    """Replace every ``nn.Linear`` inside ``module`` with a ``LoRALinear``."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
        else:
            add_lora_adapters(child, rank, alpha, dropout)
