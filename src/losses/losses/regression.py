from __future__ import annotations

import torch
import torch.nn as nn

from src.losses.registry import LOSSES


@LOSSES.register(name="mse")
class MSELossWrapper(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(preds, targets.float())


@LOSSES.register(name="mae")
class L1LossWrapper(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(preds, targets.float())


@LOSSES.register(name="huber")
class HuberLossWrapper(nn.Module):
    def __init__(self, beta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.loss = nn.SmoothL1Loss(beta=beta, reduction=reduction)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(preds, targets.float())
