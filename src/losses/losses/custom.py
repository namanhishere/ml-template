from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.registry import LOSSES


@LOSSES.register(name="dice")
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, multiclass: bool = True) -> None:
        super().__init__()
        self.smooth = smooth
        self.multiclass = multiclass

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.multiclass and preds.dim() > 3:
            n_classes = preds.size(1)
            preds = F.softmax(preds, dim=1)
            targets_one_hot = F.one_hot(targets, num_classes=n_classes).permute(0, 3, 1, 2).float()
            intersection = (preds * targets_one_hot).sum(dim=(0, 2, 3))
            union = preds.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice.mean()
        else:
            preds = torch.sigmoid(preds) if preds.size(1) == 1 else F.softmax(preds, dim=1)
            if preds.size(1) == 1:
                preds = preds.squeeze(1)
            targets = targets.float()
            intersection = (preds * targets).sum()
            union = preds.sum() + targets.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            return 1 - dice


@LOSSES.register(name="combined")
class CombinedLoss(nn.Module):
    def __init__(self, losses: list[dict[str, Any]]) -> None:
        super().__init__()
        self.sub_losses = nn.ModuleList()
        self.weights: list[float] = []
        for entry in losses:
            weight = float(entry.get("weight", 1.0))
            loss_fn = LOSSES.instantiate(name=entry["name"], **(entry.get("params", {})))
            self.sub_losses.append(loss_fn)
            self.weights.append(weight)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=preds.device)
        for weight, loss_fn in zip(self.weights, self.sub_losses):
            total_loss = total_loss + weight * loss_fn(preds, targets)
        return total_loss
