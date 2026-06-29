from __future__ import annotations

import torch
import torch.nn as nn

from src.models.registry import MODELS


def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


@MODELS.register("conv_demo")
class ConvDemo(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_ch = 3
        for ch in channels:
            layers.append(_conv_block(in_ch, ch))
            in_ch = ch

        self.backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        head_layers: list[nn.Module] = [nn.Flatten(1)]
        if dropout > 0:
            head_layers.append(nn.Dropout(dropout))
        head_layers.append(nn.Linear(in_ch, num_classes))
        self.head = nn.Sequential(*head_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        feats = self.pool(feats)
        logits = self.head(feats)
        return {"logits": logits}
