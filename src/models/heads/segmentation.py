from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        upsample_factor: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.conv2 = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        if self.upsample_factor is not None and self.upsample_factor > 1:
            x = F.interpolate(
                x,
                scale_factor=self.upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
        return x
