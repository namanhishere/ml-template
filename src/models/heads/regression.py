from __future__ import annotations

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        layers: list[nn.Module] = [
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))

        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
