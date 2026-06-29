from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch.nn as nn


class BaseExperiment(nn.Module, ABC):
    def __init__(self, model: nn.Module, config: dict[str, Any]) -> None:
        super().__init__()
        self.model = model
        self.config = config

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def mode(self) -> str:
        return "train" if self.model.training else "eval"

    def train(self, mode: bool = True) -> BaseExperiment:
        self.model.train(mode)
        return self

    def eval(self) -> BaseExperiment:
        self.model.eval()
        return self

    @abstractmethod
    def forward(self, batch: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def compute_loss(self, batch: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def compute_metrics(self, outputs: dict[str, Any], batch: dict[str, Any], phase: str) -> dict[str, float]: ...

    @abstractmethod
    def visualize(self, batch: dict[str, Any], outputs: dict[str, Any], save_dir: Path, prefix: str) -> None: ...

    @abstractmethod
    def postprocess(self, outputs: dict[str, Any]) -> Any: ...

    def on_epoch_start(self, phase: str) -> None:
        pass

    def on_epoch_end(self, phase: str) -> None:
        pass
