from __future__ import annotations

import logging
from typing import Any

import torch

from src.utils.distributed import log_main
from src.utils.registry import CALLBACKS

from .registry import Callback

logger = logging.getLogger("ai-ml-template")


@CALLBACKS.register("early_stop")
class EarlyStopping(Callback):
    def __init__(
        self,
        monitor: str = "val/loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._counter: int = 0
        self._best_score: float | None = None
        self._best_epoch: int = 0

    def on_fit_start(self, trainer: Any) -> None:
        self._counter = 0
        self._best_score = None
        self._best_epoch = 0

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, Any]) -> None:
        current = self._extract_metric(metrics)
        if current is None:
            return

        if self._best_score is None:
            self._best_score = current
            self._best_epoch = epoch
            return

        improved = self._is_improvement(current, self._best_score)

        if improved:
            self._best_score = current
            self._best_epoch = epoch
            self._counter = 0
        else:
            self._counter += 1
            log_main(
                "EarlyStopping: counter %d/%d (best %.6f at epoch %d, current %.6f)",
                self._counter,
                self.patience,
                self._best_score,
                self._best_epoch,
                current,
            )

        if self._counter >= self.patience:
            log_main(
                "EarlyStopping: stopping at epoch %d. Best %.6f at epoch %d",
                epoch,
                self._best_score,
                self._best_epoch,
            )
            trainer._should_stop = True
            trainer._best_metric = {self.monitor: self._best_score}

    def _extract_metric(self, metrics: dict[str, Any]) -> float | None:
        val = metrics.get(self.monitor)
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            return val.item()
        return float(val)

    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < (best - self.min_delta)
        return current > (best + self.min_delta)
