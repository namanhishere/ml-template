from __future__ import annotations

from abc import ABC
from typing import Any

from src.utils.registry import CALLBACKS

__all__ = ["CALLBACKS", "Callback"]


class Callback(ABC):
    def on_fit_start(self, trainer: Any) -> None:
        pass

    def on_fit_end(self, trainer: Any) -> None:
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, Any]) -> None:
        pass

    def on_train_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        pass

    def on_train_batch_end(self, trainer: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        pass

    def on_val_batch_start(self, trainer: Any, batch: Any, batch_idx: int) -> None:
        pass

    def on_val_batch_end(self, trainer: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        pass

    def on_save_checkpoint(self, trainer: Any, checkpoint: dict[str, Any]) -> None:
        pass

    def on_load_checkpoint(self, trainer: Any, checkpoint: dict[str, Any]) -> None:
        pass
