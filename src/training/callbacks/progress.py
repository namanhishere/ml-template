from __future__ import annotations

import logging
from typing import Any

from tqdm import tqdm

from src.utils.distributed import is_main_process
from src.utils.registry import CALLBACKS

from .registry import Callback

logger = logging.getLogger("ai-ml-template")


@CALLBACKS.register("progress")
class ProgressCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self._total_epochs: int = 0
        self._train_bar: tqdm | None = None

    def on_fit_start(self, trainer: Any) -> None:
        self._total_epochs = getattr(trainer.config, "max_epochs", 0)
        if is_main_process():
            logger.info("Training for %d epochs", self._total_epochs)

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        if not is_main_process():
            return

        dataloader = trainer.datamodule.train_dataloader()
        desc = f"Epoch {epoch + 1}/{self._total_epochs}"
        self._train_bar = tqdm(
            dataloader,
            desc=desc,
            dynamic_ncols=True,
            leave=True,
        )

    def on_train_batch_end(self, trainer: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if not is_main_process() or self._train_bar is None:
            return

        if isinstance(outputs, dict) and "total" in outputs:
            loss_val = outputs["total"]
            if hasattr(loss_val, "item"):
                loss_val = loss_val.item()
            self._train_bar.set_postfix(loss=f"{loss_val:.4f}")

        self._train_bar.update(1)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, Any]) -> None:
        if not is_main_process():
            return

        if self._train_bar is not None:
            self._train_bar.close()
            self._train_bar = None

        formatted = {}
        for k, v in sorted(metrics.items()):
            if isinstance(v, (int, float)):
                formatted[k.split("/")[-1]] = f"{v:.4f}"

        if formatted:
            log_parts = " | ".join(f"{k}={v}" for k, v in formatted.items())
            logger.info("Epoch %d complete: %s", epoch, log_parts)
