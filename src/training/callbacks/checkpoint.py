from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from src.utils.registry import CALLBACKS

from .registry import Callback

logger = logging.getLogger("ai-ml-template")


@CALLBACKS.register("checkpoint")
class CheckpointCallback(Callback):
    def __init__(
        self,
        save_dir: str | Path = "./checkpoints",
        save_top_k: int = 3,
        save_last: bool = True,
        save_every_n_epochs: int = 1,
        monitor: str = "val/loss",
        mode: str = "min",
    ) -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor = monitor
        self.mode = mode
        self._best_score: float | None = None
        self._best_path: Path | None = None
        self._saved_paths: list[Path] = []
        self._scores: list[tuple[float, Path]] = []

    def on_fit_start(self, trainer: Any) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._best_score = None
        self._best_path = None
        self._saved_paths = []
        self._scores = []

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, Any]) -> None:
        should_save = (epoch + 1) % self.save_every_n_epochs == 0

        if self.save_last:
            should_save = True

        if not should_save:
            return

        checkpoint = trainer._build_checkpoint()
        metric_val = self._extract_metric(metrics)

        if self.save_last:
            last_path = self.save_dir / "last.pt"
            torch.save(checkpoint, last_path)
            logger.info("Saved last checkpoint to %s", last_path)

        epoch_path = self.save_dir / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint, epoch_path)
        logger.info("Saved epoch %d checkpoint to %s", epoch, epoch_path)

        if metric_val is not None:
            is_better = self._is_better(metric_val)
            if is_better:
                if self._best_path is not None and self._best_path.exists():
                    self._best_path.unlink()
                best_path = self.save_dir / "best.pt"
                torch.save(checkpoint, best_path)
                self._best_path = best_path
                self._best_score = metric_val
                logger.info(
                    "New best checkpoint (monitor=%s, score=%.6f) saved to %s",
                    self.monitor,
                    metric_val,
                    best_path,
                )

            self._scores.append((metric_val, epoch_path))
            self._scores.sort(key=lambda x: x[0], reverse=self.mode == "max")
            if len(self._scores) > self.save_top_k and self.save_top_k > 0:
                _, to_remove = self._scores.pop()
                if to_remove != self._best_path and to_remove != (self.save_dir / "last.pt"):
                    if to_remove.exists():
                        to_remove.unlink()
                        logger.debug("Removed old checkpoint %s", to_remove)
        else:
            self._saved_paths.append(epoch_path)

    def _extract_metric(self, metrics: dict[str, Any]) -> float | None:
        val = metrics.get(self.monitor)
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            return val.item()
        return float(val)

    def _is_better(self, current: float) -> bool:
        if self._best_score is None:
            return True
        if self.mode == "min":
            return current < self._best_score
        return current > self._best_score
