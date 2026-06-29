from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from src.utils.distributed import IS_DISTRIBUTED, is_main_process, reduce_tensor

logger = logging.getLogger("ai-ml-template")


class Evaluator:
    def __init__(self, experiment: Any, config: Any) -> None:
        self.experiment = experiment
        self.config = config
        self.device = experiment.device

    def evaluate(self, dataloader: Any) -> dict[str, float]:
        model = self.experiment.model
        model.eval()

        self.experiment.on_epoch_start("val")

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = self._to_device(batch, self.device)
                outputs = self.experiment.forward(batch)
                self.experiment.compute_loss(batch, outputs)
                self.experiment.compute_metrics(outputs, batch, "val")

        metrics = self.experiment.compute_metrics(None, None, "val")
        self.experiment.on_epoch_end("val")

        if IS_DISTRIBUTED:
            reduced = {}
            for k, v in metrics.items():
                t = torch.tensor(float(v), device=self.device)
                reduced[k] = float(reduce_tensor(t, average=True).item())
            metrics = reduced

        if is_main_process():
            logger.info("Evaluation metrics: %s", metrics)

        return metrics

    def generate_report(self, metrics: dict[str, float], output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "metrics": metrics,
        }

        if hasattr(self.config, "model"):
            report["model"] = self.config.model.get("name", "unknown") if hasattr(self.config.model, "get") else str(self.config.model)

        report_path = output_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Evaluation report saved to %s", report_path)

    @staticmethod
    def _to_device(batch: Any, device: torch.device) -> Any:
        if isinstance(batch, dict):
            return {k: Evaluator._to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(Evaluator._to_device(v, device) for v in batch)
        elif isinstance(batch, torch.Tensor):
            return batch.to(device, non_blocking=True)
        return batch
