from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import Accuracy, F1Score, JaccardIndex, MetricCollection

from src.training.metrics.registry import METRICS


@METRICS.register("segmentation")
class SegmentationMetrics:
    def __init__(
        self,
        num_classes: int,
        task: str = "multiclass",
    ) -> None:
        self.metrics = MetricCollection(
            {
                "iou": JaccardIndex(task=task, num_classes=num_classes, average="macro"),
                "dice": F1Score(task=task, num_classes=num_classes, average="macro"),
                "accuracy": Accuracy(task=task, num_classes=num_classes, average="micro"),
            }
        )

    def update(self, preds: Tensor, targets: Tensor) -> None:
        if preds.dim() > 3:
            preds = torch.argmax(preds, dim=1)
        self.metrics.update(preds.cpu(), targets.cpu())

    def compute(self) -> dict[str, float]:
        result = self.metrics.compute()
        return {k: float(v) for k, v in result.items()}

    def reset(self) -> None:
        self.metrics.reset()
