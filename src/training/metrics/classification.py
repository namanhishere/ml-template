from __future__ import annotations

import torch
from torch import Tensor
from torchmetrics import AUROC, Accuracy, F1Score, MetricCollection, Precision, Recall

from src.training.metrics.registry import METRICS


@METRICS.register("classification")
class ClassificationMetrics:
    def __init__(
        self,
        num_classes: int,
        average: str = "macro",
        task: str = "multiclass",
    ) -> None:
        self.metrics = MetricCollection(
            {
                "accuracy": Accuracy(task=task, num_classes=num_classes, average="micro"),
                "f1_score": F1Score(task=task, num_classes=num_classes, average=average),
                "precision": Precision(task=task, num_classes=num_classes, average=average),
                "recall": Recall(task=task, num_classes=num_classes, average=average),
                "auroc": AUROC(task=task, num_classes=num_classes, average=average),
            }
        )

    def update(self, preds: Tensor, targets: Tensor) -> None:
        self.metrics.update(preds.cpu(), targets.cpu())

    def compute(self) -> dict[str, float]:
        result = self.metrics.compute()
        return {k: float(v) for k, v in result.items()}

    def reset(self) -> None:
        self.metrics.reset()
