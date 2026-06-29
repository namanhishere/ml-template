from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.utils import make_grid

from src.experiments.base import BaseExperiment
from src.experiments.registry import EXPERIMENTS
from src.losses.registry import LOSSES
from src.training.metrics.registry import METRICS


@EXPERIMENTS.register("classification")
class ClassificationExperiment(BaseExperiment):
    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        num_classes: int,
        loss_config: dict[str, Any],
        metric_prefix: str = "",
    ) -> None:
        super().__init__(model, config)
        self.num_classes = num_classes
        self.metric_prefix = metric_prefix
        self.loss_fn = LOSSES.instantiate(
            loss_config.get("name", "cross_entropy"),
            **(loss_config.get("params", {})),
        )

        metric_config = {"num_classes": num_classes}
        metric_config.update(config.get("metric_params", {}))
        self.train_metrics = METRICS.instantiate("classification", **metric_config)
        self.val_metrics = METRICS.instantiate("classification", **metric_config)

        self._epoch_metrics: dict[str, list[float]] = defaultdict(list)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        x = batch["image"].to(self.device)
        logits = self.model(x)
        if isinstance(logits, dict):
            logits = logits.get("logits", logits)
        return {"logits": logits}

    def compute_loss(self, batch: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
        labels = batch["label"].to(self.device)
        logits = outputs["logits"]
        loss = self.loss_fn(logits, labels)
        return {"total": loss, "ce": loss}

    def compute_metrics(self, outputs: dict[str, Any], batch: dict[str, Any], phase: str) -> dict[str, float]:
        if outputs is None or batch is None:
            return {}
        logits = outputs["logits"]
        labels = batch["label"].to(self.device)
        probs = torch.softmax(logits, dim=1)

        metrics = self.train_metrics if phase == "train" else self.val_metrics
        metrics.update(probs, labels)
        return {}

    def postprocess(self, outputs: dict[str, Any]) -> Any:
        logits = outputs["logits"]
        return torch.argmax(logits, dim=1)

    def on_epoch_end(self, phase: str) -> None:
        metrics = self.train_metrics if phase == "train" else self.val_metrics
        computed = metrics.compute()
        for k, v in computed.items():
            key = f"{self.metric_prefix}{phase}_{k}" if self.metric_prefix else f"{phase}_{k}"
            self._epoch_metrics[key].append(v)
        metrics.reset()

    def visualize(self, batch: dict[str, Any], outputs: dict[str, Any], save_dir: Path, prefix: str) -> None:
        images = batch["image"].cpu()
        labels = batch["label"].cpu()
        preds = torch.argmax(outputs["logits"].detach().cpu(), dim=1)

        n = min(16, images.size(0))
        images = images[:n]
        labels = labels[:n]
        preds = preds[:n]

        mean = self.config.get("image_mean", [0.0, 0.0, 0.0])
        std = self.config.get("image_std", [1.0, 1.0, 1.0])
        mean = torch.tensor(mean, dtype=images.dtype).view(1, len(mean), 1, 1)
        std = torch.tensor(std, dtype=images.dtype).view(1, len(std), 1, 1)

        images = images * std + mean
        images = torch.clamp(images, 0.0, 1.0)

        class_names = self.config.get("class_names")
        grid = make_grid(images, nrow=4, normalize=False)
        grid = grid.permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(grid)
        ax.axis("off")

        titles = []
        for i in range(n):
            pred_label = preds[i].item()
            true_label = labels[i].item()
            if class_names and pred_label < len(class_names) and true_label < len(class_names):
                t = f"P:{class_names[pred_label]}\nT:{class_names[true_label]}"
            else:
                t = f"P:{pred_label} T:{true_label}"
            titles.append(t)

        colors = ["green" if p == l else "red" for p, l in zip(preds.tolist(), labels.tolist())]
        cols = 4
        rows = (n + cols - 1) // cols
        for idx in range(n):
            r, c = divmod(idx, cols)
            x = c / cols + 0.5 / cols
            y = 1.0 - (r + 0.92) / rows
            ax.text(
                x,
                y,
                titles[idx],
                transform=ax.transAxes,
                fontsize=6,
                ha="center",
                va="top",
                color=colors[idx],
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7),
            )

        save_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        fig.savefig(save_dir / f"{prefix}_class_samples.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def get_epoch_metrics(self) -> dict[str, list[float]]:
        return dict(self._epoch_metrics)

    def clear_epoch_metrics(self) -> None:
        self._epoch_metrics.clear()
