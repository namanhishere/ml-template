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


@EXPERIMENTS.register("segmentation")
class SegmentationExperiment(BaseExperiment):
    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
    ) -> None:
        super().__init__(model, config)
        num_classes = config.get("num_classes", 2)
        self.num_classes = num_classes
        self.ignore_index = config.get("ignore_index", -100)

        dice_loss = LOSSES.instantiate("dice", smooth=config.get("dice_smooth", 1.0), multiclass=True)
        ce_loss = LOSSES.instantiate(
            "cross_entropy",
            ignore_index=self.ignore_index,
        )
        self.ce_loss = ce_loss
        self.dice_loss = dice_loss

        ce_weight = config.get("ce_weight", 0.5)
        dice_weight = config.get("dice_weight", 0.5)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.train_metrics = METRICS.instantiate("segmentation", num_classes=num_classes)
        self.val_metrics = METRICS.instantiate("segmentation", num_classes=num_classes)

        self._epoch_metrics: dict[str, list[float]] = defaultdict(list)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        x = batch["image"].to(self.device)
        logits = self.model(x)
        if isinstance(logits, dict):
            logits = logits.get("logits", logits)
        return {"logits": logits}

    def compute_loss(self, batch: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
        masks = batch["mask"].to(self.device)
        logits = outputs["logits"]

        ce_val = self.ce_loss(logits, masks)
        dice_val = self.dice_loss(logits, masks)

        total = self.ce_weight * ce_val + self.dice_weight * dice_val
        return {"total": total, "dice": dice_val, "ce": ce_val}

    def compute_metrics(self, outputs: dict[str, Any], batch: dict[str, Any], phase: str) -> dict[str, float]:
        if outputs is None and batch is None:
            metrics = self.train_metrics if phase == "train" else self.val_metrics
            computed = metrics.compute()
            result = {}
            for k, v in computed.items():
                key = f"{phase}_{k}"
                result[key] = float(v)
            self._epoch_metrics.update({k: [v] for k, v in result.items()})
            return result

        logits = outputs["logits"]
        masks = batch["mask"].to(self.device)
        preds = torch.argmax(logits, dim=1)

        metrics = self.train_metrics if phase == "train" else self.val_metrics
        metrics.update(preds, masks)
        return {}

    def postprocess(self, outputs: dict[str, Any]) -> Any:
        logits = outputs["logits"]
        return torch.argmax(logits, dim=1)

    def on_epoch_end(self, phase: str) -> None:
        metrics = self.train_metrics if phase == "train" else self.val_metrics
        computed = metrics.compute()
        for k, v in computed.items():
            key = f"{phase}_{k}"
            self._epoch_metrics[key].append(v)
        metrics.reset()

    def visualize(
        self, batch: dict[str, Any], outputs: dict[str, Any], save_dir: Path, prefix: str
    ) -> None:
        images = batch["image"].cpu()
        masks = batch["mask"].cpu()
        logits = outputs["logits"].detach().cpu()
        preds = torch.argmax(logits, dim=1)

        n = min(8, images.size(0))
        images = images[:n]
        masks = masks[:n]
        preds = preds[:n]

        mean = self.config.get("image_mean", [0.0, 0.0, 0.0])
        std = self.config.get("image_std", [1.0, 1.0, 1.0])
        mean = torch.tensor(mean, dtype=images.dtype).view(1, len(mean), 1, 1)
        std = torch.tensor(std, dtype=images.dtype).view(1, len(std), 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0.0, 1.0)

        num_classes = self.num_classes

        np_images = images.permute(0, 2, 3, 1).numpy()
        np_masks = masks.numpy()
        np_preds = preds.numpy()

        cmap = plt.get_cmap("tab20", num_classes)

        fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
        if n == 1:
            axes = np.expand_dims(axes, 0)

        for i in range(n):
            axes[i, 0].imshow(np_images[i])
            axes[i, 0].set_title("Image" if i == 0 else "")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(np_images[i])
            mask_overlay = cmap(np_masks[i].astype(int))
            mask_overlay[..., -1] = 0.4
            axes[i, 1].imshow(mask_overlay)
            axes[i, 1].set_title("Ground Truth" if i == 0 else "")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(np_images[i])
            pred_overlay = cmap(np_preds[i].astype(int))
            pred_overlay[..., -1] = 0.4
            axes[i, 2].imshow(pred_overlay)
            axes[i, 2].set_title("Prediction" if i == 0 else "")
            axes[i, 2].axis("off")

        plt.tight_layout()
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{prefix}_segmentation_samples.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def get_epoch_metrics(self) -> dict[str, list[float]]:
        return dict(self._epoch_metrics)

    def clear_epoch_metrics(self) -> None:
        self._epoch_metrics.clear()
