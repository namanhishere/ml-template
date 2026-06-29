from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.distributed import get_device
from src.utils.seed import count_parameters

logger = logging.getLogger("ai-ml-template")


class LinearProbe:
    """
    Freeze backbone, train linear classifier. Measures representation quality.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        config: dict,
    ) -> None:
        self.backbone = backbone
        self.num_classes = num_classes
        self.config = config
        self.device = get_device()
        self.feature_dim: int | None = None
        self.head: nn.Module | None = None

    def _validate_pretrained(self) -> None:
        bn_found = 0
        bn_random = 0
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                bn_found += 1
                weight = module.weight.data if module.weight is not None else None
                bias = module.bias.data if module.bias is not None else None
                if weight is not None and (weight != 1.0).any():
                    bn_random += 1
                elif bias is not None and (bias != 0.0).any():
                    bn_random += 1
        if bn_found > 0 and bn_random >= bn_found * 0.5:
            logger.warning(
                "Many BatchNorm layers have non-default stats: %d/%d. Backbone may not be pretrained.",
                bn_random,
                bn_found,
            )
        elif bn_found == 0:
            logger.warning("No BatchNorm layers found in backbone — cannot verify pretrained status.")
        else:
            logger.info("BatchNorm layers look pretrained (%d found, %d non-default).", bn_found, bn_random)

    def _infer_feature_dim(self, dataloader: DataLoader) -> int:
        if self.feature_dim is not None:
            return self.feature_dim
        self.backbone.eval()
        self.backbone.to(self.device)
        with torch.no_grad():
            batch = next(iter(dataloader))
            if isinstance(batch, dict):
                x = batch.get("image", batch.get("input"))
            else:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if x is None:
                raise ValueError("Could not extract input from batch")
            x = x[:1].to(self.device)
            features = self.backbone(x)
            if isinstance(features, dict):
                features = features.get("features", next(iter(features.values())))
            features = features.flatten(1)
            self.feature_dim = features.shape[1]
        logger.info("Inferred feature dimension: %d", self.feature_dim)
        return self.feature_dim

    def _build_head(self, feature_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(feature_dim, self.num_classes),
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_epochs: int = 50,
        lr: float = 0.01,
    ) -> dict:
        self._validate_pretrained()
        feature_dim = self._infer_feature_dim(train_loader)

        self.head = self._build_head(feature_dim).to(self.device)
        self.backbone.to(self.device)

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        optimizer = torch.optim.SGD(
            self.head.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=float(self.config.get("optimizer", {}).get("params", {}).get("weight_decay", 1e-4)),
        )

        criterion = nn.CrossEntropyLoss()
        metrics: dict[str, list[float]] = {"train_loss": [], "train_acc": []}
        if val_loader is not None:
            metrics["val_loss"] = []
            metrics["val_acc"] = []

        for epoch in range(max_epochs):
            self.head.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                if isinstance(batch, dict):
                    x = batch.get("image", batch.get("input"))
                    y = batch.get("label", batch.get("target"))
                else:
                    x, y = batch[0], batch[1]
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    features = self.backbone(x)
                    if isinstance(features, dict):
                        features = features.get("features", next(iter(features.values())))
                logits = self.head(features)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

            metrics["train_loss"].append(total_loss / max(len(train_loader), 1))
            metrics["train_acc"].append(correct / max(total, 1))

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                metrics["val_loss"].append(val_metrics["loss"])
                metrics["val_acc"].append(val_metrics["accuracy"])

        final_train_acc = metrics["train_acc"][-1]
        logger.info("Linear probe training complete: train_acc=%.4f", final_train_acc)

        result = {"train_loss": metrics["train_loss"][-1], "train_acc": final_train_acc}
        if val_loader is not None:
            result["val_loss"] = metrics["val_loss"][-1]
            result["val_acc"] = metrics["val_acc"][-1]
        result["metrics_history"] = metrics
        return result

    def evaluate(self, test_loader: DataLoader) -> dict:
        if self.head is None:
            raise RuntimeError("Must call train() before evaluate()")
        self.head.eval()
        self.backbone.eval()

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict):
                    x = batch.get("image", batch.get("input"))
                    y = batch.get("label", batch.get("target"))
                else:
                    x, y = batch[0], batch[1]
                x = x.to(self.device)
                y = y.to(self.device)

                features = self.backbone(x)
                if isinstance(features, dict):
                    features = features.get("features", next(iter(features.values())))
                logits = self.head(features)
                loss = criterion(logits, y)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return {
            "loss": total_loss / max(len(test_loader), 1),
            "accuracy": correct / max(total, 1),
        }

    def compare_backbones(
        self,
        backbones: list[str],
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> "pd.DataFrame":
        import pandas as pd

        from src.models.zoo import BackboneFactory

        results = []
        for name in backbones:
            logger.info("Linear probe on backbone: %s", name)
            backbone = BackboneFactory.create(name, pretrained=True)
            probe = LinearProbe(backbone, self.num_classes, self.config)
            metrics = probe.train(train_loader, val_loader, max_epochs=30, lr=0.01)
            results.append(
                {
                    "backbone": name,
                    "train_acc": metrics["train_acc"],
                    "val_acc": metrics.get("val_acc", float("nan")),
                    "train_loss": metrics["train_loss"],
                    "val_loss": metrics.get("val_loss", float("nan")),
                    "feature_dim": probe.feature_dim,
                    "trainable_params": count_parameters(probe.head, trainable_only=True),
                }
            )

        df = pd.DataFrame(results)
        logger.info("Backbone comparison:\n%s", df.to_string())
        return df
