from __future__ import annotations

import copy
import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.distributed import get_device
from src.utils.seed import set_seed

logger = logging.getLogger("ai-ml-template")


class DataAblator:
    """
    Systematically remove data portions and measure impact.
    """

    def __init__(
        self,
        model: nn.Module,
        experiment_class: type,
        config: dict,
        datamodule: Any,
    ) -> None:
        self.model = model
        self.experiment_class = experiment_class
        self.config = config
        self.datamodule = datamodule
        self.device = get_device()
        self.num_classes = config.get("model", {}).get("num_classes", config.get("dataset", {}).get("num_classes", 10))

    def ablate_by_size(self, fractions: list[float] | None = None) -> "pd.DataFrame":
        import pandas as pd

        if fractions is None:
            fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

        train_dataset = self.datamodule.train_dataset
        val_loader = self.datamodule.val_dataloader()
        total = len(train_dataset)
        results = []

        for frac in fractions:
            n_samples = max(1, int(total * frac))
            indices = np.random.RandomState(42).choice(total, size=n_samples, replace=False).tolist()
            subset = Subset(train_dataset, indices)
            train_loader = DataLoader(subset, batch_size=min(32, n_samples), shuffle=True)

            metrics = self._train_on_subset(train_loader, val_loader, epochs=10)
            results.append(
                {
                    "fraction": frac,
                    "n_samples": n_samples,
                    "train_loss": metrics.get("train/loss", float("nan")),
                    "val_loss": metrics.get("val/loss", float("nan")),
                    "accuracy": metrics.get("val/acc", metrics.get("accuracy", float("nan"))),
                }
            )
            logger.info("Size ablation %.0f%%: val_acc=%.4f", frac * 100, results[-1]["accuracy"])

        return pd.DataFrame(results)

    def ablate_by_class(self) -> "pd.DataFrame":
        import pandas as pd

        train_dataset = self.datamodule.train_dataset
        val_loader = self.datamodule.val_dataloader()

        class_indices = self._get_class_indices(train_dataset)
        num_classes = len(class_indices)
        if num_classes < 2:
            logger.warning("Only %d class(es) found; skipping class ablation.", num_classes)
            return pd.DataFrame()

        full_metrics = self._train_on_subset(None, val_loader, epochs=10)
        baseline_acc = full_metrics.get("val/acc", full_metrics.get("accuracy", 0.0))

        results = []
        for drop_class in sorted(class_indices.keys()):
            keep_indices = []
            for cls, indices in class_indices.items():
                if cls != drop_class:
                    keep_indices.extend(indices)
            if not keep_indices:
                continue

            subset = Subset(train_dataset, keep_indices)
            train_loader = DataLoader(subset, batch_size=32, shuffle=True)
            metrics = self._train_on_subset(train_loader, val_loader, epochs=10)
            acc = metrics.get("val/acc", metrics.get("accuracy", 0.0))
            results.append(
                {
                    "dropped_class": drop_class,
                    "n_remaining": len(keep_indices),
                    "accuracy": acc,
                    "delta": baseline_acc - acc,
                }
            )
            logger.info("Class ablation, dropped=%d: acc=%.4f delta=%.4f", drop_class, acc, baseline_acc - acc)

        return pd.DataFrame(results)

    def ablate_by_difficulty(self, n_bins: int = 5) -> "pd.DataFrame":
        import pandas as pd

        train_dataset = self.datamodule.train_dataset
        val_loader = self.datamodule.val_dataloader()

        sample_losses = self._compute_sample_losses(train_dataset)
        if not sample_losses:
            logger.warning("Could not compute sample losses; skipping difficulty ablation.")
            return pd.DataFrame()

        difficulties = np.array(sample_losses)
        bins = np.percentile(difficulties, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(difficulties, bins[:-1]) - 1

        results = []
        for keep_bin in range(n_bins):
            keep_mask = bin_indices != keep_bin
            keep_idx = np.where(keep_mask)[0].tolist()
            if not keep_idx:
                continue

            subset = Subset(train_dataset, keep_idx)
            train_loader = DataLoader(subset, batch_size=32, shuffle=True)
            metrics = self._train_on_subset(train_loader, val_loader, epochs=10)
            acc = metrics.get("val/acc", metrics.get("accuracy", 0.0))
            results.append(
                {
                    "dropped_bin": keep_bin,
                    "difficulty_range": f"[{bins[keep_bin]:.2f}, {bins[min(keep_bin + 1, len(bins) - 1)]:.2f}]",
                    "n_kept": len(keep_idx),
                    "accuracy": acc,
                }
            )
            logger.info("Difficulty ablation, dropped_bin=%d: acc=%.4f", keep_bin, acc)

        return pd.DataFrame(results)

    def compute_influence_scores(self) -> dict:
        train_dataset = self.datamodule.train_dataset
        n_samples = len(train_dataset)

        model = copy.deepcopy(self.model)
        model.to(self.device)
        experiment = self.experiment_class(model=model, config=self.config)
        experiment.to(self.device)

        lr = self.config.get("optimizer", {}).get("params", {}).get("lr", 0.001)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        max_epochs = 10

        sample_losses_per_epoch: list[list[float]] = [[] for _ in range(n_samples)]
        sample_preds_per_epoch: list[list[int]] = [[] for _ in range(n_samples)]

        tracker_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

        for epoch in range(max_epochs):
            model.train()

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            for batch in train_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = experiment.forward(batch)
                loss_dict = experiment.compute_loss(batch, outputs)
                loss = loss_dict["total"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                for idx, batch in enumerate(tracker_loader):
                    if idx >= n_samples:
                        break
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    outputs = experiment.forward(batch)
                    ldict = experiment.compute_loss(batch, outputs)
                    sample_losses_per_epoch[idx].append(ldict["total"].item())

                    if "logits" in outputs:
                        pred = outputs["logits"].argmax(dim=1).item()
                        sample_preds_per_epoch[idx].append(pred)

        difficulty_scores: list[float] = []
        confidence_scores: list[float] = []
        forgetting_scores: list[int] = []
        prediction_flips: list[int] = []

        for idx in range(n_samples):
            losses = sample_losses_per_epoch[idx]
            preds = sample_preds_per_epoch[idx]

            difficulty_scores.append(float(np.mean(losses)) if losses else 0.0)

            if len(losses) > 1:
                forget_count = 0
                for e in range(1, len(preds)):
                    if preds[e] != preds[e - 1]:
                        prediction_flips.append(e)
                flips = sum(1 for e in range(1, len(preds)) if preds[e] != preds[e - 1])
                prediction_flips.append(flips)

                labels = self._get_sample_labels(train_dataset, idx)
                if labels is not None:
                    true_label = labels[idx] if isinstance(labels, list) else labels
                    for e in range(1, len(preds)):
                        was_correct = preds[e - 1] == true_label
                        is_wrong = preds[e] != true_label
                        if was_correct and is_wrong:
                            forget_count += 1
                    forgetting_scores.append(forget_count)
                else:
                    forgetting_scores.append(0)
            else:
                prediction_flips.append(0)
                forgetting_scores.append(0)

            confidence_scores.append(0.0)

        forgetting_ranking = sorted(
            [(i, s) for i, s in enumerate(forgetting_scores)],
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "difficulty_scores": difficulty_scores,
            "mean_difficulty": float(np.mean(difficulty_scores)) if difficulty_scores else 0.0,
            "forgetting_scores": forgetting_scores,
            "mean_forgetting": float(np.mean(forgetting_scores)) if forgetting_scores else 0.0,
            "prediction_flips": prediction_flips,
            "mean_flips": float(np.mean(prediction_flips)) if prediction_flips else 0.0,
            "forgetting_ranking": forgetting_ranking,
            "confidence_scores": confidence_scores,
        }

    def run_full_ablation(self) -> dict:
        results: dict = {}

        logger.info("Running size ablation...")
        results["size_ablation"] = self.ablate_by_size()

        logger.info("Running class ablation...")
        results["class_ablation"] = self.ablate_by_class()

        logger.info("Running difficulty ablation...")
        results["difficulty_ablation"] = self.ablate_by_difficulty()

        logger.info("Computing influence scores...")
        results["influence"] = self.compute_influence_scores()

        return results

    def _train_on_subset(
        self,
        train_loader: DataLoader | None,
        val_loader: DataLoader,
        epochs: int = 10,
    ) -> dict:
        model = copy.deepcopy(self.model)
        model.to(self.device)
        experiment = self.experiment_class(model=model, config=self.config)
        experiment.to(self.device)

        lr = self.config.get("optimizer", {}).get("params", {}).get("lr", 0.001)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        if train_loader is None:
            train_loader = self.datamodule.train_dataloader()

        best_val_acc = 0.0
        final_train_loss = 0.0

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = experiment.forward(batch)
                loss_dict = experiment.compute_loss(batch, outputs)
                loss = loss_dict["total"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            final_train_loss = epoch_loss / max(n_batches, 1)

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    outputs = experiment.forward(batch)
                    if "logits" in outputs:
                        preds = torch.argmax(outputs["logits"], dim=1)
                        labels = batch.get("label")
                        if labels is not None:
                            correct += (preds == labels).sum().item()
                            total += labels.size(0)
            val_acc = correct / max(total, 1)
            best_val_acc = max(best_val_acc, val_acc)

        return {
            "train/loss": final_train_loss,
            "val/loss": 0.0,
            "val/acc": best_val_acc,
            "accuracy": best_val_acc,
        }

    def _get_class_indices(self, dataset: Dataset) -> dict[int, list[int]]:
        class_indices: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            item = dataset[idx]
            if isinstance(item, dict):
                label = int(item.get("label", item.get("target", 0)))
            else:
                label = int(item[1])
            class_indices[label].append(idx)
        return dict(class_indices)

    def _compute_sample_losses(self, dataset: Dataset) -> list[float]:
        model = copy.deepcopy(self.model)
        model.to(self.device)
        model.eval()

        experiment = self.experiment_class(model=model, config=self.config)
        experiment.to(self.device)

        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        losses = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                try:
                    outputs = experiment.forward(batch)
                    ldict = experiment.compute_loss(batch, outputs)
                    losses.append(ldict["total"].item())
                except Exception:
                    losses.append(0.0)
                if len(losses) >= 200:
                    break
        return losses

    @staticmethod
    def _get_sample_labels(dataset: Dataset, max_samples: int = 100) -> list[int] | None:
        try:
            labels = []
            for idx in range(min(max_samples, len(dataset))):
                item = dataset[idx]
                if isinstance(item, dict):
                    labels.append(int(item.get("label", item.get("target", 0))))
                else:
                    labels.append(int(item[1]))
            return labels
        except Exception:
            return None
