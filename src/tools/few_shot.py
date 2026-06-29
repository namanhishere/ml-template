from __future__ import annotations

import copy
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.utils.distributed import get_device
from src.utils.seed import set_seed

logger = logging.getLogger("ai-ml-template")


class FewShotEvaluator:
    """
    Evaluates performance with K samples per class. Subset-based only (no meta-learning).
    """

    def __init__(
        self,
        model_factory,
        experiment_class: type,
        config: dict,
        datamodule: Any,
    ) -> None:
        self.model_factory = model_factory
        self.experiment_class = experiment_class
        self.config = config
        self.datamodule = datamodule
        self.device = get_device()
        self.num_classes = config.get("model", {}).get("num_classes", config.get("dataset", {}).get("num_classes", 10))
        self._class_indices: dict[int, list[int]] | None = None

    def evaluate(
        self,
        k_shots: list[int] | None = None,
        mode: str = "pretrained",
        n_episodes: int = 10,
    ) -> dict:
        if k_shots is None:
            k_shots = [1, 5, 10, 20]

        self._class_indices = self._build_class_indices(self.datamodule.train_dataset)

        results: dict[str, dict] = {}
        for k in k_shots:
            episode_accs: list[float] = []
            for episode in range(n_episodes):
                seed = 42 + episode * 100 + k
                set_seed(seed)
                acc = self._few_shot_episode(k, mode, seed)
                episode_accs.append(acc)
                logger.info("K=%d episode=%d/%d acc=%.4f", k, episode + 1, n_episodes, acc)

            mean_acc = float(np.mean(episode_accs))
            std_acc = float(np.std(episode_accs))
            results[f"k_{k}"] = {
                "mean": mean_acc,
                "std": std_acc,
                "episodes": episode_accs,
            }
            logger.info("K=%d: mean=%.4f std=%.4f (n=%d)", k, mean_acc, std_acc, n_episodes)

        return results

    def _build_class_indices(self, dataset: Dataset) -> dict[int, list[int]]:
        class_indices: dict[int, list[int]] = defaultdict(list)
        for idx in range(len(dataset)):
            item = dataset[idx]
            if isinstance(item, dict):
                label = int(item.get("label", item.get("target", 0)))
            else:
                label = int(item[1])
            class_indices[label].append(idx)

        classes_found = len(class_indices)
        if classes_found < 2:
            logger.warning("Only %d class(es) found in dataset.", classes_found)

        return dict(class_indices)

    def _sample_few_shot_subset(self, dataset: Dataset, k: int, seed: int) -> Subset:
        rng = np.random.RandomState(seed)
        indices: list[int] = []
        for class_id, cls_indices in sorted(self._class_indices.items()):  # type: ignore[union-attr]
            if len(cls_indices) < k:
                sampled = rng.choice(cls_indices, size=k, replace=True).tolist()
            else:
                sampled = rng.choice(cls_indices, size=k, replace=False).tolist()
            indices.extend(sampled)

        rng.shuffle(indices)
        return Subset(dataset, indices[: k * self.num_classes])

    def _few_shot_episode(self, k: int, mode: str, seed: int) -> float:
        dataset = self.datamodule.train_dataset
        val_dataset = self.datamodule.val_dataset

        if mode == "pretrained":
            model = self.model_factory()
        elif mode == "scratch":
            model = self.model_factory()
            for module in model.modules():
                if hasattr(module, "reset_parameters"):
                    try:
                        module.reset_parameters()
                    except Exception:
                        pass
        else:
            raise ValueError(f"Unknown mode: {mode}")

        subset = self._sample_few_shot_subset(dataset, k, seed)

        batch_size = min(4, len(subset))
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        model.to(self.device)
        experiment = self.experiment_class(model=model, config=self.config)
        experiment.to(self.device)

        lr = self.config.get("optimizer", {}).get("params", {}).get("lr", 0.001)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        max_epochs = min(50, max(10, k * 5))

        best_val_acc = 0.0
        for epoch in range(max_epochs):
            model.train()
            for batch in train_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = experiment.forward(batch)
                loss_dict = experiment.compute_loss(batch, outputs)
                loss = loss_dict["total"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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

            if epoch >= 10 and val_acc >= best_val_acc * 0.99 and val_acc > 0:
                break

        return best_val_acc

    def plot_learning_curves(self, results: dict, save_path: Path | None = None) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skipping plot.")
            return

        ks = []
        means = []
        stds = []
        for key in sorted(results.keys()):
            if key.startswith("k_"):
                k = int(key.split("_")[1])
                ks.append(k)
                means.append(results[key]["mean"])
                stds.append(results[key]["std"])

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(ks, means, yerr=stds, fmt="o-", capsize=5, markersize=8, linewidth=2)
        ax.set_xlabel("K-Shots")
        ax.set_ylabel("Validation Accuracy")
        ax.set_title("Few-Shot Learning Curve")
        ax.grid(True, alpha=0.3)

        for i, (k, m) in enumerate(zip(ks, means)):
            ax.annotate(f"{m:.2%}", (k, m), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Few-shot plot saved to %s", save_path)
        plt.close(fig)
