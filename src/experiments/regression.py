from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.experiments.base import BaseExperiment
from src.experiments.registry import EXPERIMENTS
from src.losses.registry import LOSSES


@EXPERIMENTS.register("regression")
class RegressionExperiment(BaseExperiment):
    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
    ) -> None:
        super().__init__(model, config)

        loss_name = config.get("loss", {}).get("name", "mse")
        loss_params = config.get("loss", {}).get("params", {})
        self.loss_fn = LOSSES.instantiate(loss_name, **loss_params)

        self._batch_mae: list[float] = []
        self._batch_mse: list[float] = []
        self._epoch_metrics: dict[str, list[float]] = defaultdict(list)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        x = batch["image"].to(self.device)
        values = self.model(x)
        if isinstance(values, dict):
            values = values.get("values", values)
        return {"values": values}

    def compute_loss(self, batch: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
        targets = batch["target"].to(self.device)
        values = outputs["values"]
        if values.dim() > targets.dim():
            values = values.squeeze(-1)
        targets = targets.float()
        loss = self.loss_fn(values, targets)
        return {"total": loss}

    def compute_metrics(self, outputs: dict[str, Any], batch: dict[str, Any], phase: str) -> dict[str, float]:
        if outputs is None and batch is None:
            if not self._batch_mae:
                return {}

            mae = float(np.mean(self._batch_mae))
            mse_val = float(np.mean(self._batch_mse))
            rmse = float(np.sqrt(mse_val))

            result = {
                f"{phase}_mae": mae,
                f"{phase}_mse": mse_val,
                f"{phase}_rmse": rmse,
            }
            for k, v in result.items():
                self._epoch_metrics[k].append(v)

            self._batch_mae.clear()
            self._batch_mse.clear()
            return result

        targets = batch["target"].to(self.device, dtype=torch.float32)
        values = outputs["values"].detach()
        if values.dim() > targets.dim():
            values = values.squeeze(-1)

        mae = torch.abs(values - targets).mean().item()
        mse_val = ((values - targets) ** 2).mean().item()

        self._batch_mae.append(mae)
        self._batch_mse.append(mse_val)
        return {}

    def postprocess(self, outputs: dict[str, Any]) -> Any:
        return outputs["values"]

    def on_epoch_end(self, phase: str) -> None:
        pass

    def visualize(
        self, batch: dict[str, Any], outputs: dict[str, Any], save_dir: Path, prefix: str
    ) -> None:
        values = outputs["values"].detach().cpu()
        targets = batch.get("target")
        if targets is None:
            return
        targets = targets.cpu()

        if values.dim() > 1:
            values = values.squeeze(-1)
        if targets.dim() > 1:
            targets = targets.squeeze(-1)

        values_np = values.numpy()[:1000]
        targets_np = targets.numpy()[:1000]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(targets_np, values_np, alpha=0.5, s=8)
        mn = min(targets_np.min(), values_np.min())
        mx = max(targets_np.max(), values_np.max())
        axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1, label="y = x")
        axes[0].set_xlabel("Target")
        axes[0].set_ylabel("Predicted")
        axes[0].set_title("Predicted vs Target")
        axes[0].legend()

        errors = values_np - targets_np
        axes[1].hist(errors, bins=50, alpha=0.7, edgecolor="black")
        axes[1].axvline(0, color="r", linestyle="--", linewidth=1)
        axes[1].set_xlabel("Prediction Error")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(f"Error Distribution\nMAE: {np.abs(errors).mean():.4f}")

        plt.tight_layout()
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"{prefix}_regression_analysis.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def get_epoch_metrics(self) -> dict[str, list[float]]:
        return dict(self._epoch_metrics)

    def clear_epoch_metrics(self) -> None:
        self._epoch_metrics.clear()
