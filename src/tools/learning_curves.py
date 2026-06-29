from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("ai-ml-template")


class LearningCurveAnalyzer:
    """
    Auto-diagnoses training behavior from metrics history.
    """

    def __init__(
        self,
        overfit_window: int = 10,
        plateau_window: int = 15,
        divergence_ratio: float = 3.0,
    ) -> None:
        self.overfit_window = overfit_window
        self.plateau_window = plateau_window
        self.divergence_ratio = divergence_ratio

    def analyze(self, metrics_history: list[dict]) -> dict:
        if not metrics_history:
            return {
                "status": "normal",
                "recommendations": [],
                "early_stop_epoch": None,
                "details": {"error": "No metrics history provided"},
            }

        checks = [
            self._check_overfitting,
            self._check_underfitting,
            self._check_divergence,
            self._check_plateau,
            self._check_high_lr,
            self._check_low_lr,
        ]

        diagnoses: list[dict] = []
        for check_fn in checks:
            result = check_fn(metrics_history)
            if result is not None:
                diagnoses.append(result)

        if not diagnoses:
            status = "normal"
            recommendations = []
        else:
            statuses_by_priority = ["diverging", "overfitting", "high_lr", "low_lr", "underfitting", "plateau"]
            for s in statuses_by_priority:
                if any(d["status"] == s for d in diagnoses):
                    status = s
                    break
            else:
                status = diagnoses[0]["status"]

            recommendations = []
            for d in diagnoses:
                recommendations.extend(d.get("recommendations", []))

        early_stop = self.recommend_early_stopping(metrics_history)

        details: dict = {}
        for d in diagnoses:
            details.update(d.get("details", {}))

        train_losses = self._extract_metric(metrics_history, "train/loss")
        val_losses = self._extract_metric(metrics_history, "val/loss")
        acc_vals = (
            self._extract_metric(metrics_history, "val/acc")
            or self._extract_metric(metrics_history, "val_accuracy")
            or self._extract_metric(metrics_history, "accuracy")
        )

        if train_losses:
            details["final_train_loss"] = train_losses[-1]
            details["initial_train_loss"] = train_losses[0]
            details["train_loss_change_pct"] = (train_losses[-1] - train_losses[0]) / (train_losses[0] + 1e-8) * 100

        if val_losses:
            details["final_val_loss"] = val_losses[-1]
            details["initial_val_loss"] = val_losses[0]
            details["best_val_loss"] = min(val_losses)
            details["best_val_epoch"] = val_losses.index(min(val_losses)) + 1

        if acc_vals:
            details["final_accuracy"] = acc_vals[-1]
            details["best_accuracy"] = max(acc_vals)
            details["best_acc_epoch"] = acc_vals.index(max(acc_vals)) + 1

        return {
            "status": status,
            "recommendations": recommendations,
            "early_stop_epoch": early_stop,
            "details": details,
        }

    def _extract_metric(self, metrics: list[dict], key: str) -> list[float] | None:
        values = []
        for m in metrics:
            v = m.get(key)
            if v is not None:
                values.append(float(v))
        return values if values else None

    def _check_overfitting(self, metrics: list[dict]) -> dict | None:
        train_losses = self._extract_metric(metrics, "train/loss")
        val_losses = self._extract_metric(metrics, "val/loss")

        if train_losses is None or val_losses is None:
            return None
        if len(val_losses) < self.overfit_window:
            return None

        recent_train = train_losses[-self.overfit_window :]
        recent_val = val_losses[-self.overfit_window :]

        train_decreasing = recent_train[-1] < recent_train[0] * 0.95

        val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]
        val_increasing = val_trend > 0

        _, val_slope = np.polyfit(range(len(val_losses)), val_losses, 1)

        if train_decreasing and val_increasing and val_slope > 0:
            gap = val_losses[-1] - train_losses[-1]
            return {
                "status": "overfitting",
                "recommendations": [
                    "Apply stronger regularization (weight_decay, dropout)",
                    "Use early stopping at the best validation loss epoch",
                    "Reduce model capacity or add data augmentation",
                ],
                "details": {
                    "train_loss_trend": "decreasing",
                    "val_loss_trend": "increasing",
                    "train_val_gap": gap,
                    "val_loss_slope": val_slope,
                },
            }
        return None

    def _check_underfitting(self, metrics: list[dict]) -> dict | None:
        train_losses = self._extract_metric(metrics, "train/loss")
        val_losses = self._extract_metric(metrics, "val/loss")

        if train_losses is None or val_losses is None:
            return None
        if len(train_losses) < 5:
            return None

        n = min(5, len(train_losses))
        recent_train = train_losses[-n:]
        recent_val = val_losses[-n:]

        train_slope, _ = np.polyfit(range(n), recent_train, 1)
        val_slope, _ = np.polyfit(range(n), recent_val, 1)

        both_decreasing = train_slope < -1e-4 and val_slope < -1e-4
        train_drop_pct = abs(train_slope * n) / (recent_train[0] + 1e-8) * 100

        if both_decreasing and train_drop_pct > 3.0:
            acc_vals = self._extract_metric(metrics, "val/acc") or self._extract_metric(metrics, "accuracy")
            if acc_vals and acc_vals[-1] < 0.5 * (1.0 - 1e-4):
                return {
                    "status": "underfitting",
                    "recommendations": [
                        "Train for more epochs",
                        "Increase model capacity (wider/larger backbone)",
                        "Reduce regularization (dropout, weight_decay)",
                        "Try a higher learning rate",
                    ],
                    "details": {
                        "train_loss_slope": train_slope,
                        "val_loss_slope": val_slope,
                        "train_drop_pct_per_epoch": train_drop_pct,
                    },
                }
        return None

    def _check_divergence(self, metrics: list[dict]) -> dict | None:
        val_losses = self._extract_metric(metrics, "val/loss")
        if val_losses is None or len(val_losses) < 2:
            return None

        initial_loss = val_losses[0]
        current_loss = val_losses[-1]
        if current_loss > initial_loss * self.divergence_ratio:
            return {
                "status": "diverging",
                "recommendations": [
                    "Reduce learning rate significantly",
                    "Check for exploding gradients (use gradient clipping)",
                    "Verify data preprocessing/normalization",
                    "Try warmup scheduler",
                ],
                "details": {
                    "initial_loss": initial_loss,
                    "current_loss": current_loss,
                    "divergence_ratio": current_loss / (initial_loss + 1e-8),
                    "divergence_threshold": self.divergence_ratio,
                },
            }

        for i in range(len(val_losses) - 1):
            if val_losses[i + 1] > val_losses[i] * 2.0 and val_losses[i + 1] > 1.5 * initial_loss:
                return {
                    "status": "diverging",
                    "recommendations": [
                        "Reduce learning rate significantly",
                        "Check for exploding gradients (use gradient clipping)",
                        "Verify data preprocessing/normalization",
                        "Try warmup scheduler",
                    ],
                    "details": {
                        "initial_loss": initial_loss,
                        "spike_epoch": i + 2,
                        "current_loss": val_losses[i + 1],
                        "divergence_ratio": val_losses[i + 1] / (initial_loss + 1e-8),
                    },
                }
        return None

    def _check_plateau(self, metrics: list[dict]) -> dict | None:
        train_losses = self._extract_metric(metrics, "train/loss")
        val_losses = self._extract_metric(metrics, "val/loss")

        if train_losses is None or val_losses is None:
            return None
        if len(val_losses) < self.plateau_window:
            return None

        recent_train = train_losses[-self.plateau_window :]
        recent_val = val_losses[-self.plateau_window :]

        if len(recent_val) < 2:
            return None

        val_slope, _ = np.polyfit(range(len(recent_val)), recent_val, 1)
        train_slope, _ = np.polyfit(range(len(recent_train)), recent_train, 1)

        train_rel_change = abs(train_slope * len(recent_train)) / (recent_train[0] + 1e-8)
        val_rel_change = abs(val_slope * len(recent_val)) / (recent_val[0] + 1e-8)

        threshold = 0.01

        if train_rel_change < threshold and val_rel_change < threshold:
            return {
                "status": "plateau",
                "recommendations": [
                    "Reduce learning rate (e.g., by factor of 10)",
                    "Use ReduceLROnPlateau scheduler",
                    "Try a different optimizer",
                ],
                "details": {
                    "plateau_epochs": self.plateau_window,
                    "train_rel_change": train_rel_change,
                    "val_rel_change": val_rel_change,
                    "current_val_loss": recent_val[-1],
                },
            }
        return None

    def _check_high_lr(self, metrics: list[dict]) -> dict | None:
        train_losses = self._extract_metric(metrics, "train/loss")
        if train_losses is None or len(train_losses) < 5:
            return None

        diffs = [train_losses[i + 1] - train_losses[i] for i in range(len(train_losses) - 1)]
        if len(diffs) < 3:
            return None

        mean_abs_diff = float(np.mean(np.abs(diffs)))
        mean_loss = np.mean(train_losses)
        oscillation_ratio = mean_abs_diff / (mean_loss + 1e-8)

        sign_changes = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i - 1] < 0)
        sign_change_rate = sign_changes / max(len(diffs) - 1, 1)

        if (oscillation_ratio > 0.15 and sign_change_rate > 0.3) or oscillation_ratio > 0.3:
            return {
                "status": "high_lr",
                "recommendations": [
                    "Reduce learning rate (try 0.5x to 0.1x current)",
                    "Add learning rate warmup",
                    "Use gradient clipping",
                ],
                "details": {
                    "oscillation_ratio": oscillation_ratio,
                    "sign_change_rate": sign_change_rate,
                    "mean_loss": mean_loss,
                },
            }
        return None

    def _check_low_lr(self, metrics: list[dict]) -> dict | None:
        train_losses = self._extract_metric(metrics, "train/loss")
        if train_losses is None or len(train_losses) < 10:
            return None

        total_change_pct = abs(train_losses[-1] - train_losses[0]) / (train_losses[0] + 1e-8) * 100

        if len(train_losses) >= 10 and total_change_pct < 1.0:
            initial_loss = train_losses[0]
            final_loss = train_losses[-1]
            if final_loss > initial_loss * 0.8:
                return {
                    "status": "low_lr",
                    "recommendations": [
                        f"Increase learning rate (current loss change: {total_change_pct:.2f}%)",
                        "Try 2x-10x the current learning rate",
                    ],
                    "details": {
                        "total_loss_change_pct": total_change_pct,
                        "initial_loss": train_losses[0],
                        "final_loss": train_losses[-1],
                        "epochs_examined": len(train_losses),
                    },
                }
        return None

    def recommend_early_stopping(self, metrics: list[dict]) -> int | None:
        val_losses = self._extract_metric(metrics, "val/loss")
        if val_losses is None or len(val_losses) < 5:
            return None

        best_epoch = val_losses.index(min(val_losses)) + 1
        patience = max(self.overfit_window, len(val_losses) // 3)

        if len(val_losses) >= patience + best_epoch:
            recent_min = min(val_losses[-patience:])
            if recent_min > min(val_losses[:best_epoch]) * 1.02:
                return best_epoch

        if len(val_losses) > self.overfit_window:
            recent = val_losses[-self.overfit_window :]
            if all(recent[i + 1] > recent[i] for i in range(len(recent) - 1)):
                return best_epoch

        return None

    def generate_summary(self, diagnosis: dict) -> str:
        status = diagnosis.get("status", "normal")
        icon_map = {
            "normal": "\u2713",
            "overfitting": "\u26a0",
            "underfitting": "\u26a0",
            "diverging": "\u2717",
            "plateau": "\u26a0",
            "high_lr": "\u2717",
            "low_lr": "\u26a0",
        }

        icon = icon_map.get(status, "")
        lines = [f"{icon} Training Status: {status.upper()}"]

        es = diagnosis.get("early_stop_epoch")
        if es is not None:
            lines.append(f"  Early stop recommended at epoch: {es}")
        else:
            lines.append(f"  No early stop recommended")

        recommendations = diagnosis.get("recommendations", [])
        if recommendations:
            lines.append(f"  Recommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"    {i}. {rec}")
        else:
            lines.append(f"  No recommendations needed")

        details = diagnosis.get("details", {})
        if details:
            lines.append("  Metrics summary:")
            for k, v in sorted(details.items()):
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.4f}")
                else:
                    lines.append(f"    {k}: {v}")

        return "\n".join(lines)
