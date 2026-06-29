from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure


def _maybe_binarize(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if y_true.ndim == 1:
        from sklearn.preprocessing import label_binarize

        classes = sorted(np.unique(y_true))
        n_classes = len(classes)
        y_true_bin = label_binarize(y_true, classes=classes)
        return y_true_bin, y_scores, n_classes
    return y_true, y_scores, y_true.shape[1]


def plot_roc_curves(
    y_true: Any,
    y_scores: Any,
    class_names: list[str],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> Figure:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)
    y_true_bin, y_scores_arr_use, n_classes = _maybe_binarize(y_true_arr, y_scores_arr)

    if len(class_names) < n_classes:
        class_names = [f"Class {i}" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores_arr_use[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def plot_pr_curves(
    y_true: Any,
    y_scores: Any,
    class_names: list[str],
    save_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> Figure:
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score

    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)
    y_true_bin, y_scores_arr_use, n_classes = _maybe_binarize(y_true_arr, y_scores_arr)

    if len(class_names) < n_classes:
        class_names = [f"Class {i}" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_scores_arr_use[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_scores_arr_use[:, i])
        ax.plot(recall, precision, linewidth=2, label=f"{class_names[i]} (AP={ap:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (One-vs-Rest)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig
