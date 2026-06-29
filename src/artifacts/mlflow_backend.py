from __future__ import annotations

from pathlib import Path
from typing import Any

from src.artifacts.base import ArtifactManager


class MLflowArtifactManager(ArtifactManager):
    def __init__(self) -> None:
        try:
            import mlflow  # noqa: F401
        except ImportError:
            raise ImportError("mlflow is required for MLflowArtifactManager. Install it with: pip install mlflow")

    def save_predictions(self, predictions: dict[str, Any], name: str, step: int | None = None) -> None:
        import mlflow

        mlflow.log_dict(predictions, f"{name}.json")

    def save_plot(self, fig: Any, name: str, step: int | None = None) -> None:
        import mlflow

        mlflow.log_figure(fig, f"{name}.png")

    def save_confusion_matrix(
        self,
        y_true: Any,
        y_pred: Any,
        class_names: list[str],
        name: str,
        step: int | None = None,
    ) -> None:
        import mlflow

        from src.viz.confusion import plot_confusion_matrix

        fig = plot_confusion_matrix(y_true, y_pred, class_names, normalize=True)
        mlflow.log_figure(fig, f"{name}.png")

    def save_model(self, model_path: Path, name: str) -> None:
        import mlflow

        mlflow.log_artifact(str(model_path), artifact_path=name)
