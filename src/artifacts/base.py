from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ArtifactManager(ABC):
    @abstractmethod
    def save_predictions(self, predictions: dict[str, Any], name: str, step: int | None = None) -> None:
        ...

    @abstractmethod
    def save_plot(self, fig: Any, name: str, step: int | None = None) -> None:
        ...

    @abstractmethod
    def save_confusion_matrix(
        self,
        y_true: Any,
        y_pred: Any,
        class_names: list[str],
        name: str,
        step: int | None = None,
    ) -> None:
        ...

    @abstractmethod
    def save_model(self, model_path: Path, name: str) -> None:
        ...
