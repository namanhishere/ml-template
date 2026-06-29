from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.artifacts.base import ArtifactManager


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class LocalArtifactManager(ArtifactManager):
    def __init__(self, output_dir: str | Path = Path("outputs")) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _make_filename(self, name: str, ext: str, step: int | None = None) -> str:
        if step is not None:
            return f"{step}_{name}.{ext}"
        return f"{name}.{ext}"

    def save_predictions(self, predictions: dict[str, Any], name: str, step: int | None = None) -> None:
        filepath = self.output_dir / self._make_filename(name, "json", step)
        with open(filepath, "w") as f:
            json.dump(predictions, f, indent=2, cls=_NumpyEncoder)

    def save_plot(self, fig: Any, name: str, step: int | None = None) -> None:
        filepath = self.output_dir / self._make_filename(name, "png", step)
        fig.savefig(filepath, bbox_inches="tight", dpi=150)

    def save_confusion_matrix(
        self,
        y_true: Any,
        y_pred: Any,
        class_names: list[str],
        name: str,
        step: int | None = None,
    ) -> None:
        from src.viz.confusion import plot_confusion_matrix

        filepath = self.output_dir / self._make_filename(name, "png", step)
        plot_confusion_matrix(y_true, y_pred, class_names, save_path=filepath, normalize=True)

    def save_model(self, model_path: Path, name: str) -> None:
        dest = self.output_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, dest)
