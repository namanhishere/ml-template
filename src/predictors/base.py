from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BasePredictor(ABC):
    def __init__(
        self,
        model: nn.Module,
        ckpt_path: str | Path | None = None,
        device: str = "cpu",
        experiment: Any = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.experiment = experiment

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

        self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def preprocess(self, raw_input: Any) -> torch.Tensor: ...

    @abstractmethod
    def postprocess(self, outputs: dict[str, Any]) -> Any: ...

    def _forward(self, inputs: torch.Tensor) -> dict[str, Any]:
        if self.experiment is not None:
            return self.experiment({"image": inputs})
        return self.model(inputs)

    def predict(self, input_data: Any) -> Any:
        with torch.no_grad():
            inputs = self.preprocess(input_data)
            outputs = self._forward(inputs)
            return self.postprocess(outputs)

    def predict_batch(self, inputs: list[Any]) -> list[Any]:
        results: list[Any] = []
        with torch.no_grad():
            for item in inputs:
                tensor_in = self.preprocess(item)
                outputs = self._forward(tensor_in)
                results.append(self.postprocess(outputs))
        return results

    def load_checkpoint(self, ckpt_path: str | Path) -> None:
        ckpt = torch.load(str(ckpt_path), map_location=self.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)

    def visualize(self, input_data: Any, save_path: str | Path | None = None) -> Any:
        if self.experiment is not None:
            from pathlib import Path as _Path

            save_dir = _Path(save_path).parent if save_path else _Path(".")
            prefix = _Path(save_path).stem if save_path else "viz"
            outputs = self.predict(input_data)
            return self.experiment.visualize(
                {"image": input_data},
                outputs,
                save_dir=save_dir,
                prefix=prefix,
            )
        return None
