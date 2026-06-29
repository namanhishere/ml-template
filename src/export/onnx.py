from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn

from src.export.base import BaseExporter


class ONNXExporter(BaseExporter):
    def export(
        self,
        model: nn.Module,
        sample_input: Any,
        output_path: str | Path,
        opset_version: int = 17,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        **kwargs: Any,
    ) -> Path:
        try:
            import onnx  # noqa: F401
        except ImportError:
            raise ImportError("onnx is required for ONNX export. Install it with: pip install onnx onnxruntime")

        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()

        torch.onnx.export(
            model,
            sample_input,
            str(output_path),
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            **kwargs,
        )

        return output_path
