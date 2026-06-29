from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.export.base import BaseExporter


class TorchScriptExporter(BaseExporter):
    def export(
        self,
        model: nn.Module,
        sample_input: Any,
        output_path: str | Path,
        method: str = "trace",
        **kwargs: Any,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()

        if method == "trace":
            traced = torch.jit.trace(model, sample_input, **kwargs)
        elif method == "script":
            traced = torch.jit.script(model, **kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'trace' or 'script'.")

        if output_path.suffix not in (".pt", ".pth"):
            output_path = output_path.with_suffix(".torchscript.pt")

        torch.jit.save(traced, str(output_path))

        return output_path
