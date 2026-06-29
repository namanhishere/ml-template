from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch.nn as nn


class BaseExporter(ABC):
    @abstractmethod
    def export(
        self,
        model: nn.Module,
        sample_input: Any,
        output_path: str | Path,
        **kwargs: Any,
    ) -> Path:
        ...
