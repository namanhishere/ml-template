from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generator, Iterator

import torch.nn as nn

from src.utils.seed import count_parameters


class BaseModel(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.backbone: nn.Module | None = None
        self.head: nn.Module | None = None

    @abstractmethod
    def forward(self, x: Any) -> Dict[str, Any]: ...

    def get_backbone(self) -> nn.Module:
        if self.backbone is None:
            raise RuntimeError("backbone has not been assigned")
        return self.backbone

    def get_head(self) -> nn.Module:
        if self.head is None:
            raise RuntimeError("head has not been assigned")
        return self.head

    def count_params(self) -> int:
        return count_parameters(self, trainable_only=True)

    @contextmanager
    def freezer(self) -> Generator[None, None, None]:
        backbone = self.get_backbone()
        saved = [(p, p.requires_grad) for p in backbone.parameters()]
        for p in backbone.parameters():
            p.requires_grad = False
        try:
            yield
        finally:
            for p, req in saved:
                p.requires_grad = req

    def unfreeze_backbone(self) -> None:
        for p in self.get_backbone().parameters():
            p.requires_grad = True

    def __repr__(self) -> str:
        trainable = self.count_params()
        total = count_parameters(self, trainable_only=False)
        return f"{self.__class__.__name__}(trainable={trainable:,}, total={total:,})"
