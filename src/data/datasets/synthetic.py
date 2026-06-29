from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.registry import DATASETS


@DATASETS.register("synthetic")
class SyntheticDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        num_samples_per_class: int = 100,
        num_classes: int = 10,
        image_size: int = 64,
        seed: int = 42,
        transform: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.image_size = image_size
        self.transform = transform

        g = torch.Generator()
        g.manual_seed(seed)

        total = num_samples_per_class * num_classes
        self._images = torch.randn(total, 3, image_size, image_size, generator=g)
        self._labels = torch.arange(num_classes).repeat_interleave(num_samples_per_class)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image = self._images[idx]
        label = self._labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "label": label}
