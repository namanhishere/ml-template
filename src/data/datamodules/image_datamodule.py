from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig

from src.data.datamodule import BaseDataModule
from src.data.registry import DATASETS
from src.data.transforms import TransformRegistry

logger = logging.getLogger("ai-ml-template")


class ImageDataModule(BaseDataModule):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.dataset_cfg = config.get("dataset", {})
        self.dataset_name: str = self.dataset_cfg.get("name", "")
        self.num_classes_val: int = self.dataset_cfg.get("num_classes", 10)

        augment_name = config.get("augmentation", "basic")
        augment_cfg = config.get("augmentation_config", None)
        self.train_transform, self.val_transform, self.test_transform = TransformRegistry.get_transforms(
            augment_name, augment_cfg
        )

    def setup(self, stage: str | None = None) -> None:
        dataset_kwargs: dict[str, Any] = {k: v for k, v in self.dataset_cfg.items() if k != "name"}

        if stage in ("fit", None):
            if self._train_dataset is None:
                self._train_dataset = DATASETS.instantiate(
                    self.dataset_name, transform=self.train_transform, **dataset_kwargs
                )
            if self._val_dataset is None:
                self._val_dataset = DATASETS.instantiate(
                    self.dataset_name, transform=self.val_transform, **dataset_kwargs
                )

        if stage in ("test", None) and self._test_dataset is None:
            self._test_dataset = DATASETS.instantiate(
                self.dataset_name, transform=self.test_transform, **dataset_kwargs
            )

    @property
    def num_classes(self) -> int:
        return self.num_classes_val
