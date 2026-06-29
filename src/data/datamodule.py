from __future__ import annotations

import logging
from functools import cached_property
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.utils.distributed import IS_DISTRIBUTED, RANK, WORLD_SIZE
from src.utils.seed import seed_worker, set_seed

logger = logging.getLogger("ai-ml-template")


class BaseDataModule:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.batch_size: int = config.get("batch_size", 32)
        self.num_workers: int = config.get("num_workers", 4)
        self.pin_memory: bool = config.get("pin_memory", True)
        self.drop_last: bool = config.get("drop_last", False)
        self.seed: int = config.get("seed", 42)

        self._train_dataset: Dataset[Any] | None = None
        self._val_dataset: Dataset[Any] | None = None
        self._test_dataset: Dataset[Any] | None = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str | None = None) -> None:
        raise NotImplementedError

    @cached_property
    def generator(self) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self.seed)
        return g

    @property
    def train_dataset(self) -> Dataset[Any]:
        if self._train_dataset is None:
            self.setup("fit")
        if self._train_dataset is None:
            raise RuntimeError("train_dataset not set after setup('fit')")
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, dataset: Dataset[Any]) -> None:
        self._train_dataset = dataset

    @property
    def val_dataset(self) -> Dataset[Any]:
        if self._val_dataset is None:
            self.setup("fit")
        if self._val_dataset is None:
            raise RuntimeError("val_dataset not set after setup('fit')")
        return self._val_dataset

    @val_dataset.setter
    def val_dataset(self, dataset: Dataset[Any]) -> None:
        self._val_dataset = dataset

    @property
    def test_dataset(self) -> Dataset[Any]:
        if self._test_dataset is None:
            self.setup("test")
        if self._test_dataset is None:
            raise RuntimeError("test_dataset not set after setup('test')")
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, dataset: Dataset[Any]) -> None:
        self._test_dataset = dataset

    def _create_dataloader(
        self,
        dataset: Dataset[Any],
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        sampler: DistributedSampler | None = None
        if IS_DISTRIBUTED:
            sampler = DistributedSampler(
                dataset,
                num_replicas=WORLD_SIZE,
                rank=RANK,
                shuffle=shuffle,
                seed=self.seed,
            )
            shuffle = False

        worker_init_fn = seed_worker if self.seed is not None else None

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
            generator=self.generator,
        )

    def train_dataloader(self) -> DataLoader:
        if self._train_dataset is None:
            self.setup("fit")
        return self._create_dataloader(self.train_dataset, shuffle=True, drop_last=self.drop_last)

    def val_dataloader(self) -> DataLoader:
        if self._val_dataset is None:
            self.setup("fit")
        return self._create_dataloader(self.val_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self) -> DataLoader:
        if self._test_dataset is None:
            self.setup("test")
        return self._create_dataloader(self.test_dataset, shuffle=False, drop_last=False)

    def train_dataloader_len(self) -> int:
        total = len(self.train_dataset)
        return total // self.batch_size + (1 if total % self.batch_size != 0 else 0)

    def val_dataloader_len(self) -> int:
        total = len(self.val_dataset)
        return total // self.batch_size + (1 if total % self.batch_size != 0 else 0)

    def test_dataloader_len(self) -> int:
        total = len(self.test_dataset)
        return total // self.batch_size + (1 if total % self.batch_size != 0 else 0)

    @property
    def num_classes(self) -> int:
        raise NotImplementedError
