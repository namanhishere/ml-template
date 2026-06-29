from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger("ai-ml-template")


class DatasetCache:
    def __init__(self, cache_dir: str | None = None) -> None:
        if cache_dir is None:
            cache_dir = os.path.join(str(Path.home()), ".cache", "ai-ml-template")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    @staticmethod
    def hash_config(config: dict[str, Any]) -> str:
        serialized = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def _cache_path(self, dataset_name: str, split: str) -> str:
        safe_name = dataset_name.replace("/", "_").replace("\\", "_")
        return os.path.join(self.cache_dir, f"{safe_name}_{split}.pt")

    def get(self, dataset_name: str, split: str) -> Any | None:
        cache_path = self._cache_path(dataset_name, split)
        if not os.path.exists(cache_path):
            logger.debug("Cache miss: %s [%s]", dataset_name, split)
            return None
        try:
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            logger.debug("Cache hit: %s [%s]", dataset_name, split)
            return data
        except Exception as e:
            logger.warning("Failed to load cache for %s [%s]: %s", dataset_name, split, e)
            return None

    def put(self, dataset_name: str, split: str, dataset: Dataset[Any]) -> None:
        cache_path = self._cache_path(dataset_name, split)
        try:
            torch.save(dataset, cache_path)
            logger.info("Cached dataset: %s [%s] -> %s", dataset_name, split, cache_path)
        except Exception as e:
            logger.error("Failed to cache dataset %s [%s]: %s", dataset_name, split, e)

    def exists(self, dataset_name: str, split: str) -> bool:
        return os.path.exists(self._cache_path(dataset_name, split))

    def clear(self, dataset_name: str | None = None) -> None:
        if dataset_name is not None:
            safe_name = dataset_name.replace("/", "_").replace("\\", "_")
            for entry in os.listdir(self.cache_dir):
                if entry.startswith(safe_name):
                    path = os.path.join(self.cache_dir, entry)
                    os.remove(path)
                    logger.info("Removed cache: %s", path)
        else:
            for entry in os.listdir(self.cache_dir):
                path = os.path.join(self.cache_dir, entry)
                os.remove(path)
            logger.info("Cleared entire cache directory: %s", self.cache_dir)
