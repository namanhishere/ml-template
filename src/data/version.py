from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger("ai-ml-template")


@dataclass
class DatasetVersion:
    dataset_name: str
    dataset_hash: str
    train_checksum: str
    val_checksum: str
    num_train: int
    num_val: int
    num_classes: int
    file_hashes: dict[str, str] = field(default_factory=dict)


def _hash_file(path: str, chunk_size: int = 65536) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_file_list(data_dir: str, file_paths: list[str]) -> tuple[str, dict[str, str]]:
    hasher = hashlib.sha256()
    file_hashes: dict[str, str] = {}

    sorted_paths = sorted(file_paths)
    for rel_path in sorted_paths:
        full_path = os.path.join(data_dir, rel_path)
        if not os.path.isfile(full_path):
            logger.warning("Skipping missing file: %s", full_path)
            continue

        fhash = _hash_file(full_path)
        file_hashes[rel_path] = fhash
        hasher.update(rel_path.encode("utf-8"))
        hasher.update(fhash.encode("utf-8"))

    return hasher.hexdigest(), file_hashes


def compute_dataset_version(
    dataset_name: str,
    data_dir: str,
    train_files: list[str],
    val_files: list[str],
    num_classes: int = -1,
) -> DatasetVersion:
    train_checksum, train_file_hashes = _hash_file_list(data_dir, train_files)
    val_checksum, val_file_hashes = _hash_file_list(data_dir, val_files)

    combined = hashlib.sha256()
    combined.update(dataset_name.encode("utf-8"))
    combined.update(train_checksum.encode("utf-8"))
    combined.update(val_checksum.encode("utf-8"))
    dataset_hash = combined.hexdigest()[:16]

    file_hashes = {**train_file_hashes, **val_file_hashes}

    return DatasetVersion(
        dataset_name=dataset_name,
        dataset_hash=dataset_hash,
        train_checksum=train_checksum,
        val_checksum=val_checksum,
        num_train=len(train_files),
        num_val=len(val_files),
        num_classes=num_classes,
        file_hashes=file_hashes,
    )
