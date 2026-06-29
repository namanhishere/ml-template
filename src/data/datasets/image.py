from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.data.registry import DATASETS

logger = logging.getLogger("ai-ml-template")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _find_classes(root: str) -> tuple[list[str], dict[str, int]]:
    entries = sorted(entry.name for entry in os.scandir(root) if entry.is_dir() and not entry.name.startswith("."))
    class_to_idx = {name: i for i, name in enumerate(entries)}
    return entries, class_to_idx


def _gather_samples_from_dir(root: str) -> list[tuple[str, int]]:
    _, class_to_idx = _find_classes(root)
    samples: list[tuple[str, int]] = []
    for class_name, idx in class_to_idx.items():
        class_dir = os.path.join(root, class_name)
        for fname in sorted(os.listdir(class_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in _IMAGE_EXTENSIONS:
                samples.append((os.path.join(class_dir, fname), idx))
    return samples


def _gather_samples_from_csv(csv_path: str, root_dir: str | None = None) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    label_map: dict[str, int] = {}
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV file '{csv_path}' is empty.")

        try:
            path_idx = header.index("image_path")
            label_idx = header.index("label")
        except ValueError:
            raise ValueError(f"CSV '{csv_path}' must contain 'image_path' and 'label' columns. Found: {header}")

        for row in reader:
            img_path = row[path_idx]
            label_str = row[label_idx]

            if root_dir is not None:
                img_path = os.path.join(root_dir, img_path)

            try:
                label = int(label_str)
            except ValueError:
                if label_str not in label_map:
                    label_map[label_str] = len(label_map)
                label = label_map[label_str]

            samples.append((img_path, label))

    return samples


@DATASETS.register("image_classification")
class ImageClassificationDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        data_dir: str | None = None,
        csv_file: str | None = None,
        root_dir: str | None = None,
        transform: object | None = None,
        target_transform: object | None = None,
    ) -> None:
        self.transform = transform
        self.target_transform = target_transform

        if data_dir is not None:
            self.samples = _gather_samples_from_dir(data_dir)
            _, self.class_to_idx = _find_classes(data_dir)
            self.classes = sorted(self.class_to_idx.keys())
        elif csv_file is not None:
            self.samples = _gather_samples_from_csv(csv_file, root_dir)
            labels = sorted({label for _, label in self.samples})
            self.class_to_idx = {str(label): label for label in labels}
            self.classes = labels
        else:
            raise ValueError("Either `data_dir` or `csv_file` must be provided.")

        if len(self.samples) == 0:
            raise RuntimeError(f"No image samples found. data_dir={data_dir}, csv_file={csv_file}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)  # type: ignore[assignment]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return {"image": image, "label": label}
