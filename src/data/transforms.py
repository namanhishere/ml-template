from __future__ import annotations

from typing import Any

import torchvision.transforms as T


class TransformRegistry:
    _presets: dict[str, dict[str, Any]] = {
        "imagenet": {
            "train": [
                {"name": "RandomResizedCrop", "size": 224},
                {"name": "RandomHorizontalFlip", "p": 0.5},
                {"name": "ColorJitter", "brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1},
                {"name": "ToTensor"},
                {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            ],
            "val": [
                {"name": "Resize", "size": 256},
                {"name": "CenterCrop", "size": 224},
                {"name": "ToTensor"},
                {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            ],
            "test": [
                {"name": "Resize", "size": 256},
                {"name": "CenterCrop", "size": 224},
                {"name": "ToTensor"},
                {"name": "Normalize", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            ],
        },
        "basic": {
            "train": [
                {"name": "Resize", "size": [256, 256]},
                {"name": "ToTensor"},
            ],
            "val": [
                {"name": "Resize", "size": [256, 256]},
                {"name": "ToTensor"},
            ],
            "test": [
                {"name": "Resize", "size": [256, 256]},
                {"name": "ToTensor"},
            ],
        },
    }

    _transform_map: dict[str, type] = {
        "RandomResizedCrop": T.RandomResizedCrop,
        "RandomHorizontalFlip": T.RandomHorizontalFlip,
        "ColorJitter": T.ColorJitter,
        "ToTensor": T.ToTensor,
        "Normalize": T.Normalize,
        "Resize": T.Resize,
        "CenterCrop": T.CenterCrop,
        "RandomCrop": T.RandomCrop,
        "RandomVerticalFlip": T.RandomVerticalFlip,
        "RandomRotation": T.RandomRotation,
        "RandomAffine": T.RandomAffine,
        "RandomGrayscale": T.RandomGrayscale,
        "RandomPerspective": T.RandomPerspective,
        "RandomErasing": T.RandomErasing,
        "GaussianBlur": T.GaussianBlur,
        "RandomInvert": T.RandomInvert,
        "RandomPosterize": T.RandomPosterize,
        "RandomSolarize": T.RandomSolarize,
        "RandomAdjustSharpness": T.RandomAdjustSharpness,
        "RandomAutocontrast": T.RandomAutocontrast,
        "RandomEqualize": T.RandomEqualize,
        "Lambda": T.Lambda,
        "Pad": T.Pad,
        "Grayscale": T.Grayscale,
        "ConvertImageDtype": T.ConvertImageDtype,
    }

    @classmethod
    def _build_from_spec(cls, spec: list[dict[str, Any]]) -> T.Compose:
        transforms: list[Any] = []
        for item in spec:
            name = item["name"]
            if name == "ToTensor":
                transforms.append(T.ToTensor())
                continue
            kwargs = {k: v for k, v in item.items() if k != "name"}
            cls_ = cls._transform_map.get(name)
            if cls_ is None:
                raise ValueError(f"Unknown transform '{name}'. Available: {list(cls._transform_map.keys())}")
            transforms.append(cls_(**kwargs))
        return T.Compose(transforms)

    @classmethod
    def _resolve_pipeline(cls, name: str, augmentation_config: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
        if name in cls._presets:
            return cls._presets[name]

        if augmentation_config is None:
            raise ValueError(
                f"Unknown preset '{name}'. Available: {list(cls._presets.keys())}. "
                "Provide an augmentation_config for custom transforms."
            )

        if "train" not in augmentation_config or "val" not in augmentation_config:
            raise ValueError("augmentation_config must contain 'train' and 'val' keys.")

        test_spec = augmentation_config.get("test", augmentation_config["val"])
        return {
            "train": augmentation_config["train"],
            "val": augmentation_config["val"],
            "test": test_spec,
        }

    @classmethod
    def get_transforms(
        cls, name: str, augmentation_config: dict[str, Any] | None = None
    ) -> tuple[T.Compose, T.Compose, T.Compose]:
        pipeline = cls._resolve_pipeline(name, augmentation_config)
        train_transform = cls._build_from_spec(pipeline["train"])
        val_transform = cls._build_from_spec(pipeline["val"])
        test_transform = cls._build_from_spec(pipeline["test"])
        return train_transform, val_transform, test_transform
