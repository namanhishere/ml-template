"""Albumentations plugin — integrates albumentations transforms."""
import logging
import numpy as np
from torchvision import transforms as T

def get_augmentations(config: dict, phase: str = "train"):
    """Build albumentations transform pipeline from config."""
    try:
        import albumentations as A
        aug_list = []
        if phase == "train" and config.get("augmentation"):
            for aug in config["augmentation"]:
                name = aug["name"]
                params = aug.get("params", {})
                aug_list.append(getattr(A, name)(**params))
        aug_list.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        aug_list.append(ToTensorV2_albu())
        return A.Compose(aug_list)
    except ImportError:
        logging.getLogger("ai-ml-template").warning("albumentations not installed, falling back to torchvision transforms")
        return T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class ToTensorV2_albu:
    """Adapter to make albumentations output tensors instead of numpy."""
    def __call__(self, **kwargs):
        import torch
        return {k: torch.from_numpy(v).permute(2, 0, 1) if isinstance(v, np.ndarray) and v.ndim == 3 else v for k, v in kwargs.items()}
