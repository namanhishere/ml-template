import pytest
import torch
import numpy as np
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture
def seed():
    from src.utils.seed import set_seed
    set_seed(42)

@pytest.fixture
def dummy_image_batch():
    return {"image": torch.randn(4, 3, 224, 224), "label": torch.randint(0, 10, (4,))}

@pytest.fixture
def dummy_config():
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "seed": 42, "max_epochs": 10, "batch_size": 4, "num_workers": 0,
        "experiment": {"name": "classification", "num_classes": 10},
        "model": {"name": "resnet18", "pretrained": False, "num_classes": 10, "backbone": "torchvision://resnet18"},
        "dataset": {"name": "image_classification", "data_dir": "./data", "image_size": 224, "num_classes": 10},
        "loss": {"name": "cross_entropy"},
        "optimizer": {"name": "adam", "params": {"lr": 0.001}},
        "scheduler": {"name": "cosine", "T_max": 10},
        "callbacks": [{"name": "checkpoint"}, {"name": "early_stop"}, {"name": "progress"}],
        "trainer": {"max_epochs": 10, "batch_size": 4, "num_workers": 0, "precision": "fp32", "grad_accumulation": 1}
    })
