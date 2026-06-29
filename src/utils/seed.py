from __future__ import annotations

import logging
import random
import sys
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_random_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": {},
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            state["cuda_rng"][i] = torch.cuda.get_rng_state(i)
    return state


def restore_random_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    if torch.cuda.is_available():
        for i, cuda_state in state["cuda_rng"].items():
            torch.cuda.set_rng_state(cuda_state, i)


def get_torch_version() -> str:
    return torch.__version__


def get_cuda_version() -> str:
    return torch.version.cuda or "none"


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


logger = logging.getLogger("ai-ml-template")
