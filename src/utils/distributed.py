from __future__ import annotations

import logging
import sys
from typing import Any

import torch

RANK = 0
LOCAL_RANK = 0
WORLD_SIZE = 1
IS_DISTRIBUTED = False


def _detect_distributed() -> None:
    global RANK, LOCAL_RANK, WORLD_SIZE, IS_DISTRIBUTED
    IS_DISTRIBUTED = torch.distributed.is_available() and torch.distributed.is_initialized()
    if IS_DISTRIBUTED:
        RANK = torch.distributed.get_rank()
        WORLD_SIZE = torch.distributed.get_world_size()
        LOCAL_RANK = int(torch.distributed.get_rank() % torch.cuda.device_count() if torch.cuda.is_available() else 0)


def setup_logging(log_level: int = logging.INFO) -> None:
    _detect_distributed()
    formatter = logging.Formatter(
        f"[%(asctime)s][rank={RANK}/{WORLD_SIZE}][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger("ai-ml-template")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)
    root.propagate = False


def is_main_process() -> bool:
    return RANK == 0


def log_main(msg: str, *args: Any, level: int = logging.INFO) -> None:
    if is_main_process():
        logger = logging.getLogger("ai-ml-template")
        logger.log(level, msg, *args)


def barrier() -> None:
    if IS_DISTRIBUTED:
        torch.distributed.barrier()


def gather_tensors(tensor: torch.Tensor, dst: int = 0) -> list[torch.Tensor] | None:
    if not IS_DISTRIBUTED:
        return [tensor]
    gathered = [torch.zeros_like(tensor) for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather(gathered, tensor)
    return gathered


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    if not IS_DISTRIBUTED:
        return tensor
    rt = tensor.clone().detach()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    if average:
        rt /= WORLD_SIZE
    return rt


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{LOCAL_RANK}")
    return torch.device("cpu")


_detect_distributed()
