from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from src.utils.distributed import RANK, WORLD_SIZE, get_device, log_main
from src.utils.seed import get_random_state

logger = logging.getLogger("ai-ml-template")


def build_checkpoint(trainer: Any, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    model_state = trainer.model.state_dict()
    optimizer_state = trainer.optimizer.state_dict() if trainer.optimizer else {}
    scheduler_state = trainer.scheduler.state_dict() if trainer.scheduler else {}
    scaler_state = trainer.scaler.state_dict() if trainer.scaler else {}

    epoch = trainer._current_epoch
    sampler_state = {
        "train": {"sampler_rng_state": None, "sampler_epoch": epoch},
        "val": {"sampler_rng_state": None, "sampler_epoch": epoch},
    }

    best_metric = getattr(trainer, "_best_metric", {}) or {}
    metrics_history = getattr(trainer, "_metrics_history", []) or []
    last_metrics = metrics or {}
    config = OmegaConf.to_container(trainer.config, resolve=True) if hasattr(trainer, "config") else {}
    seed = trainer.config.get("seed", 42) if hasattr(trainer, "config") else 42
    command = " ".join(sys.argv)

    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        git_hash = "unknown"

    try:
        git_diff = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        git_diff = ""

    dataset_version: dict[str, Any] = {}
    dataset_version_attr = getattr(trainer, "_dataset_version", None)
    if dataset_version_attr is not None:
        from dataclasses import asdict

        dataset_version = asdict(dataset_version_attr) if hasattr(dataset_version_attr, "__dataclass_fields__") else {}
    elif hasattr(trainer, "_dataset_version") and isinstance(trainer._dataset_version, dict):
        dataset_version = trainer._dataset_version

    checkpoint = {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "scaler_state": scaler_state,
        "epoch": epoch,
        "global_step": trainer._global_step,
        "dataloader_state": sampler_state,
        "best_metric": best_metric,
        "metrics_history": metrics_history,
        "last_metrics": last_metrics,
        "config": config,
        "seed": seed,
        "command": command,
        "random_state": get_random_state(),
        "framework_version": "0.1.0",
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
        "git_hash": git_hash,
        "git_diff": git_diff,
        "world_size": WORLD_SIZE,
        "local_rank": RANK,
        "timestamp": datetime.now().isoformat(),
        "device": str(get_device()),
        "dataset_version": dataset_version,
    }

    return checkpoint


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    return torch.load(str(path), map_location=device, weights_only=False)


def resume_from_checkpoint(trainer: Any, path: str | Path) -> None:
    checkpoint = load_checkpoint(path, device=str(trainer.device))

    trainer.model.load_state_dict(checkpoint["model_state"])

    if trainer.optimizer and "optimizer_state" in checkpoint:
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])

    if trainer.scheduler and "scheduler_state" in checkpoint:
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state"])

    if trainer.scaler and "scaler_state" in checkpoint and checkpoint["scaler_state"]:
        trainer.scaler.load_state_dict(checkpoint["scaler_state"])

    trainer._current_epoch = checkpoint.get("epoch", 0)
    trainer._global_step = checkpoint.get("global_step", 0)
    trainer._best_metric = checkpoint.get("best_metric", {})
    trainer._metrics_history = checkpoint.get("metrics_history", [])

    ema_state = checkpoint.get("ema_model_state")
    if ema_state is not None:
        for callback in trainer.callbacks:
            if hasattr(callback, "on_load_checkpoint"):
                callback.on_load_checkpoint(trainer, checkpoint)
                break

    log_main(
        "Resumed from checkpoint: epoch=%d step=%d",
        trainer._current_epoch,
        trainer._global_step,
    )
    log_main(
        "Checkpoint created at: %s, pytorch=%s, cuda=%s",
        checkpoint.get("timestamp", "unknown"),
        checkpoint.get("pytorch_version", "unknown"),
        checkpoint.get("cuda_version", "unknown"),
    )
