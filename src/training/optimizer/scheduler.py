from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    OneCycleLR,
    LRScheduler,
)

logger = logging.getLogger("ai-ml-template")


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.cosine_epochs = total_epochs - warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            alpha = epoch / max(self.warmup_epochs, 1)
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha for base_lr in self.base_lrs]
        else:
            cosine_epoch = epoch - self.warmup_epochs
            progress = cosine_epoch / max(self.cosine_epochs - 1, 1)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [self.eta_min + (base_lr - self.eta_min) * factor for base_lr in self.base_lrs]


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    steps_per_epoch: int | None = None,
) -> LRScheduler | None:
    scheduler_cfg = cfg.get("scheduler", cfg)
    if not scheduler_cfg:
        return None

    name = scheduler_cfg.get("name", "").lower()
    if not name:
        return None

    params = scheduler_cfg.get("params", {})

    if name == "cosine":
        max_epochs = int(cfg.get("max_epochs", scheduler_cfg.get("T_max", 100)))
        eta_min = float(params.get("eta_min", 0.0))
        logger.info("Building CosineAnnealingLR: T_max=%d eta_min=%.2e", max_epochs, eta_min)
        return CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=eta_min)

    elif name == "step":
        step_size = int(params.get("step_size", 30))
        gamma = float(params.get("gamma", 0.1))
        logger.info("Building StepLR: step_size=%d gamma=%.2f", step_size, gamma)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif name == "plateau":
        mode = params.get("mode", "min")
        factor = float(params.get("factor", 0.1))
        patience = int(params.get("patience", 10))
        logger.info(
            "Building ReduceLROnPlateau: mode=%s factor=%.2f patience=%d",
            mode,
            factor,
            patience,
        )
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)

    elif name == "onecycle":
        max_lr = float(params.get("max_lr", 1e-3))
        epochs = int(cfg.get("max_epochs", params.get("epochs", 100)))
        if steps_per_epoch is None:
            steps_per_epoch = int(params.get("steps_per_epoch", 1))
        pct_start = float(params.get("pct_start", 0.3))
        anneal_strategy = params.get("anneal_strategy", "cos")
        div_factor = float(params.get("div_factor", 25.0))
        final_div_factor = float(params.get("final_div_factor", 1e4))
        logger.info(
            "Building OneCycleLR: max_lr=%.2e epochs=%d steps_per_epoch=%d pct_start=%.2f",
            max_lr,
            epochs,
            steps_per_epoch,
            pct_start,
        )
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )

    elif name == "cosine_warmup":
        max_epochs = int(cfg.get("max_epochs", params.get("total_epochs", 100)))
        warmup_epochs = int(params.get("warmup_epochs", 5))
        warmup_start_lr = float(params.get("warmup_start_lr", 0.0))
        eta_min = float(params.get("eta_min", 0.0))
        logger.info(
            "Building WarmupCosineScheduler: warmup=%d total=%d warmup_start_lr=%.2e eta_min=%.2e",
            warmup_epochs,
            max_epochs,
            warmup_start_lr,
            eta_min,
        )
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=max_epochs,
            warmup_start_lr=warmup_start_lr,
            eta_min=eta_min,
        )

    else:
        raise ValueError(f"Unknown scheduler: {name}. Available: cosine, step, plateau, onecycle, cosine_warmup")
