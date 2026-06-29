from __future__ import annotations

import logging
from typing import Any

import torch
from torch.cuda.amp import autocast

from src.utils.distributed import is_main_process

logger = logging.getLogger("ai-ml-template")


def train_epoch(trainer: Any, epoch: int) -> dict[str, float]:
    model = trainer.model
    model.train()

    dataloader = trainer.datamodule.train_dataloader()
    experiment = trainer.experiment
    optimizer = trainer.optimizer
    scheduler = trainer.scheduler
    scaler = trainer.scaler

    use_amp = scaler is not None
    grad_accum = trainer.config.get("trainer", {}).get("grad_accumulation", 1)
    device = trainer.device

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        for callback in trainer.callbacks:
            callback.on_train_batch_start(trainer, batch, batch_idx)

        batch = _to_device(batch, device)

        if use_amp:
            with autocast():
                outputs = experiment.forward(batch)
                loss_dict = experiment.compute_loss(batch, outputs)
        else:
            outputs = experiment.forward(batch)
            loss_dict = experiment.compute_loss(batch, outputs)

        loss = loss_dict["total"] / grad_accum

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss_dict["total"].detach().item()
        num_batches += 1

        if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(dataloader):
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            trainer._global_step += 1

            if scheduler is not None and trainer._step_scheduler:
                scheduler.step()

        loss_dict_detached = {k: v.detach() for k, v in loss_dict.items()}
        for callback in trainer.callbacks:
            callback.on_train_batch_end(trainer, loss_dict_detached, batch, batch_idx)

    avg_loss = total_loss / max(num_batches, 1)
    return {"train/loss": avg_loss}


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(v, device) for v in batch)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch
