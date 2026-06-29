from __future__ import annotations

import logging
from typing import Any

import torch

from src.utils.distributed import IS_DISTRIBUTED, reduce_tensor

logger = logging.getLogger("ai-ml-template")


def validate_epoch(trainer: Any, epoch: int) -> dict[str, float]:
    model = trainer.model
    model.eval()

    dataloader = trainer.datamodule.val_dataloader()
    experiment = trainer.experiment
    device = trainer.device

    experiment.on_epoch_start("val")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            for callback in trainer.callbacks:
                callback.on_val_batch_start(trainer, batch, batch_idx)

            batch = _to_device(batch, device)
            outputs = experiment.forward(batch)
            loss_dict = experiment.compute_loss(batch, outputs)
            experiment.compute_metrics(outputs, batch, "val")

            for callback in trainer.callbacks:
                callback.on_val_batch_end(trainer, loss_dict, batch, batch_idx)

    metrics = experiment.compute_metrics(None, None, "val")
    experiment.on_epoch_end("val")

    if IS_DISTRIBUTED:
        reduced = {}
        for k, v in metrics.items():
            t = torch.tensor(float(v), device=device)
            reduced[k] = float(reduce_tensor(t, average=True).item())
        metrics = reduced

    return metrics


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(_to_device(v, device) for v in batch)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch
