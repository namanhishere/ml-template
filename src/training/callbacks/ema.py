from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.utils.registry import CALLBACKS

from .registry import Callback

logger = logging.getLogger("ai-ml-template")


@CALLBACKS.register("ema")
class EMACallback(Callback):
    def __init__(self, decay: float = 0.999, device: torch.device | None = None) -> None:
        super().__init__()
        self.decay = decay
        self.device = device
        self._ema_params: OrderedDict[str, torch.Tensor] | None = None
        self._steps: int = 0

    def on_fit_start(self, trainer: Any) -> None:
        model = trainer.model
        self._ema_params = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._ema_params[name] = param.data.clone().detach()
        self._steps = 0
        logger.info("Initialized EMA with decay=%.4f, %d parameters", self.decay, len(self._ema_params))

    def on_train_batch_end(self, trainer: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self._ema_params is None:
            return
        self._steps += 1
        decay = min(self.decay, (1.0 + self._steps) / (10.0 + self._steps))
        with torch.no_grad():
            for name, param in trainer.model.named_parameters():
                if name in self._ema_params:
                    self._ema_params[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, Any]) -> None:
        decay = min(self.decay, (1.0 + self._steps) / (10.0 + self._steps))
        logger.info("EMA decay: %.4f, steps: %d", decay, self._steps)
        metrics["ema/decay"] = decay
        metrics["ema/steps"] = self._steps

    def on_save_checkpoint(self, trainer: Any, checkpoint: dict[str, Any]) -> None:
        if self._ema_params is not None:
            checkpoint["ema_model_state"] = {k: v.clone() for k, v in self._ema_params.items()}
            checkpoint["ema_decay"] = self.decay
            checkpoint["ema_steps"] = self._steps
            logger.debug("Saved EMA state in checkpoint (%d params)", len(self._ema_params))

    def on_load_checkpoint(self, trainer: Any, checkpoint: dict[str, Any]) -> None:
        if self._ema_params is None:
            model = trainer.model
            self._ema_params = OrderedDict()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self._ema_params[name] = param.data.clone().detach()

        if "ema_model_state" in checkpoint:
            ema_state = checkpoint["ema_model_state"]
            for name, tensor in ema_state.items():
                if name in self._ema_params:
                    self._ema_params[name].copy_(tensor)
            self.decay = checkpoint.get("ema_decay", self.decay)
            self._steps = checkpoint.get("ema_steps", 0)
            logger.info("Restored EMA state from checkpoint (%d params)", len(ema_state))

    @property
    def model_ema(self) -> nn.Module:
        if self._ema_params is None:
            raise RuntimeError("EMA parameters have not been initialized. Call on_fit_start first.")
        model = copy.deepcopy(trainer.model) if hasattr(self, "_trainer_ref") else None
        if model is None:
            raise RuntimeError("No model reference available for EMA property.")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self._ema_params:
                    param.data.copy_(self._ema_params[name])
        return model

    def apply_ema_weights(self, model: nn.Module) -> None:
        if self._ema_params is None:
            raise RuntimeError("EMA parameters have not been initialized.")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self._ema_params:
                    param.data.copy_(self._ema_params[name])

    def store_original_weights(self, model: nn.Module) -> dict[str, torch.Tensor]:
        stored = {}
        for name, param in model.named_parameters():
            if name in self._ema_params:
                stored[name] = param.data.clone()
        return stored

    def restore_original_weights(self, model: nn.Module, stored: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in stored:
                    param.data.copy_(stored[name])
