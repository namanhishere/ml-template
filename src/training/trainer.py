from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.experiments.registry import EXPERIMENTS
from src.training.checkpoint import build_checkpoint, load_checkpoint, resume_from_checkpoint
from src.training.engine.train import train_epoch
from src.training.engine.validate import validate_epoch
from src.training.engine.test import test_epoch
from src.training.optimizer.factory import build_optimizer
from src.training.optimizer.scheduler import build_scheduler
from src.utils.distributed import get_device, is_main_process, log_main
from src.utils.registry import CALLBACKS

logger = logging.getLogger("ai-ml-template")


class Trainer:
    def __init__(
        self,
        experiment: Any,
        config: DictConfig,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.experiment = experiment
        self.config = config
        self.callbacks = callbacks or []

        self._current_epoch: int = 0
        self._global_step: int = 0
        self._should_stop: bool = False
        self._best_metric: dict[str, float] = {}
        self._metrics_history: list[dict[str, float]] = []

        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: Any = None
        self._scaler: torch.cuda.amp.GradScaler | None = None
        self._datamodule: Any = None
        self._dataset_version: Any = None

        self._step_scheduler: bool = False

        self._device: torch.device = get_device()

    @property
    def model(self) -> nn.Module:
        return self.experiment.model

    @property
    def optimizer(self) -> torch.optim.Optimizer | None:
        return self._optimizer

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    @property
    def scaler(self) -> torch.cuda.amp.GradScaler | None:
        return self._scaler

    @property
    def datamodule(self) -> Any:
        if self._datamodule is None:
            raise RuntimeError("datamodule has not been built. Call fit() first.")
        return self._datamodule

    @property
    def device(self) -> torch.device:
        return self._device

    def fit(self, ckpt_path: str | None = None) -> None:
        for callback in self.callbacks:
            callback.on_fit_start(self)

        self._build_datamodule()
        self._build_optimizer()
        self._build_scheduler()
        self._build_scaler()

        if ckpt_path is not None:
            resume_from_checkpoint(self, ckpt_path)

        max_epochs = self.config.get("trainer", {}).get("max_epochs", 100)
        self.model.to(self._device)

        for epoch in range(self._current_epoch, max_epochs):
            self._current_epoch = epoch

            for callback in self.callbacks:
                callback.on_epoch_start(self, epoch)

            train_metrics = train_epoch(self, epoch)
            val_metrics = validate_epoch(self, epoch)

            metrics = {**train_metrics, **val_metrics}
            self._metrics_history.append(metrics)

            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, metrics)

            if self._should_stop:
                log_main("Training stopped early at epoch %d", epoch)
                break

            if self._scheduler is not None and not self._step_scheduler:
                if hasattr(self._scheduler, "step") and not isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self._scheduler.step()
                elif isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    monitor_metric = metrics.get("val/loss", 0.0)
                    self._scheduler.step(monitor_metric)

        for callback in self.callbacks:
            callback.on_fit_end(self)

    def validate(self) -> dict[str, float]:
        self.model.to(self._device)
        self.model.eval()
        return validate_epoch(self, self._current_epoch)

    def test(self) -> dict[str, float]:
        self.model.to(self._device)
        self.model.eval()
        return test_epoch(self)

    def _build_optimizer(self) -> None:
        self._optimizer = build_optimizer(self.model, self.config)

    def _build_scheduler(self) -> None:
        scheduler_cfg = self.config.get("trainer", {}).get("scheduler", {}) or self.config.get("scheduler", {})
        if scheduler_cfg:
            name = scheduler_cfg.get("name", "").lower()
            if name in ("onecycle",):
                self._step_scheduler = True

            steps_per_epoch = None
            try:
                steps_per_epoch = self._datamodule.train_dataloader_len()
            except Exception:
                pass

            self._scheduler = build_scheduler(
                self._optimizer,
                self.config,
                steps_per_epoch=steps_per_epoch,
            )

    def _build_scaler(self) -> None:
        precision = self.config.get("trainer", {}).get("precision", "fp32")
        use_amp = precision in ("fp16", "bf16")
        if use_amp and torch.cuda.is_available():
            self._scaler = torch.cuda.amp.GradScaler()
            logger.info(f"AMP enabled with GradScaler (precision={precision})")

    def _build_datamodule(self) -> None:
        dataset_cfg = self.config.get("dataset", {})
        dataset_name = dataset_cfg.get("name", "")
        if dataset_name:
            from src.data.datamodule import BaseDataModule
            self._datamodule = BaseDataModule(config=self.config)
        else:
            raise ValueError("No dataset name specified in config.dataset.name")

    def _build_checkpoint(self) -> dict[str, Any]:
        checkpoint = build_checkpoint(self, self._metrics_history[-1] if self._metrics_history else None)

        for callback in self.callbacks:
            if hasattr(callback, "on_save_checkpoint"):
                callback.on_save_checkpoint(self, checkpoint)

        return checkpoint

    def save_checkpoint(self, path: str) -> None:
        checkpoint = self._build_checkpoint()
        torch.save(checkpoint, path)
        log_main("Checkpoint saved to %s", path)

    @classmethod
    def build_from_config(cls, config: DictConfig) -> Trainer:
        from src.utils.registry import MODELS
        from src.models.zoo import BackboneFactory

        experiment_name = config.get("experiment", {}).get("name", "classification")

        model_cfg = config.get("model", {})
        model_name = model_cfg.get("name", "resnet50")
        backbone_source = model_cfg.get("backbone", f"torchvision://{model_name}")
        pretrained = model_cfg.get("pretrained", True)
        num_classes = model_cfg.get("num_classes", config.get("dataset", {}).get("num_classes", 10))

        if MODELS.__contains__(model_name):
            model = MODELS.instantiate(model_name, num_classes=num_classes, pretrained=pretrained)
        else:
            backbone = BackboneFactory.create(backbone_source, pretrained=pretrained, num_classes=num_classes)
            feature_dim = BackboneFactory.get_feature_dim(backbone_source)
            from src.models.heads.classification import ClassificationHead
            head = ClassificationHead(in_features=feature_dim, num_classes=num_classes)
            model = nn.Sequential(backbone, nn.Flatten(1), head)

        experiment = EXPERIMENTS.instantiate(
            experiment_name,
            model=model,
            config=config,
        )

        callbacks_list: list[Any] = []
        callback_configs = config.get("callbacks", [])
        for cb_cfg in callback_configs:
            if isinstance(cb_cfg, (dict, DictConfig)):
                cb_name = cb_cfg.get("name", "")
                cb_params = {k: v for k, v in cb_cfg.items() if k != "name"}
                if cb_name:
                    cb = CALLBACKS.instantiate(cb_name, **cb_params)
                    callbacks_list.append(cb)

        return cls(experiment, config, callbacks_list)
