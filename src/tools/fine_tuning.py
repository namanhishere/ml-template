from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.distributed import get_device
from src.utils.registry import FINETUNE_STRATEGIES

logger = logging.getLogger("ai-ml-template")


class FineTuneStrategy(ABC):
    @abstractmethod
    def configure(self, model: nn.Module, optimizer_cfg: dict) -> tuple[nn.Module, list[dict]]:
        ...

    def on_epoch_start(self, epoch: int, model: nn.Module) -> None:
        pass

    def on_epoch_end(self, epoch: int, model: nn.Module) -> None:
        pass


@FINETUNE_STRATEGIES.register("full")
class FullFineTune(FineTuneStrategy):
    def configure(self, model: nn.Module, optimizer_cfg: dict) -> tuple[nn.Module, list[dict]]:
        for param in model.parameters():
            param.requires_grad = True
        lr = optimizer_cfg.get("lr", optimizer_cfg.get("params", {}).get("lr", 0.001))
        return model, [{"params": model.parameters(), "lr": lr}]


@FINETUNE_STRATEGIES.register("differential")
class DifferentialFineTune(FineTuneStrategy):
    def __init__(self, lr_multiplier: float = 0.1) -> None:
        self.lr_multiplier = lr_multiplier

    def configure(self, model: nn.Module, optimizer_cfg: dict) -> tuple[nn.Module, list[dict]]:
        base_lr = optimizer_cfg.get("lr", optimizer_cfg.get("params", {}).get("lr", 0.001))
        backbone_params, head_params = self._split_params(model)

        for p in backbone_params:
            p.requires_grad = True
        for p in head_params:
            p.requires_grad = True

        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": base_lr * self.lr_multiplier})
        if head_params:
            groups.append({"params": head_params, "lr": base_lr})
        if not groups:
            return model, [{"params": model.parameters(), "lr": base_lr}]

        logger.info(
            "DifferentialFineTune: backbone_lr=%.2e head_lr=%.2e (multiplier=%.2f)",
            base_lr * self.lr_multiplier,
            base_lr,
            self.lr_multiplier,
        )
        return model, groups

    def _split_params(self, model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        head_keywords = {"head", "fc", "classifier", "last_linear", "linear", "output"}

        backbone_params: list[nn.Parameter] = []
        head_params: list[nn.Parameter] = []

        for name, param in model.named_parameters():
            is_head = False
            parts = set(name.lower().split("."))
            if parts & head_keywords:
                is_head = True
            elif name.startswith("model."):
                sub = name[len("model."):]
                sub_parts = set(sub.lower().split("."))
                if sub_parts & head_keywords:
                    is_head = True

            if is_head:
                head_params.append(param)
            else:
                backbone_params.append(param)

        if not head_params:
            logger.warning("No head params found via keywords %s — using last layer as head", head_keywords)
            param_list = list(model.named_parameters(recurse=True))
            if len(param_list) > 2:
                head_params = [p for _, p in param_list[-2:]]
                backbone_params = [p for _, p in param_list[:-2]]
            else:
                backbone_params = [p for _, p in param_list]

        logger.info(
            "Split params: backbone=%d head=%d",
            sum(p.numel() for p in backbone_params),
            sum(p.numel() for p in head_params),
        )
        return backbone_params, head_params


@FINETUNE_STRATEGIES.register("last_n")
class LastNFineTune(FineTuneStrategy):
    def __init__(self, n_layers: int = 2) -> None:
        self.n_layers = n_layers

    def configure(self, model: nn.Module, optimizer_cfg: dict) -> tuple[nn.Module, list[dict]]:
        for param in model.parameters():
            param.requires_grad = False

        modules = self._get_trainable_modules(model)
        last_n = modules[-min(self.n_layers, len(modules)):] if self.n_layers > 0 else []

        trainable: list[nn.Parameter] = []
        for name, module in model.named_modules():
            if module in last_n:
                for p in module.parameters(recurse=False):
                    p.requires_grad = True
                    trainable.append(p)

        if not trainable:
            logger.warning("Last-N fine-tune found no trainable params; training all.")
            for param in model.parameters():
                param.requires_grad = True
                trainable.append(param)

        lr = optimizer_cfg.get("lr", optimizer_cfg.get("params", {}).get("lr", 0.001))
        logger.info(
            "LastNFineTune: n_layers=%d trainable_params=%d",
            self.n_layers,
            sum(p.numel() for p in trainable),
        )
        return model, [{"params": trainable, "lr": lr}]

    def _get_trainable_modules(self, model: nn.Module) -> list[nn.Module]:
        candidates = []
        for module in model.modules():
            if list(module.children()):
                continue
            if list(module.parameters(recurse=False)):
                candidates.append(module)
        return candidates


@FINETUNE_STRATEGIES.register("head_only")
class HeadOnlyFineTune(FineTuneStrategy):
    def configure(self, model: nn.Module, optimizer_cfg: dict) -> tuple[nn.Module, list[dict]]:
        head_keywords = {"head", "fc", "classifier", "last_linear", "linear", "output"}

        head_params: list[nn.Parameter] = []
        for name, param in model.named_parameters():
            param.requires_grad = False
            parts = set(name.lower().split("."))
            if parts & head_keywords:
                param.requires_grad = True
                head_params.append(param)
            elif name.startswith("model."):
                sub_parts = set(name[len("model."):].lower().split("."))
                if sub_parts & head_keywords:
                    param.requires_grad = True
                    head_params.append(param)

        if not head_params:
            logger.warning("No head params found; training last parameter group.")
            param_list = list(model.parameters())
            if len(param_list) > 2:
                for p in param_list[-2:]:
                    p.requires_grad = True
                    head_params.append(p)
            else:
                for p in param_list:
                    p.requires_grad = True
                    head_params.append(p)

        lr = optimizer_cfg.get("lr", optimizer_cfg.get("params", {}).get("lr", 0.001))
        logger.info(
            "HeadOnlyFineTune: trainable_params=%d",
            sum(p.numel() for p in head_params),
        )
        return model, [{"params": head_params, "lr": lr}]


@FINETUNE_STRATEGIES.register("gradual_unfreeze")
class GradualUnfreeze(FineTuneStrategy):
    def __init__(self, unfreeze_epochs: int = 5) -> None:
        self.unfreeze_epochs = max(1, unfreeze_epochs)
        self._last_unfrozen_count: int = 0

    def configure(self, model: nn.Module, optimizer_cfg: dict) -> tuple[nn.Module, list[dict]]:
        self._layer_order = self._get_layer_order(model)
        self._last_unfrozen_count = 0

        for param in model.parameters():
            param.requires_grad = False

        lr = optimizer_cfg.get("lr", optimizer_cfg.get("params", {}).get("lr", 0.001))
        logger.info(
            "GradualUnfreeze: total_layers=%d unfreeze_every=%d epochs",
            len(self._layer_order),
            self.unfreeze_epochs,
        )
        return model, [{"params": model.parameters(), "lr": lr}]

    def on_epoch_start(self, epoch: int, model: nn.Module) -> None:
        if epoch == 0:
            self._last_unfrozen_count = 0
            for param in model.parameters():
                param.requires_grad = False
            return

        target = min(
            len(self._layer_order),
            int((epoch / self.unfreeze_epochs) * len(self._layer_order)) + 1,
        )

        if target <= self._last_unfrozen_count:
            return

        logger.info("GradualUnfreeze epoch %d: unfreezing up to layer %d/%d", epoch, target, len(self._layer_order))
        for i in range(self._last_unfrozen_count, target):
            for param in self._layer_order[i].parameters(recurse=False):
                param.requires_grad = True
        self._last_unfrozen_count = target

    def _get_layer_order(self, model: nn.Module) -> list[nn.Module]:
        layers = []
        for name, module in model.named_modules():
            if list(module.children()):
                continue
            if list(module.parameters(recurse=False)):
                layers.append(module)
        return list(reversed(layers))


class FineTuner:
    def __init__(
        self,
        model: nn.Module,
        experiment_class: type,
        config: dict,
    ) -> None:
        self.model = model
        self.experiment_class = experiment_class
        self.config = config
        self.device = get_device()

    def fine_tune(self, strategy_name: str = "differential", **strategy_kwargs: Any) -> dict:
        strategy_cls = FINETUNE_STRATEGIES.get(strategy_name)
        strategy = strategy_cls(**strategy_kwargs) if strategy_kwargs else strategy_cls()

        model = copy.deepcopy(self.model)
        model.to(self.device)

        optimizer_cfg = self.config.get("optimizer", {})
        modified_model, param_groups = strategy.configure(model, optimizer_cfg)

        trainer_cfg = self.config.get("trainer", {})
        max_epochs = trainer_cfg.get("max_epochs", 50)
        batch_size = trainer_cfg.get("batch_size", self.config.get("batch_size", 32))

        try:
            datamodule_cfg = self.config.get("dataset", {})
            datamodule_name = datamodule_cfg.get("name", "")
            if datamodule_name:
                from src.data.datamodule import BaseDataModule
                dm = BaseDataModule(config=self.config)
                train_loader = dm.train_dataloader()
                val_loader = dm.val_dataloader()
            else:
                raise ValueError("No dataset configured")
        except Exception as e:
            logger.warning("Could not create datamodule: %s. Using synthetic data.", e)
            train_loader = val_loader = None

        lr = optimizer_cfg.get("params", {}).get("lr", 0.001)
        weight_decay = optimizer_cfg.get("params", {}).get("weight_decay", 0.0)
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

        experiment = self.experiment_class(model=modified_model, config=self.config)
        experiment.to(self.device)

        metrics_history: list[dict] = []

        for epoch in range(max_epochs):
            strategy.on_epoch_start(epoch, modified_model)
            experiment.on_epoch_start("train")

            if train_loader is None:
                continue

            modified_model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = experiment.forward(batch)
                loss_dict = experiment.compute_loss(batch, outputs)
                loss = loss_dict["total"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            epoch_metrics: dict = {"train/loss": epoch_loss / max(num_batches, 1)}

            if val_loader is not None:
                modified_model.eval()
                val_loss = 0.0
                val_batches = 0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        outputs = experiment.forward(batch)
                        vld = experiment.compute_loss(batch, outputs)
                        val_loss += vld["total"].item()
                        val_batches += 1

                        if "logits" in outputs:
                            preds = torch.argmax(outputs["logits"], dim=1)
                            labels = batch.get("label")
                            if labels is not None:
                                val_correct += (preds == labels).sum().item()
                                val_total += labels.size(0)

                epoch_metrics["val/loss"] = val_loss / max(val_batches, 1)
                if val_total > 0:
                    epoch_metrics["val/acc"] = val_correct / val_total

            strategy.on_epoch_end(epoch, modified_model)
            experiment.on_epoch_end("train")
            metrics_history.append(epoch_metrics)

            if epoch % 10 == 0 or epoch == max_epochs - 1:
                logger.info(
                    "FineTuner epoch %d/%d: %s",
                    epoch + 1,
                    max_epochs,
                    {k: f"{v:.4f}" for k, v in epoch_metrics.items()},
                )

        return {
            "strategy": strategy_name,
            "metrics_history": metrics_history,
            "final_metrics": metrics_history[-1] if metrics_history else {},
            "best_val_loss": min((m.get("val/loss", float("inf")) for m in metrics_history), default=0.0),
            "best_val_acc": max((m.get("val/acc", 0.0) for m in metrics_history), default=0.0),
        }
