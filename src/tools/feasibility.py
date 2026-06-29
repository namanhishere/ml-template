from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from src.utils.distributed import get_device

logger = logging.getLogger("ai-ml-template")


class FeasibilityAnalyzer:
    """
    Validates that the model + pipeline can actually learn before committing to full training.
    """

    def __init__(
        self,
        model: nn.Module,
        experiment_class: type,
        config: dict,
        datamodule: Any = None,
    ) -> None:
        self.model = model
        self.experiment_class = experiment_class
        self.config = config
        self.datamodule = datamodule
        self.device = get_device()

    def run(self, tests: list[str] | None = None) -> dict:
        if tests is None:
            tests = ["overfit", "random_label"]
        results = {}
        for test_name in tests:
            logger.info("Running feasibility test: %s", test_name)
            if test_name == "overfit":
                results["overfit"] = self._overfit_test()
            elif test_name == "random_label":
                results["random_label"] = self._random_label_test()
            elif test_name == "gradient_sanity":
                results["gradient_sanity"] = self._gradient_sanity_check()
            elif test_name == "activation_stats":
                results["activation_stats"] = self._activation_stats()
            else:
                logger.warning("Unknown feasibility test: %s", test_name)
        return results

    def _overfit_test(
        self,
        num_samples: int = 50,
        max_epochs: int = 100,
        target_acc: float = 0.99,
    ) -> dict:
        model = deepcopy_model(self.model)
        experiment, dataloader = self._make_tiny_experiment(model, num_samples)

        lr = self.config.get("optimizer", {}).get("params", {}).get("lr", 0.001)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        train_acc_history = []
        final_loss = float("inf")
        epochs_to_reach: int | None = None

        for epoch in range(max_epochs):
            model.train()
            experiment.on_epoch_start("train")
            correct = 0
            total = 0
            epoch_loss = 0.0

            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = experiment.forward(batch)
                loss_dict = experiment.compute_loss(batch, outputs)
                loss = loss_dict["total"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                preds = torch.argmax(outputs["logits"], dim=1)
                labels = batch.get("label")
                if labels is not None:
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            train_acc = correct / max(total, 1)
            train_acc_history.append(train_acc)
            final_loss = epoch_loss / max(len(dataloader), 1)

            if train_acc >= target_acc and epochs_to_reach is None:
                epochs_to_reach = epoch + 1

            if train_acc >= target_acc and epoch >= 5:
                break

        success = train_acc >= target_acc
        logger.info(
            "Overfit test: success=%s train_acc=%.4f epochs=%s final_loss=%.4f",
            success,
            train_acc,
            epochs_to_reach or max_epochs,
            final_loss,
        )

        return {
            "success": success,
            "train_acc": train_acc,
            "epochs_to_reach": epochs_to_reach if epochs_to_reach is not None else max_epochs,
            "final_loss": final_loss,
            "acc_history": train_acc_history,
        }

    def _random_label_test(
        self,
        num_samples: int = 100,
        max_epochs: int = 200,
    ) -> dict:
        model = deepcopy_model(self.model)
        experiment, original_dataloader = self._make_tiny_experiment(model, num_samples)

        all_batches = list(original_dataloader)
        num_classes = self.config.get("model", {}).get(
            "num_classes", self.config.get("dataset", {}).get("num_classes", 10)
        )

        shuffled_data: list[dict] = []
        for batch in all_batches:
            labels = batch["label"].clone()
            rand_labels = torch.randint(0, num_classes, labels.shape, dtype=labels.dtype)
            batch = dict(batch)
            batch["label"] = rand_labels
            shuffled_data.append(batch)

        class _ListLoader:
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        rl_dataloader = _ListLoader(shuffled_data)

        lr = self.config.get("optimizer", {}).get("params", {}).get("lr", 0.001)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        final_acc = 0.0
        for epoch in range(max_epochs):
            model.train()
            experiment.on_epoch_start("train")
            correct = 0
            total = 0

            for batch in rl_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = experiment.forward(batch)
                loss_dict = experiment.compute_loss(batch, outputs)
                loss = loss_dict["total"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs["logits"], dim=1)
                labels = batch.get("label")
                if labels is not None:
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            final_acc = correct / max(total, 1)
            if final_acc > 0.95:
                break

        can_memorize = final_acc > 0.90
        baseline = 1.0 / num_classes
        gap = final_acc - baseline

        logger.info(
            "Random label test: can_memorize=%s final_acc=%.4f gap_to_baseline=%.4f",
            can_memorize,
            final_acc,
            gap,
        )

        return {
            "can_memorize": can_memorize,
            "final_acc": final_acc,
            "gap_to_baseline": gap,
        }

    def _gradient_sanity_check(self) -> dict:
        model = self.model
        model.train()
        model.to(self.device)

        batch_size = self.config.get("batch_size", self.config.get("trainer", {}).get("batch_size", 4))
        num_classes = self.config.get("model", {}).get(
            "num_classes", self.config.get("dataset", {}).get("num_classes", 10)
        )

        try:
            dummy = torch.randn(batch_size, 3, 224, 224, device=self.device)
            labels = torch.randint(0, num_classes, (batch_size,), device=self.device)
        except Exception:
            dummy = torch.randn(batch_size, 3, 32, 32, device=self.device)
            labels = torch.randint(0, num_classes, (batch_size,), device=self.device)

        output = model(dummy)
        if isinstance(output, dict):
            logits = output.get("logits", output)
        else:
            logits = output

        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()

        max_grad_norm = 0.0
        has_nan = False
        has_zero = False
        layer_stats: dict[str, dict] = {}

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.data
            norm = grad.norm().item()
            max_grad_norm = max(max_grad_norm, norm)

            if torch.isnan(grad).any():
                has_nan = True
            if (grad == 0).all():
                has_zero = True

            key = name.split(".")[0]
            if key not in layer_stats:
                layer_stats[key] = {"min_norm": float("inf"), "max_norm": 0.0, "count": 0}
            layer_stats[key]["min_norm"] = min(layer_stats[key]["min_norm"], norm)
            layer_stats[key]["max_norm"] = max(layer_stats[key]["max_norm"], norm)
            layer_stats[key]["count"] += 1

        for key in layer_stats:
            if layer_stats[key]["min_norm"] == float("inf"):
                layer_stats[key]["min_norm"] = 0.0

        gradient_flow_ok = not has_nan and max_grad_norm > 0

        logger.info(
            "Gradient sanity: flow_ok=%s max_norm=%.4e has_nan=%s",
            gradient_flow_ok,
            max_grad_norm,
            has_nan,
        )

        return {
            "gradient_flow_ok": gradient_flow_ok,
            "max_grad_norm": max_grad_norm,
            "has_nan": has_nan,
            "has_zero": has_zero,
            "layer_stats": layer_stats,
        }

    def _activation_stats(self) -> dict:
        model = self.model
        model.eval()
        model.to(self.device)

        batch_size = self.config.get("batch_size", self.config.get("trainer", {}).get("batch_size", 4))

        try:
            dummy = torch.randn(batch_size, 3, 224, 224, device=self.device)
        except Exception:
            dummy = torch.randn(batch_size, 3, 32, 32, device=self.device)

        activations: list[torch.Tensor] = []
        hooks: list[Any] = []

        def _hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                activations.append(out.detach())

        for module in model.modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.ELU)):
                hooks.append(module.register_forward_hook(_hook))

        with torch.no_grad():
            try:
                model(dummy)
            except Exception:
                pass

        for h in hooks:
            h.remove()

        if not activations:
            logger.warning("No activation tensors captured for activation stats check")
            return {
                "dead_neurons_pct": 0.0,
                "activation_mean": 0.0,
                "activation_std": 0.0,
            }

        all_acts = torch.cat([a.flatten() for a in activations])
        total = all_acts.numel()
        dead_neurons = (all_acts == 0).sum().item()
        dead_neurons_pct = (dead_neurons / total * 100) if total > 0 else 0.0
        act_mean = all_acts.mean().item()
        act_std = all_acts.std().item()

        logger.info(
            "Activation stats: dead=%.1f%% mean=%.4f std=%.4f",
            dead_neurons_pct,
            act_mean,
            act_std,
        )

        return {
            "dead_neurons_pct": dead_neurons_pct,
            "activation_mean": act_mean,
            "activation_std": act_std,
        }

    def _make_tiny_experiment(self, model: nn.Module, num_samples: int) -> tuple[Any, DataLoader]:
        if self.datamodule is not None:
            try:
                full_dataset = self.datamodule.train_dataset
                indices = list(range(min(num_samples, len(full_dataset))))
                subset = Subset(full_dataset, indices)
            except Exception:
                subset = self._make_synthetic_dataset(num_samples)
        else:
            subset = self._make_synthetic_dataset(num_samples)

        batch_size = min(num_samples, 16)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        model.to(self.device)

        experiment = self.experiment_class(model=model, config=self.config)
        experiment.to(self.device)
        return experiment, dataloader

    def _make_synthetic_dataset(self, num_samples: int) -> Dataset:
        num_classes = self.config.get("model", {}).get(
            "num_classes", self.config.get("dataset", {}).get("num_classes", 10)
        )
        images = torch.randn(num_samples, 3, 224, 224)
        labels = torch.randint(0, num_classes, (num_samples,))
        return TensorDataset(images, labels)


def deepcopy_model(model: nn.Module) -> nn.Module:
    import copy

    try:
        return copy.deepcopy(model)
    except Exception:
        try:
            state = model.state_dict()
            new_model = copy.deepcopy(model)
            new_model.load_state_dict(state)
            return new_model
        except Exception:
            logger.warning("Could not deepcopy model; reusing the original (state mutated)")
            return model
