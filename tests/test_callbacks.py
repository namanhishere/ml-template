import pytest
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from src.training.callbacks.early_stop import EarlyStopping
from src.training.callbacks.checkpoint import CheckpointCallback
from src.training.callbacks.ema import EMACallback


def _mock_trainer():
    trainer = MagicMock()
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, 2, 1),
        nn.ReLU(),
    )
    for p in model.parameters():
        p.requires_grad = True
    trainer.model = model
    trainer._should_stop = False
    trainer._best_metric = {}
    trainer._current_epoch = 0
    trainer._global_step = 0
    trainer._metrics_history = []
    trainer.config = MagicMock()
    trainer.device = torch.device("cpu")
    trainer.optimizer = None
    trainer.scheduler = None
    trainer.scaler = None
    trainer.datamodule = None
    trainer.callbacks = []

    def build_checkpoint():
        return {"model_state": model.state_dict(), "epoch": trainer._current_epoch, "config": {}}

    trainer._build_checkpoint = build_checkpoint
    return trainer


class TestEarlyStopping:
    def test_instantiate(self):
        cb = EarlyStopping(monitor="val/loss", patience=5)
        assert cb.monitor == "val/loss"
        assert cb.patience == 5
        assert cb._counter == 0

    def test_on_fit_start_resets_state(self):
        cb = EarlyStopping(patience=5)
        trainer = _mock_trainer()
        cb._counter = 3
        cb._best_score = 0.5
        cb.on_fit_start(trainer)
        assert cb._counter == 0
        assert cb._best_score is None

    def test_no_trigger_before_patience(self):
        cb = EarlyStopping(patience=10, mode="min")
        trainer = _mock_trainer()
        cb.on_fit_start(trainer)
        for epoch in range(5):
            metrics = {"val/loss": 1.0 - epoch * 0.01}
            cb.on_epoch_end(trainer, epoch, metrics)
            trainer._current_epoch = epoch
        assert trainer._should_stop is False

    def test_triggers_after_patience(self):
        cb = EarlyStopping(patience=3, mode="min")
        trainer = _mock_trainer()
        cb.on_fit_start(trainer)
        for epoch in range(10):
            increasing_loss = 0.5 + epoch * 0.1
            metrics = {"val/loss": increasing_loss}
            cb.on_epoch_end(trainer, epoch, metrics)
            trainer._current_epoch = epoch
            if trainer._should_stop:
                break
        assert trainer._should_stop is True

    def test_does_not_trigger_if_improving(self):
        cb = EarlyStopping(patience=3, mode="min")
        trainer = _mock_trainer()
        cb.on_fit_start(trainer)
        for epoch in range(10):
            decreasing_loss = 5.0 - epoch * 0.5
            metrics = {"val/loss": decreasing_loss}
            cb.on_epoch_end(trainer, epoch, metrics)
        assert trainer._should_stop is False

    def test_mode_max(self):
        cb = EarlyStopping(monitor="val/accuracy", patience=3, mode="max")
        trainer = _mock_trainer()
        cb.on_fit_start(trainer)
        for epoch in range(10):
            metrics = {"val/accuracy": 0.5 - epoch * 0.05}
            cb.on_epoch_end(trainer, epoch, metrics)
            if trainer._should_stop:
                break
        assert trainer._should_stop is True

    def test_min_delta_respects(self):
        cb = EarlyStopping(patience=2, min_delta=0.1, mode="min")
        trainer = _mock_trainer()
        cb.on_fit_start(trainer)
        values = [1.0, 0.85, 0.70, 0.72, 0.74]
        for epoch, val in enumerate(values):
            metrics = {"val/loss": val}
            cb.on_epoch_end(trainer, epoch, metrics)
            if trainer._should_stop:
                break
        assert cb._counter == 2
        assert cb._best_score == 0.70

    def test_missing_metric_no_trigger(self):
        cb = EarlyStopping(patience=2)
        trainer = _mock_trainer()
        cb.on_fit_start(trainer)
        for epoch in range(10):
            cb.on_epoch_end(trainer, epoch, {"other/loss": 0.5})
        assert trainer._should_stop is False


class TestCheckpointCallback:
    def test_instantiate(self, tmp_path):
        cb = CheckpointCallback(save_dir=str(tmp_path / "ckpts"), save_top_k=2)
        assert cb.save_top_k == 2
        assert cb.monitor == "val/loss"

    def test_on_fit_start_creates_dir(self, tmp_path):
        save_dir = tmp_path / "test_ckpts"
        cb = CheckpointCallback(save_dir=str(save_dir))
        trainer = _mock_trainer()
        cb.on_fit_start(trainer)
        assert save_dir.exists()
        assert save_dir.is_dir()

    def test_saves_last_checkpoint(self, tmp_path):
        save_dir = tmp_path / "ckpts"
        cb = CheckpointCallback(save_dir=str(save_dir), save_last=True, save_top_k=0)
        trainer = _mock_trainer()
        cb.on_fit_start(trainer)

        cb.on_epoch_end(trainer, 0, {"val/loss": 1.0})
        last_path = save_dir / "last.pt"
        assert last_path.exists()
        ckpt = torch.load(last_path)
        assert "model_state" in ckpt
        assert "epoch" in ckpt

    def test_saves_best_checkpoint(self, tmp_path):
        save_dir = tmp_path / "ckpts_best"
        cb = CheckpointCallback(save_dir=str(save_dir), save_top_k=2, save_last=False)
        trainer = _mock_trainer()
        cb.on_fit_start(trainer)

        trainer._current_epoch = 0
        cb.on_epoch_end(trainer, 0, {"val/loss": 2.0})
        trainer._current_epoch = 1
        cb.on_epoch_end(trainer, 1, {"val/loss": 1.0})
        trainer._current_epoch = 2
        cb.on_epoch_end(trainer, 2, {"val/loss": 3.0})

        best_path = save_dir / "best.pt"
        assert best_path.exists()
        ckpt = torch.load(best_path)
        assert ckpt["epoch"] == 1

    def test_save_top_k_prunes_old(self, tmp_path):
        save_dir = tmp_path / "ckpts_prune"
        cb = CheckpointCallback(save_dir=str(save_dir), save_top_k=2, save_last=False)
        trainer = _mock_trainer()
        cb.on_fit_start(trainer)
        for epoch in range(6):
            trainer._current_epoch = epoch
            cb.on_epoch_end(trainer, epoch, {"val/loss": 10.0 - epoch})

        epoch_files = sorted(save_dir.glob("epoch_*.pt"))
        best_exists = (save_dir / "best.pt").exists()
        assert best_exists

    def test_save_every_n_epochs(self, tmp_path):
        save_dir = tmp_path / "ckpts_every_n"
        cb = CheckpointCallback(save_dir=str(save_dir), save_every_n_epochs=3, save_last=False, save_top_k=0)
        trainer = _mock_trainer()
        cb.on_fit_start(trainer)
        for epoch in range(9):
            trainer._current_epoch = epoch
            cb.on_epoch_end(trainer, epoch, {"val/loss": 1.0})

        epoch_files = sorted(save_dir.glob("epoch_*.pt"))
        assert len(epoch_files) == 3

    def test_extract_metric_handles_tensor(self, tmp_path):
        cb = CheckpointCallback(save_dir=str(tmp_path))
        val = cb._extract_metric({"val/loss": torch.tensor(0.5)})
        assert val == 0.5

    def test_extract_metric_handles_missing(self, tmp_path):
        cb = CheckpointCallback(save_dir=str(tmp_path))
        val = cb._extract_metric({"other": 0.5})
        assert val is None

    def test_is_better_mode_min(self, tmp_path):
        cb = CheckpointCallback(save_dir=str(tmp_path), mode="min")
        cb._best_score = 0.5
        assert cb._is_better(0.3) is True
        assert cb._is_better(0.7) is False

    def test_is_better_mode_max(self, tmp_path):
        cb = CheckpointCallback(save_dir=str(tmp_path), mode="max")
        cb._best_score = 0.7
        assert cb._is_better(0.9) is True
        assert cb._is_better(0.5) is False


class TestEMACallback:
    def test_instantiate(self):
        cb = EMACallback(decay=0.999)
        assert cb.decay == 0.999
        assert cb._steps == 0

    def test_on_fit_start_initializes_ema(self, seed):
        cb = EMACallback(decay=0.999)
        trainer = _mock_trainer()
        cb.on_fit_start(trainer)
        assert cb._ema_params is not None
        assert len(cb._ema_params) > 0
        assert cb._steps == 0

    def test_on_train_batch_end_updates_ema(self, seed):
        cb = EMACallback(decay=0.99)
        trainer = _mock_trainer()

        trainer.model = nn.Linear(10, 5)
        for p in trainer.model.parameters():
            p.requires_grad = True
        with torch.no_grad():
            trainer.model.weight.fill_(1.0)
            trainer.model.bias.fill_(0.0)

        cb.on_fit_start(trainer)
        for p in trainer.model.parameters():
            p.data.fill_(2.0)

        for _ in range(10):
            cb.on_train_batch_end(trainer, None, None, 0)

        for name, ema_val in cb._ema_params.items():
            original = dict(trainer.model.named_parameters())[name]
            assert not torch.allclose(ema_val, original)

    def test_on_epoch_end_adds_metrics(self, seed):
        cb = EMACallback(decay=0.999)
        trainer = _mock_trainer()
        cb.on_fit_start(trainer)
        for _ in range(5):
            cb.on_train_batch_end(trainer, None, None, 0)
        metrics = {}
        cb.on_epoch_end(trainer, 0, metrics)
        assert "ema/decay" in metrics
        assert "ema/steps" in metrics
        assert metrics["ema/steps"] == 5

    def test_save_load_ema_state(self, seed, tmp_path):
        cb = EMACallback(decay=0.99)
        trainer = _mock_trainer()
        trainer.model = nn.Linear(10, 5)
        for p in trainer.model.parameters():
            p.requires_grad = True
            nn.init.constant_(p, 1.0)

        cb.on_fit_start(trainer)
        for _ in range(10):
            cb.on_train_batch_end(trainer, None, None, 0)

        checkpoint = {}
        cb.on_save_checkpoint(trainer, checkpoint)
        assert "ema_model_state" in checkpoint
        assert "ema_decay" in checkpoint

        cb2 = EMACallback(decay=0.99)
        cb2.on_fit_start(trainer)
        cb2.on_load_checkpoint(trainer, checkpoint)
        assert cb2._steps > 0

    def test_apply_ema_weights(self, seed):
        cb = EMACallback(decay=0.99)
        trainer = _mock_trainer()
        trainer.model = nn.Linear(10, 5)
        for p in trainer.model.parameters():
            p.requires_grad = True
            nn.init.constant_(p, 1.0)

        cb.on_fit_start(trainer)
        original = {name: param.data.clone() for name, param in trainer.model.named_parameters() if param.requires_grad}

        for p in trainer.model.parameters():
            p.data.fill_(5.0)

        for _ in range(20):
            cb.on_train_batch_end(trainer, None, None, 0)

        cb.apply_ema_weights(trainer.model)
        ema_vals = {name: param.data.clone() for name, param in trainer.model.named_parameters() if param.requires_grad}

        for name in original:
            assert not torch.allclose(ema_vals[name], original[name])

    def test_store_restore_original_weights(self, seed):
        cb = EMACallback(decay=0.99)
        trainer = _mock_trainer()
        trainer.model = nn.Linear(10, 5)
        for p in trainer.model.parameters():
            p.requires_grad = True
            nn.init.constant_(p, 1.0)

        cb.on_fit_start(trainer)
        stored = cb.store_original_weights(trainer.model)

        for p in trainer.model.parameters():
            nn.init.constant_(p, 42.0)

        cb.restore_original_weights(trainer.model, stored)
        for p in trainer.model.parameters():
            assert torch.allclose(p.data, torch.tensor(1.0))
