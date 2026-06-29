import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

from omegaconf import OmegaConf

from src.training.checkpoint import build_checkpoint, load_checkpoint, resume_from_checkpoint


def _mock_trainer():
    trainer = MagicMock()
    model = nn.Linear(10, 5)
    trainer.model = model
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, step_size=10)
    trainer.scaler = None
    trainer._current_epoch = 5
    trainer._global_step = 500
    trainer._best_metric = {"val/loss": 0.1234}
    trainer._metrics_history = [
        {"train/loss": 1.0, "val/loss": 0.8},
        {"train/loss": 0.7, "val/loss": 0.5},
    ]
    trainer.config = OmegaConf.create({"seed": 42})
    trainer.device = torch.device("cpu")
    trainer.callbacks = []
    trainer.datamodule = None
    type(trainer)._dataset_version = PropertyMock(return_value=None)
    return trainer


class TestBuildCheckpoint:
    def test_build_checkpoint_returns_dict(self, seed):
        trainer = _mock_trainer()
        ckpt = build_checkpoint(trainer)
        assert isinstance(ckpt, dict)

    def test_checkpoint_has_required_keys(self, seed):
        trainer = _mock_trainer()
        ckpt = build_checkpoint(trainer)

        required_keys = [
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "epoch",
            "global_step",
            "timestamp",
            "pytorch_version",
        ]
        for key in required_keys:
            assert key in ckpt, f"Missing key: {key}"

    def test_epoch_value_match(self, seed):
        trainer = _mock_trainer()
        trainer._current_epoch = 7
        ckpt = build_checkpoint(trainer)
        assert ckpt["epoch"] == 7

    def test_global_step_match(self, seed):
        trainer = _mock_trainer()
        trainer._global_step = 1234
        ckpt = build_checkpoint(trainer)
        assert ckpt["global_step"] == 1234

    def test_model_state_is_state_dict(self, seed):
        trainer = _mock_trainer()
        ckpt = build_checkpoint(trainer)
        assert isinstance(ckpt["model_state"], dict)
        for key in ckpt["model_state"]:
            assert isinstance(ckpt["model_state"][key], torch.Tensor)

    def test_optimizer_state_preserved(self, seed):
        trainer = _mock_trainer()
        ckpt = build_checkpoint(trainer)
        assert "param_groups" in ckpt["optimizer_state"]

    def test_scheduler_state_preserved(self, seed):
        trainer = _mock_trainer()
        ckpt = build_checkpoint(trainer)
        assert isinstance(ckpt["scheduler_state"], dict)

    def test_best_metric_preserved(self, seed):
        trainer = _mock_trainer()
        trainer._best_metric = {"val/accuracy": 0.95}
        ckpt = build_checkpoint(trainer)
        assert ckpt["best_metric"] == {"val/accuracy": 0.95}

    def test_metrics_history_preserved(self, seed):
        trainer = _mock_trainer()
        trainer._metrics_history = [{"val/loss": 0.5}]
        ckpt = build_checkpoint(trainer)
        assert ckpt["metrics_history"] == [{"val/loss": 0.5}]

    def test_with_metrics_arg(self, seed):
        trainer = _mock_trainer()
        ckpt = build_checkpoint(trainer, metrics={"val/loss": 0.42})
        assert ckpt["last_metrics"] == {"val/loss": 0.42}

    def test_timestamp_format(self, seed):
        trainer = _mock_trainer()
        ckpt = build_checkpoint(trainer)
        assert "T" in ckpt["timestamp"]

    def test_metadata_fields(self, seed):
        trainer = _mock_trainer()
        ckpt = build_checkpoint(trainer)
        assert "pytorch_version" in ckpt
        assert "cuda_version" in ckpt
        assert "framework_version" in ckpt
        assert "device" in ckpt
        assert "seed" in ckpt
        assert "world_size" in ckpt
        assert "local_rank" in ckpt


class TestLoadCheckpoint:
    def test_load_roundtrip(self, seed, tmp_path):
        trainer = _mock_trainer()
        ckpt = build_checkpoint(trainer)

        save_path = tmp_path / "test_ckpt.pt"
        torch.save(ckpt, save_path)

        loaded = load_checkpoint(save_path)
        assert isinstance(loaded, dict)
        assert loaded["epoch"] == ckpt["epoch"]
        assert loaded["global_step"] == ckpt["global_step"]
        for key in ckpt["model_state"]:
            assert torch.equal(ckpt["model_state"][key], loaded["model_state"][key])

    def test_load_with_device_map(self, seed, tmp_path):
        trainer = _mock_trainer()
        ckpt = build_checkpoint(trainer)

        save_path = tmp_path / "ckpt.pt"
        torch.save(ckpt, save_path)

        if torch.cuda.is_available():
            loaded = load_checkpoint(save_path, device="cuda:0")
            for key in loaded["model_state"]:
                assert loaded["model_state"][key].device.type == "cuda"


class TestResumeFromCheckpoint:
    def test_resume_restores_epoch_and_step(self, seed, tmp_path):
        trainer = _mock_trainer()
        ckpt = build_checkpoint(trainer)
        ckpt["epoch"] = 10
        ckpt["global_step"] = 1000

        save_path = tmp_path / "resume.pt"
        torch.save(ckpt, save_path)

        new_trainer = _mock_trainer()
        resume_from_checkpoint(new_trainer, save_path)
        assert new_trainer._current_epoch == 10
        assert new_trainer._global_step == 1000

    def test_resume_restores_model_weights(self, seed, tmp_path):
        trainer = _mock_trainer()
        ckpt = build_checkpoint(trainer)
        saved_bias = ckpt["model_state"]["bias"].clone()

        save_path = tmp_path / "resume_model.pt"
        torch.save(ckpt, save_path)

        new_trainer = _mock_trainer()
        nn.init.constant_(new_trainer.model.bias, 99.0)
        assert not torch.equal(new_trainer.model.bias, saved_bias)

        resume_from_checkpoint(new_trainer, save_path)
        assert torch.equal(new_trainer.model.bias, saved_bias)

    def test_resume_restores_best_metric(self, seed, tmp_path):
        trainer = _mock_trainer()
        trainer._best_metric = {"val/loss": 0.05}
        ckpt = build_checkpoint(trainer)

        save_path = tmp_path / "resume_metric.pt"
        torch.save(ckpt, save_path)

        new_trainer = _mock_trainer()
        resume_from_checkpoint(new_trainer, save_path)
        assert new_trainer._best_metric == {"val/loss": 0.05}

    def test_resume_restores_metrics_history(self, seed, tmp_path):
        trainer = _mock_trainer()
        trainer._metrics_history = [{"val/loss": 1.0}, {"val/loss": 0.5}]
        ckpt = build_checkpoint(trainer)

        save_path = tmp_path / "resume_history.pt"
        torch.save(ckpt, save_path)

        new_trainer = _mock_trainer()
        resume_from_checkpoint(new_trainer, save_path)
        assert len(new_trainer._metrics_history) == 2
