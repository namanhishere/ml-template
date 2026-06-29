import torch
from omegaconf import OmegaConf

from src.data.datamodules.image_datamodule import ImageDataModule
from src.data.registry import DATASETS
from src.experiments.classification import ClassificationExperiment
from src.models.conv_demo import ConvDemo
from src.models.registry import MODELS


class TestDemoPipeline:
    """End-to-end test of the ConvDemo pipeline — model, experiment, datamodule, loss, grad flow."""

    @staticmethod
    def _make_demo_config() -> OmegaConf:
        return OmegaConf.create(
            {
                "seed": 42,
                "dataset": {
                    "name": "synthetic",
                    "num_samples_per_class": 20,
                    "num_classes": 4,
                    "image_size": 64,
                    "seed": 42,
                },
                "augmentation": "none",
                "experiment": {"name": "classification", "num_classes": 4, "loss": {"name": "cross_entropy"}},
                "model": {"name": "conv_demo", "num_classes": 4, "channels": [8, 16, 32], "dropout": 0.0},
                "loss": {"name": "cross_entropy"},
                "optimizer": {"name": "adam", "params": {"lr": 0.01, "weight_decay": 0.0}},
                "scheduler": {"name": "cosine", "T_max": 2},
                "trainer": {
                    "max_epochs": 2,
                    "batch_size": 8,
                    "num_workers": 0,
                    "precision": "fp32",
                    "grad_accumulation": 1,
                },
                "callbacks": [
                    {
                        "name": "checkpoint",
                        "save_dir": "./checkpoints_demo",
                        "save_top_k": 1,
                        "save_last": False,
                        "save_every_n_epochs": 1,
                        "monitor": "val/loss",
                        "mode": "min",
                    },
                    {"name": "progress"},
                ],
            }
        )

    def test_model_registry(self):
        assert "conv_demo" in MODELS
        model = MODELS.instantiate("conv_demo", num_classes=4, channels=(8, 16, 32))
        assert isinstance(model, ConvDemo)

    def test_model_forward(self):
        model = ConvDemo(num_classes=4, channels=(8, 16, 32))
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        assert "logits" in out
        assert out["logits"].shape == (2, 4)

    def test_dataset_registry(self):
        assert "synthetic" in DATASETS

    def test_dataset(self):
        ds = DATASETS.instantiate("synthetic", num_samples_per_class=10, num_classes=4, image_size=64)
        assert len(ds) == 40
        sample = ds[0]
        assert "image" in sample
        assert "label" in sample
        assert sample["image"].shape == (3, 64, 64)

    def test_experiment_forward_loss(self):
        model = MODELS.instantiate("conv_demo", num_classes=4, channels=(8, 16, 32))
        experiment = ClassificationExperiment(
            model=model,
            config={"image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0]},
            num_classes=4,
            loss_config={"name": "cross_entropy"},
        )
        batch = {"image": torch.randn(4, 3, 64, 64), "label": torch.randint(0, 4, (4,))}
        outputs = experiment(batch)
        loss = experiment.compute_loss(batch, outputs)
        assert loss["total"].dim() == 0
        assert loss["total"].item() > 0

    def test_experiment_metrics(self):
        model = MODELS.instantiate("conv_demo", num_classes=4, channels=(8, 16, 32))
        experiment = ClassificationExperiment(
            model=model,
            config={"image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0]},
            num_classes=4,
            loss_config={"name": "cross_entropy"},
        )
        batch = {"image": torch.randn(4, 3, 64, 64), "label": torch.randint(0, 4, (4,))}
        outputs = experiment(batch)
        experiment.val_metrics.update(outputs["logits"].softmax(dim=1), batch["label"])
        result = experiment.val_metrics.compute()
        assert "accuracy" in result
        experiment.val_metrics.reset()

    def test_gradient_flow(self):
        model = ConvDemo(num_classes=4, channels=(8, 16, 32))
        experiment = ClassificationExperiment(
            model=model,
            config={},
            num_classes=4,
            loss_config={"name": "cross_entropy"},
        )
        batch = {"image": torch.randn(4, 3, 64, 64), "label": torch.randint(0, 4, (4,))}
        outputs = experiment(batch)
        loss = experiment.compute_loss(batch, outputs)
        loss["total"].backward()

        grad_params = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_params += 1
        assert grad_params > 0, "No gradients flowed through the model"

    def test_datamodule(self):
        cfg = self._make_demo_config()
        dm = ImageDataModule(config=cfg)
        dm.setup("fit")
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert len(dm.train_dataset) == 80  # 20 * 4
        batch = next(iter(dm.train_dataloader()))
        assert "image" in batch
        assert "label" in batch
        assert batch["image"].shape == (8, 3, 64, 64)

    def test_full_train_one_epoch(self, tmp_path):
        cfg = self._make_demo_config()
        cfg.trainer.max_epochs = 1
        cfg.trainer.batch_size = 4

        from src.training.trainer import Trainer

        trainer = Trainer.build_from_config(cfg)
        trainer.fit()
        assert len(trainer._metrics_history) == 1
        assert "train/loss" in trainer._metrics_history[0]
        epoch_metrics = trainer.experiment.get_epoch_metrics()
        assert "val_accuracy" in epoch_metrics
