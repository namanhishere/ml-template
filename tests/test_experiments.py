import pytest
import torch
import torch.nn as nn

from src.experiments.classification import ClassificationExperiment


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        feats = self.backbone(x).flatten(1)
        return {"logits": self.head(feats)}


class TestClassificationExperiment:
    @pytest.fixture
    def experiment(self, seed, dummy_config):
        model = SimpleCNN(num_classes=10)
        return ClassificationExperiment(
            model=model,
            config={"image_mean": [0.0, 0.0, 0.0], "image_std": [1.0, 1.0, 1.0]},
            num_classes=10,
            loss_config={"name": "cross_entropy"},
        )

    @pytest.fixture
    def batch(self):
        return {"image": torch.randn(4, 3, 224, 224), "label": torch.randint(0, 10, (4,))}

    def test_forward_returns_dict(self, experiment, batch):
        outputs = experiment(batch)
        assert isinstance(outputs, dict)
        assert "logits" in outputs
        assert outputs["logits"].shape == (4, 10)

    def test_forward_output_shape(self, experiment, batch):
        outputs = experiment(batch)
        assert outputs["logits"].dtype == torch.float32

    def test_compute_loss(self, experiment, batch):
        outputs = experiment(batch)
        loss = experiment.compute_loss(batch, outputs)
        assert isinstance(loss, dict)
        assert "total" in loss
        assert loss["total"].dim() == 0
        assert loss["total"].item() > 0

    def test_postprocess_returns_labels(self, experiment, batch):
        outputs = experiment(batch)
        predicted = experiment.postprocess(outputs)
        assert predicted.shape == (4,)
        assert predicted.max() < 10
        assert predicted.min() >= 0

    def test_compute_metrics(self, experiment, batch):
        outputs = experiment(batch)
        logits = outputs["logits"]
        labels = batch["label"].to(experiment.device)
        probs = torch.softmax(logits, dim=1)
        experiment.val_metrics.update(probs, labels)
        result = experiment.val_metrics.compute()
        assert "accuracy" in result
        experiment.val_metrics.reset()

    def test_on_epoch_end_resets_metrics(self, experiment, batch):
        outputs = experiment(batch)
        logits = outputs["logits"]
        labels = batch["label"].to(experiment.device)
        probs = torch.softmax(logits, dim=1)
        experiment.train_metrics.update(probs, labels)
        experiment.on_epoch_end("train")
        epoch_metrics = experiment.get_epoch_metrics()
        assert len(epoch_metrics) > 0

    def test_clear_epoch_metrics(self, experiment, batch):
        outputs = experiment(batch)
        logits = outputs["logits"]
        labels = batch["label"].to(experiment.device)
        probs = torch.softmax(logits, dim=1)
        experiment.train_metrics.update(probs, labels)
        experiment.on_epoch_end("train")
        assert len(experiment.get_epoch_metrics()) > 0
        experiment.clear_epoch_metrics()
        assert len(experiment.get_epoch_metrics()) == 0

    def test_train_eval_modes(self, experiment):
        experiment.train()
        assert experiment.model.training
        experiment.eval()
        assert not experiment.model.training

    def test_device_property(self, experiment):
        dev = experiment.device
        assert isinstance(dev, torch.device)

    def test_visualize(self, experiment, batch, tmp_path):
        outputs = experiment(batch)
        experiment.visualize(batch, outputs, tmp_path, "test")
        saved_file = tmp_path / "test_class_samples.png"
        assert saved_file.exists()

    def test_forward_cuda_batch(self, experiment):
        if torch.cuda.is_available():
            batch = {
                "image": torch.randn(4, 3, 224, 224).cuda(),
                "label": torch.randint(0, 10, (4,)).cuda(),
            }
            experiment.model = experiment.model.cuda()
            outputs = experiment(batch)
            assert outputs["logits"].device.type == "cuda"
