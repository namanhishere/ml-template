import pytest
import torch

from src.training.metrics.registry import METRICS


class TestClassificationMetrics:
    def test_instantiate(self):
        metrics = METRICS.instantiate("classification", num_classes=10)
        assert metrics is not None

    def test_update_compute_reset_cycle(self, seed):
        metrics = METRICS.instantiate("classification", num_classes=10)

        preds = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        metrics.update(preds, targets)

        result = metrics.compute()
        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "f1_score" in result
        assert "precision" in result
        assert "recall" in result
        assert "auroc" in result

        for v in result.values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

        metrics.reset()

    def test_multiple_updates(self, seed):
        metrics = METRICS.instantiate("classification", num_classes=10)

        for _ in range(5):
            preds = torch.randn(16, 10)
            targets = torch.randint(0, 10, (16,))
            metrics.update(preds, targets)

        result = metrics.compute()
        assert all(isinstance(v, float) for v in result.values())
        metrics.reset()

    def test_perfect_predictions(self):
        metrics = METRICS.instantiate("classification", num_classes=5)
        targets = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        probs = torch.zeros(10, 5)
        for i, t in enumerate(targets):
            probs[i, t] = 100.0
        metrics.update(probs, targets)
        result = metrics.compute()
        assert result["accuracy"] == 1.0
        assert result["f1_score"] == 1.0
        metrics.reset()

    def test_worst_predictions(self):
        metrics = METRICS.instantiate("classification", num_classes=4)
        targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        probs = torch.zeros(8, 4)
        wrong_map = {0: 3, 1: 2, 2: 1, 3: 0}
        for i, t in enumerate(targets):
            probs[i, wrong_map[t.item()]] = 100.0
        metrics.update(probs, targets)
        result = metrics.compute()
        assert result["accuracy"] == 0.0
        metrics.reset()

    def test_custom_average(self):
        metrics = METRICS.instantiate("classification", num_classes=10, average="macro")
        preds = torch.randn(20, 10)
        targets = torch.randint(0, 10, (20,))
        metrics.update(preds, targets)
        result = metrics.compute()
        assert isinstance(result["accuracy"], float)
        metrics.reset()

    def test_binary_classification(self):
        metrics = METRICS.instantiate("classification", num_classes=2, task="binary")
        targets = torch.randint(0, 2, (32,))
        preds = torch.randint(0, 2, (32,))
        metrics.update(preds, targets)
        result = metrics.compute()
        assert "accuracy" in result
        metrics.reset()


class TestSegmentationMetrics:
    def test_instantiate(self):
        metrics = METRICS.instantiate("segmentation", num_classes=5)
        assert metrics is not None

    def test_update_compute_reset_cycle(self, seed):
        metrics = METRICS.instantiate("segmentation", num_classes=5)

        preds = torch.randn(4, 5, 64, 64)
        targets = torch.randint(0, 5, (4, 64, 64))
        metrics.update(preds, targets)

        result = metrics.compute()
        assert isinstance(result, dict)
        assert "iou" in result
        assert "dice" in result
        assert "accuracy" in result

        for v in result.values():
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

        metrics.reset()

    def test_update_with_logits(self, seed):
        metrics = METRICS.instantiate("segmentation", num_classes=3)

        preds = torch.randn(8, 3, 32, 32)
        targets = torch.randint(0, 3, (8, 32, 32))
        metrics.update(preds, targets)

        result = metrics.compute()
        assert all(isinstance(v, float) for v in result.values())
        metrics.reset()

    def test_multiple_updates(self, seed):
        metrics = METRICS.instantiate("segmentation", num_classes=4)

        for _ in range(3):
            preds = torch.randn(4, 4, 16, 16)
            targets = torch.randint(0, 4, (4, 16, 16))
            metrics.update(preds, targets)

        result = metrics.compute()
        assert all(isinstance(v, float) for v in result.values())
        metrics.reset()
