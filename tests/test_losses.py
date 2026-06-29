import pytest
import torch

from src.losses.registry import LOSSES


class TestCrossEntropyLoss:
    def test_instantiate(self):
        loss_fn = LOSSES.instantiate("cross_entropy")
        assert loss_fn is not None

    def test_forward_returns_scalar(self):
        loss_fn = LOSSES.instantiate("cross_entropy")
        preds = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(preds, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_forward_batch(self):
        loss_fn = LOSSES.instantiate("cross_entropy")
        preds = torch.randn(16, 100)
        targets = torch.randint(0, 100, (16,))
        loss = loss_fn(preds, targets)
        assert loss.item() > 0

    def test_label_smoothing(self):
        loss_fn = LOSSES.instantiate("cross_entropy", label_smoothing=0.1)
        preds = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_class_weights(self):
        weight = torch.ones(10)
        loss_fn = LOSSES.instantiate("cross_entropy", weight=weight)
        preds = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0

    def test_reduction_sum(self):
        loss_fn = LOSSES.instantiate("cross_entropy", reduction="sum")
        preds = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0
        assert loss > 0

    def test_ignore_index(self):
        loss_fn = LOSSES.instantiate("cross_entropy", ignore_index=-100)
        preds = torch.randn(4, 10)
        targets = torch.tensor([0, -100, 2, 3])
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0


class TestBCELoss:
    def test_instantiate(self):
        loss_fn = LOSSES.instantiate("bce")
        assert loss_fn is not None

    def test_forward_scalar(self):
        loss_fn = LOSSES.instantiate("bce")
        preds = torch.randn(4, 1)
        targets = torch.randint(0, 2, (4, 1))
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0

    def test_forward_multiclass(self):
        loss_fn = LOSSES.instantiate("bce")
        preds = torch.randn(8, 3)
        targets = torch.randint(0, 2, (8, 3))
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0


class TestFocalLoss:
    def test_instantiate(self):
        loss_fn = LOSSES.instantiate("focal")
        assert loss_fn is not None

    def test_forward(self):
        loss_fn = LOSSES.instantiate("focal")
        preds = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_custom_alpha_gamma(self):
        loss_fn = LOSSES.instantiate("focal", alpha=0.5, gamma=3.0)
        preds = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0


class TestMSELoss:
    def test_instantiate(self):
        loss_fn = LOSSES.instantiate("mse")
        assert loss_fn is not None

    def test_forward(self):
        loss_fn = LOSSES.instantiate("mse")
        preds = torch.randn(4, 1)
        targets = torch.randn(4, 1)
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestMAELoss:
    def test_instantiate(self):
        loss_fn = LOSSES.instantiate("mae")
        assert loss_fn is not None

    def test_forward(self):
        loss_fn = LOSSES.instantiate("mae")
        preds = torch.randn(4, 1)
        targets = torch.randn(4, 1)
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0


class TestHuberLoss:
    def test_instantiate(self):
        loss_fn = LOSSES.instantiate("huber")
        assert loss_fn is not None

    def test_forward(self):
        loss_fn = LOSSES.instantiate("huber")
        preds = torch.randn(8, 1)
        targets = torch.randn(8, 1)
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0


class TestDiceLoss:
    def test_instantiate(self):
        loss_fn = LOSSES.instantiate("dice")
        assert loss_fn is not None

    def test_forward_multiclass(self):
        loss_fn = LOSSES.instantiate("dice")
        preds = torch.randn(2, 3, 64, 64)
        targets = torch.randint(0, 3, (2, 64, 64))
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0
        assert 0.0 <= loss.item() <= 2.0

    def test_forward_binary(self):
        loss_fn = LOSSES.instantiate("dice", multiclass=False)
        preds = torch.randn(4, 1, 64, 64)
        targets = torch.randint(0, 2, (4, 64, 64))
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0


class TestCombinedLoss:
    def test_instantiate(self):
        loss_fn = LOSSES.instantiate("combined", losses=[
            {"name": "cross_entropy", "weight": 1.0},
            {"name": "dice", "weight": 0.5},
        ])
        assert loss_fn is not None

    def test_forward(self):
        loss_fn = LOSSES.instantiate("combined", losses=[
            {"name": "cross_entropy", "weight": 1.0},
            {"name": "focal", "weight": 0.5},
        ])
        preds = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        loss = loss_fn(preds, targets)
        assert loss.dim() == 0
        assert loss.item() > 0
