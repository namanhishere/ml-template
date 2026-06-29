import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image

from src.predictors.classification import ClassificationPredictor


class _ClassifierModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        feats = self.body(x).flatten(1)
        logits = self.fc(feats)
        return {"logits": logits}


class TestClassificationPredictor:
    def test_init_no_checkpoint(self, seed):
        model = _ClassifierModel(num_classes=10)
        predictor = ClassificationPredictor(model=model, device="cpu")
        assert predictor.device.type == "cpu"

    def test_predict_image(self, seed):
        model = _ClassifierModel(num_classes=10)
        predictor = ClassificationPredictor(
            model=model,
            device="cpu",
            class_names=[f"c{i}" for i in range(10)],
        )
        img = Image.fromarray(
            (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        )
        result = predictor.predict(img)
        assert isinstance(result, dict)
        assert "class_id" in result
        assert "class_name" in result
        assert "confidence" in result
        assert "top5" in result
        assert len(result["top5"]) == 5
        assert 0 <= result["class_id"] < 10
        assert 0 <= result["confidence"] <= 1.0

    def test_predict_top5_structure(self, seed):
        model = _ClassifierModel(num_classes=10)
        predictor = ClassificationPredictor(
            model=model,
            device="cpu",
            class_names=[f"c{i}" for i in range(10)],
        )
        img = Image.fromarray(
            (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
        )
        result = predictor.predict(img)
        top5 = result["top5"]
        assert len(top5) == 5
        for entry in top5:
            assert "class_id" in entry
            assert "class_name" in entry
            assert "confidence" in entry

    def test_predict_batch(self, seed):
        model = _ClassifierModel(num_classes=10)
        predictor = ClassificationPredictor(
            model=model,
            device="cpu",
            class_names=[f"c{i}" for i in range(10)],
        )
        imgs = [
            Image.fromarray(
                (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
            )
            for _ in range(3)
        ]
        results = predictor.predict_batch(imgs)
        assert len(results) == 3
        for r in results:
            assert "class_id" in r

    def test_preprocess_returns_tensor(self, seed):
        model = _ClassifierModel(num_classes=10)
        predictor = ClassificationPredictor(model=model, device="cpu")
        img = Image.fromarray(
            (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        )
        tensor = predictor.preprocess(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.ndim == 4
        assert tensor.shape[1:] == (3, 224, 224)

    def test_postprocess_returns_dict(self, seed):
        model = _ClassifierModel(num_classes=10)
        predictor = ClassificationPredictor(
            model=model,
            device="cpu",
            class_names=[f"c{i}" for i in range(10)],
        )
        outputs = {"logits": torch.randn(1, 10)}
        result = predictor.postprocess(outputs)
        assert isinstance(result, dict)
        assert "top5" in result

    def test_visualize(self, seed, tmp_path):
        model = _ClassifierModel(num_classes=10)
        predictor = ClassificationPredictor(
            model=model,
            device="cpu",
            class_names=[f"c{i}" for i in range(10)],
        )
        img = Image.fromarray(
            (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        )
        save_path = tmp_path / "pred_viz.png"
        fig = predictor.visualize(img, save_path=str(save_path))
        assert save_path.exists()

    def test_save_load_checkpoint(self, seed, tmp_path):
        model = _ClassifierModel(num_classes=10)
        ckpt_path = tmp_path / "model.pt"
        torch.save({"state_dict": model.state_dict()}, ckpt_path)

        new_model = _ClassifierModel(num_classes=10)
        predictor = ClassificationPredictor(
            model=new_model,
            ckpt_path=str(ckpt_path),
            device="cpu",
        )
        assert isinstance(predictor.model, nn.Module)

    def test_device_gpu_if_available(self, seed):
        model = _ClassifierModel(num_classes=10)
        if torch.cuda.is_available():
            predictor = ClassificationPredictor(model=model, device="cuda")
            assert predictor.device.type == "cuda"
            assert next(predictor.model.parameters()).device.type == "cuda"

    def test_predict_consistent_output(self, seed):
        model = _ClassifierModel(num_classes=10)
        predictor = ClassificationPredictor(model=model, device="cpu")
        img = Image.fromarray(
            (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        )
        torch.manual_seed(42)
        r1 = predictor.predict(img)
        torch.manual_seed(42)
        r2 = predictor.predict(img)
        assert r1["class_id"] == r2["class_id"]
