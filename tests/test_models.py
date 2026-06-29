import pytest
import torch
import torch.nn as nn

import src.models.backbones.cnn  # noqa: F401 — triggers BackboneFactory registration
from src.models.zoo import BackboneFactory
from src.models.registry import MODELS
from src.models.base import BaseModel
from src.models.heads.classification import ClassificationHead


class DummyModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = ClassificationHead(64, num_classes)

    def forward(self, x):
        feats = self.backbone(x).flatten(1)
        logits = self.head(feats)
        return {"logits": logits, "features": feats}


@MODELS.register("dummy_test_model")
class RegisteredDummyModel(DummyModel):
    pass


class TestBackboneFactory:
    def test_create_torchvision_resnet18(self, seed):
        backbone = BackboneFactory.create("torchvision://resnet18", pretrained=False)
        assert isinstance(backbone, nn.Module)
        x = torch.randn(1, 3, 224, 224)
        out = backbone(x)
        assert out.ndim == 2
        assert out.shape[0] == 1

    def test_create_torchvision_efficientnet_b0(self, seed):
        backbone = BackboneFactory.create("torchvision://efficientnet_b0", pretrained=False)
        assert isinstance(backbone, nn.Module)
        out = backbone(torch.randn(1, 3, 224, 224))
        assert out.shape[0] == 1

    def test_create_torchvision_densenet121(self, seed):
        backbone = BackboneFactory.create("torchvision://densenet121", pretrained=False)
        assert isinstance(backbone, nn.Module)
        out = backbone(torch.randn(1, 3, 224, 224))
        assert out.shape[0] == 1

    def test_create_torchvision_mobilenet_v2(self, seed):
        backbone = BackboneFactory.create("torchvision://mobilenet_v2", pretrained=False)
        assert isinstance(backbone, nn.Module)
        out = backbone(torch.randn(1, 3, 224, 224))
        assert out.shape[0] == 1

    def test_create_torchvision_mobilenet_v3_small(self, seed):
        backbone = BackboneFactory.create("torchvision://mobilenet_v3_small", pretrained=False)
        assert isinstance(backbone, nn.Module)

    def test_get_feature_dim_resnet18(self):
        dim = BackboneFactory.get_feature_dim("torchvision://resnet18")
        assert dim == 512

    def test_get_feature_dim_resnet50(self):
        dim = BackboneFactory.get_feature_dim("torchvision://resnet50")
        assert dim == 2048

    def test_get_feature_dim_densenet121(self):
        dim = BackboneFactory.get_feature_dim("torchvision://densenet121")
        assert dim == 1024

    def test_get_feature_dim_efficientnet_b0(self):
        dim = BackboneFactory.get_feature_dim("torchvision://efficientnet_b0")
        assert dim == 1280

    def test_list_available_torchvision(self):
        models = BackboneFactory.list_available("torchvision")
        assert len(models) > 10
        assert "torchvision://resnet18" in models
        assert "torchvision://resnet50" in models

    def test_list_available_all(self):
        models = BackboneFactory.list_available()
        assert len(models) > 10
        assert any("torchvision://" in m for m in models)

    def test_parse_name_format(self):
        source, name = BackboneFactory._parse_name("torchvision://resnet18")
        assert source == "torchvision"
        assert name == "resnet18"

    def test_parse_name_invalid(self):
        with pytest.raises(ValueError):
            BackboneFactory._parse_name("resnet18")

    def test_unknown_source_raises(self):
        with pytest.raises(KeyError):
            BackboneFactory.create("unknown://model", pretrained=False)


class TestModelRegistry:
    def test_register_and_retrieve(self):
        assert "dummy_test_model" in MODELS
        model_cls = MODELS.get("dummy_test_model")
        assert model_cls is RegisteredDummyModel

    def test_instantiate_model(self):
        model = MODELS.instantiate("dummy_test_model", num_classes=5)
        assert isinstance(model, BaseModel)
        assert isinstance(model, nn.Module)

    def test_instantiate_unknown_raises(self):
        with pytest.raises(KeyError):
            MODELS.instantiate("nonexistent_model")


class TestBaseModel:
    def test_forward_returns_dict(self, seed):
        model = DummyModel(num_classes=10)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert isinstance(out, dict)
        assert "logits" in out
        assert out["logits"].shape == (4, 10)

    def test_forward_batch_size(self, seed):
        model = DummyModel(num_classes=5)
        out = model(torch.randn(8, 3, 224, 224))
        assert out["logits"].shape == (8, 5)

    def test_get_backbone(self):
        model = DummyModel(num_classes=10)
        backbone = model.get_backbone()
        assert isinstance(backbone, nn.Module)

    def test_get_head(self):
        model = DummyModel(num_classes=10)
        head = model.get_head()
        assert isinstance(head, nn.Module)

    def test_count_params(self):
        model = DummyModel(num_classes=10)
        n_params = model.count_params()
        assert n_params > 0

    def test_freeze_unfreeze_backbone(self, seed):
        model = DummyModel(num_classes=10)
        model.unfreeze_backbone()
        for p in model.get_backbone().parameters():
            assert p.requires_grad

    def test_freezer_context(self, seed):
        model = DummyModel(num_classes=10)
        backbone = model.get_backbone()
        with model.freezer():
            for p in backbone.parameters():
                assert not p.requires_grad
        for p in backbone.parameters():
            assert p.requires_grad

    def test_repr(self):
        model = DummyModel(num_classes=10)
        r = repr(model)
        assert "DummyModel" in r
        assert "trainable=" in r
