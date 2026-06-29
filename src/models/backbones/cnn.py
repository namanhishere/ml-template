from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch.nn as nn

from src.models.zoo import BackboneFactory


class _CNNBackend:
    _TORCHVISION_CNNS: Dict[str, tuple[str, int]] = {
        "resnet18": ("torchvision", 512),
        "resnet34": ("torchvision", 512),
        "resnet50": ("torchvision", 2048),
        "resnet101": ("torchvision", 2048),
        "resnet152": ("torchvision", 2048),
        "resnext50_32x4d": ("torchvision", 2048),
        "resnext101_32x8d": ("torchvision", 2048),
        "resnext101_64x4d": ("torchvision", 2048),
        "wide_resnet50_2": ("torchvision", 2048),
        "wide_resnet101_2": ("torchvision", 2048),
        "efficientnet_b0": ("torchvision", 1280),
        "efficientnet_b1": ("torchvision", 1280),
        "efficientnet_b2": ("torchvision", 1408),
        "efficientnet_b3": ("torchvision", 1536),
        "efficientnet_b4": ("torchvision", 1792),
        "efficientnet_b5": ("torchvision", 2048),
        "efficientnet_b6": ("torchvision", 2304),
        "efficientnet_b7": ("torchvision", 2560),
        "efficientnet_v2_s": ("torchvision", 1280),
        "efficientnet_v2_m": ("torchvision", 1280),
        "efficientnet_v2_l": ("torchvision", 1280),
        "densenet121": ("torchvision", 1024),
        "densenet161": ("torchvision", 2208),
        "densenet169": ("torchvision", 1664),
        "densenet201": ("torchvision", 1920),
        "mobilenet_v2": ("torchvision", 1280),
        "mobilenet_v3_small": ("torchvision", 576),
        "mobilenet_v3_large": ("torchvision", 960),
        "convnext_tiny": ("torchvision", 768),
        "convnext_small": ("torchvision", 768),
        "convnext_base": ("torchvision", 1024),
        "convnext_large": ("torchvision", 1536),
        "regnet_y_400mf": ("torchvision", 440),
        "regnet_y_800mf": ("torchvision", 784),
        "regnet_y_1_6gf": ("torchvision", 888),
        "regnet_y_3_2gf": ("torchvision", 1512),
        "regnet_y_8gf": ("torchvision", 2016),
        "regnet_y_16gf": ("torchvision", 3024),
        "regnet_y_32gf": ("torchvision", 3712),
        "vgg16": ("torchvision", 4096),
        "vgg19": ("torchvision", 4096),
        "shufflenet_v2_x0_5": ("torchvision", 1024),
        "shufflenet_v2_x1_0": ("torchvision", 1024),
        "shufflenet_v2_x1_5": ("torchvision", 1024),
        "shufflenet_v2_x2_0": ("torchvision", 2048),
        "mnasnet0_5": ("torchvision", 1280),
        "mnasnet0_75": ("torchvision", 1280),
        "mnasnet1_0": ("torchvision", 1280),
        "mnasnet1_3": ("torchvision", 1280),
        "squeezenet1_0": ("torchvision", 512),
        "squeezenet1_1": ("torchvision", 512),
        "inception_v3": ("torchvision", 2048),
        "googlenet": ("torchvision", 1024),
        "alexnet": ("torchvision", 4096),
    }

    def create(self, name: str, pretrained: bool, **kwargs: Any) -> nn.Module:
        return CNNBackbone(name, pretrained=pretrained, **kwargs)

    def get_feature_dim(self, name: str) -> int:
        info = self._TORCHVISION_CNNS.get(name)
        if info is None:
            raise ValueError(
                f"Unknown CNN model '{name}'. Available: {self.list_available()}"
            )
        return info[1]

    def list_available(self) -> list[str]:
        return sorted(self._TORCHVISION_CNNS.keys())


class CNNBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        weights: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if model_name not in _CNNBackend._TORCHVISION_CNNS:
            raise ValueError(
                f"Unknown CNN model '{model_name}'. "
                f"Available: {sorted(_CNNBackend._TORCHVISION_CNNS.keys())}"
            )

        self.model_name = model_name
        self.encoder = self._build_encoder(model_name, pretrained, weights)
        self.feature_dim = _CNNBackend._TORCHVISION_CNNS[model_name][1]

    def _build_encoder(
        self, name: str, pretrained: bool, weights: Any
    ) -> nn.Module:
        import torchvision.models as models

        if weights is None and pretrained:
            try:
                weights = "IMAGENET1K_V1"
            except Exception:
                pass

        model_fn = getattr(models, name, None)
        if model_fn is None:
            raise ValueError(f"torchvision has no model named '{name}'")

        if weights is not None and pretrained:
            try:
                from torchvision.models import get_weight
                weights_enum = get_weight(weights)
                model = model_fn(weights=weights_enum)
            except Exception:
                model = model_fn(pretrained=True)
        elif weights is not None:
            model = model_fn(weights=weights)
        else:
            model = model_fn(weights=None)

        _remove_classifier(model, name)
        return model

    def forward(self, x: Any) -> Any:
        return self.encoder(x)

    def get_feature_dim(self) -> int:
        return self.feature_dim


def _remove_classifier(model: nn.Module, name: str) -> None:
    resnet_like = (
        "resnet", "resnext", "wide_resnet",
        "shufflenet", "mnasnet", "regnet",
    )
    if any(name.startswith(p) for p in resnet_like):
        model.fc = nn.Identity()
    elif name.startswith("densenet"):
        model.classifier = nn.Identity()
    elif name.startswith("efficientnet") and not name.startswith("efficientnet_v2"):
        model.classifier = nn.Identity()
    elif name.startswith("efficientnet_v2"):
        model.classifier = nn.Identity()
    elif name.startswith("mobilenet_v2"):
        model.classifier = nn.Identity()
    elif name.startswith("mobilenet_v3"):
        model.classifier = nn.Identity()
    elif name.startswith("convnext"):
        model.classifier = nn.Identity()
    elif name.startswith("vgg"):
        model.classifier = nn.Identity()
    elif name.startswith("squeezenet"):
        model.classifier = nn.Identity()
    elif name in ("inception_v3", "googlenet"):
        model.fc = nn.Identity()
    elif name == "alexnet":
        model.classifier = nn.Identity()


_backend = _CNNBackend()
BackboneFactory.register_source("torchvision", _backend)
BackboneFactory.register_source("cnn", _backend)
