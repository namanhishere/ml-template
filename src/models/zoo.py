from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

import torch
import torch.nn as nn
import torchvision.models as tv_models


class _BackendProtocol(Protocol):
    def create(self, name: str, pretrained: bool, **kwargs: Any) -> nn.Module: ...

    def get_feature_dim(self, name: str) -> int: ...

    def list_available(self) -> list[str]: ...


class _TorchVisionBackend:
    _TV_MODELS = {
        "resnet18": (tv_models.resnet18, 512),
        "resnet34": (tv_models.resnet34, 512),
        "resnet50": (tv_models.resnet50, 2048),
        "resnet101": (tv_models.resnet101, 2048),
        "resnet152": (tv_models.resnet152, 2048),
        "resnext50_32x4d": (tv_models.resnext50_32x4d, 2048),
        "resnext101_32x8d": (tv_models.resnext101_32x8d, 2048),
        "wide_resnet50_2": (tv_models.wide_resnet50_2, 2048),
        "wide_resnet101_2": (tv_models.wide_resnet101_2, 2048),
        "efficientnet_b0": (tv_models.efficientnet_b0, 1280),
        "efficientnet_b1": (tv_models.efficientnet_b1, 1280),
        "efficientnet_b2": (tv_models.efficientnet_b2, 1408),
        "efficientnet_b3": (tv_models.efficientnet_b3, 1536),
        "efficientnet_b4": (tv_models.efficientnet_b4, 1792),
        "efficientnet_b5": (tv_models.efficientnet_b5, 2048),
        "efficientnet_b6": (tv_models.efficientnet_b6, 2304),
        "efficientnet_b7": (tv_models.efficientnet_b7, 2560),
        "mobilenet_v2": (tv_models.mobilenet_v2, 1280),
        "mobilenet_v3_small": (tv_models.mobilenet_v3_small, 576),
        "mobilenet_v3_large": (tv_models.mobilenet_v3_large, 960),
        "densenet121": (tv_models.densenet121, 1024),
        "densenet169": (tv_models.densenet169, 1664),
        "densenet201": (tv_models.densenet201, 1920),
        "squeezenet1_0": (tv_models.squeezenet1_0, 512),
        "squeezenet1_1": (tv_models.squeezenet1_1, 512),
        "shufflenet_v2_x0_5": (tv_models.shufflenet_v2_x0_5, 1024),
        "shufflenet_v2_x1_0": (tv_models.shufflenet_v2_x1_0, 1024),
        "shufflenet_v2_x1_5": (tv_models.shufflenet_v2_x1_5, 1024),
        "shufflenet_v2_x2_0": (tv_models.shufflenet_v2_x2_0, 2048),
        "vgg16": (tv_models.vgg16, 4096),
        "vgg19": (tv_models.vgg19, 4096),
        "alexnet": (tv_models.alexnet, 4096),
        "inception_v3": (tv_models.inception_v3, 2048),
        "googlenet": (tv_models.googlenet, 1024),
    }

    _CUSTOM_KEYS: dict[str, list[str]] = {
        "resnet": ["fc", "linear"],
        "resnext": ["fc", "linear"],
        "wide_resnet": ["fc", "linear"],
        "efficientnet": ["classifier"],
        "mobilenet_v2": ["classifier"],
        "mobilenet_v3": ["classifier"],
        "densenet": ["classifier"],
        "squeezenet": ["classifier"],
        "shufflenet": ["fc", "linear"],
        "vgg": ["classifier"],
        "alexnet": ["classifier"],
        "googlenet": ["fc", "linear"],
    }

    def _get_custom_keys(self, model_name: str) -> list[str]:
        for prefix, keys in self._CUSTOM_KEYS.items():
            if model_name.startswith(prefix):
                return keys
        return ["fc", "linear", "classifier", "head", "last_linear"]

    def _remove_head(self, model: nn.Module, model_name: str) -> nn.Module:
        keys = self._get_custom_keys(model_name)
        for key in keys:
            if hasattr(model, key):
                setattr(model, key, nn.Identity())
                return model
        return model

    def create(self, name: str, pretrained: bool = True, **kwargs: Any) -> nn.Module:
        num_classes = kwargs.pop("num_classes", 1000)
        weights = "DEFAULT" if pretrained else None
        entry = self._TV_MODELS.get(name)
        if entry is None:
            available = list(self._TV_MODELS.keys())
            raise KeyError(f"Unknown torchvision model '{name}'. Available: {available}")
        model_fn, _ = entry
        model = model_fn(weights=weights, **kwargs)
        self._remove_head(model, name)
        return model

    def get_feature_dim(self, name: str) -> int:
        entry = self._TV_MODELS.get(name)
        if entry is None:
            return 512
        _, feat_dim = entry
        return feat_dim

    def list_available(self) -> list[str]:
        return list(self._TV_MODELS.keys())


class BackboneFactory:
    source_registry: Dict[str, _BackendProtocol] = {}
    source_registry: Dict[str, _BackendProtocol] = {}

    @classmethod
    def register_source(cls, source_name: str, backend: _BackendProtocol) -> None:
        cls.source_registry[source_name] = backend

    @classmethod
    def create(cls, name: str, pretrained: bool = True, **kwargs: Any) -> nn.Module:
        source, model_name = cls._parse_name(name)
        backend = cls.source_registry.get(source)
        if backend is None:
            raise KeyError(f"Unknown backbone source '{source}'. Available: {list(cls.source_registry.keys())}")
        return backend.create(model_name, pretrained=pretrained, **kwargs)

    @classmethod
    def get_feature_dim(cls, name: str) -> int:
        source, model_name = cls._parse_name(name)
        backend = cls.source_registry.get(source)
        if backend is None:
            raise KeyError(f"Unknown backbone source '{source}'. Available: {list(cls.source_registry.keys())}")
        return backend.get_feature_dim(model_name)

    @classmethod
    def list_available(cls, source: Optional[str] = None) -> list[str]:
        if source is not None:
            backend = cls.source_registry.get(source)
            if backend is None:
                raise KeyError(f"Unknown backbone source '{source}'. Available: {list(cls.source_registry.keys())}")
            return [f"{source}://{m}" for m in backend.list_available()]
        result: list[str] = []
        for src, backend in cls.source_registry.items():
            for model in backend.list_available():
                result.append(f"{src}://{model}")
        return result

    @classmethod
    def _from_torchvision(cls, name: str, pretrained: bool = True, **kwargs: Any) -> nn.Module:
        return cls.create(f"torchvision://{name}", pretrained=pretrained, **kwargs)

    @classmethod
    def _from_timm(cls, name: str, pretrained: bool = True, **kwargs: Any) -> nn.Module:
        return cls.create(f"timm://{name}", pretrained=pretrained, **kwargs)

    @classmethod
    def _from_hf(cls, name: str, pretrained: bool = True, **kwargs: Any) -> nn.Module:
        return cls.create(f"hf://{name}", pretrained=pretrained, **kwargs)

    @staticmethod
    def _parse_name(name: str) -> tuple[str, str]:
        if "://" in name:
            source, model_name = name.split("://", 1)
            return source, model_name
        raise ValueError(f"Invalid backbone name '{name}'. Expected format: 'source://model_name'")


BackboneFactory.register_source("torchvision", _TorchVisionBackend())
