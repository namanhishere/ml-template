from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from src.models.zoo import BackboneFactory


class _ViTBackend:
    _VISION_TRANSFORMERS: Dict[str, tuple[str, int]] = {
        "vit_b_16": ("torchvision", 768),
        "vit_b_32": ("torchvision", 768),
        "vit_l_16": ("torchvision", 1024),
        "vit_l_32": ("torchvision", 1024),
        "vit_h_14": ("torchvision", 1280),
    }

    def create(self, name: str, pretrained: bool, **kwargs: Any) -> nn.Module:
        return ViTBackbone(name, pretrained=pretrained, **kwargs)

    def get_feature_dim(self, name: str) -> int:
        info = self._VISION_TRANSFORMERS.get(name)
        if info is None:
            raise ValueError(f"Unknown ViT model '{name}'. Available: {self.list_available()}")
        return info[1]

    def list_available(self) -> list[str]:
        return sorted(self._VISION_TRANSFORMERS.keys())


class ViTBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        weights: Any = None,
        image_size: int = 224,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if model_name not in _ViTBackend._VISION_TRANSFORMERS:
            raise ValueError(
                f"Unknown ViT model '{model_name}'. Available: {sorted(_ViTBackend._VISION_TRANSFORMERS.keys())}"
            )

        self.model_name = model_name
        self.feature_dim = _ViTBackend._VISION_TRANSFORMERS[model_name][1]
        self.image_size = image_size
        self.vit = self._build_vit(model_name, pretrained, weights, image_size)

    def _build_vit(self, name: str, pretrained: bool, weights: Any, image_size: int) -> nn.Module:
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
                model = model_fn(weights=weights_enum, image_size=image_size)
            except Exception:
                model = model_fn(weights=None if not pretrained else "DEFAULT", image_size=image_size)
        elif weights is not None:
            model = model_fn(weights=weights, image_size=image_size)
        else:
            model = model_fn(image_size=image_size)

        model.heads = nn.Identity()
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vit._process_input(x)
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)
        return x[:, 0]

    def get_feature_dim(self) -> int:
        return self.feature_dim


_backend = _ViTBackend()
BackboneFactory.register_source("vit", _backend)
