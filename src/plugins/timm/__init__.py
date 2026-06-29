"""timm plugin — registers timm backbones with BackboneFactory."""

from src.models.zoo import BackboneFactory
import logging


class TimmBackend:
    def create(self, name: str, pretrained: bool = True, **kwargs):
        try:
            import timm

            model = timm.create_model(name, pretrained=pretrained, **kwargs)
            if hasattr(model, "reset_classifier"):
                model.reset_classifier(0)
            elif hasattr(model, "head"):
                import torch.nn as nn

                model.head = nn.Identity()
            elif hasattr(model, "classifier"):
                import torch.nn as nn

                model.classifier = nn.Identity()
            return model
        except ImportError:
            raise ImportError("timm is not installed. Install with: pip install timm")

    def get_feature_dim(self, name: str, **kwargs):
        try:
            import torch
            import timm

            model = timm.create_model(name, pretrained=False, num_classes=0)
            features = model(torch.randn(1, 3, 224, 224)).shape[1]
            return features
        except Exception:
            return 2048

    def list_available(self):
        try:
            import timm

            return timm.list_models()
        except ImportError:
            return []


def register():
    try:
        BackboneFactory.register_source("timm", TimmBackend())
        logging.getLogger("ai-ml-template").info("Registered timm backend")
    except Exception as e:
        logging.getLogger("ai-ml-template").warning(f"Failed to register timm: {e}")


register()
