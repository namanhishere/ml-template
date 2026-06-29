"""HuggingFace plugin — registers HF models with BackboneFactory."""
from src.models.zoo import BackboneFactory
import logging

class HuggingFaceBackend:
    def create(self, name: str, pretrained: bool = True, **kwargs):
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(name if pretrained else None, **kwargs)
            return model
        except ImportError:
            raise ImportError("transformers is not installed. Install with: pip install transformers")

    def get_feature_dim(self, name: str, **kwargs):
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(name)
            return config.hidden_size
        except Exception:
            return 768

    def list_available(self):
        return ["bert-base-uncased", "roberta-base", "distilbert-base-uncased", "gpt2"]

def register():
    try:
        BackboneFactory.register_source("hf", HuggingFaceBackend())
        logging.getLogger("ai-ml-template").info("Registered huggingface backend")
    except Exception as e:
        logging.getLogger("ai-ml-template").warning(f"Failed to register huggingface: {e}")

register()
