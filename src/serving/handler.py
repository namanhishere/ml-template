import torch

from src.predictors import PREDICTORS


def create_handler(ckpt_path: str, device: str = None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return PREDICTORS.instantiate("classification", ckpt_path=ckpt_path, device=device)
