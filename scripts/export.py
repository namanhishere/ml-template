#!/usr/bin/env python
import argparse
from pathlib import Path

import torch

from src.export.onnx import ONNXExporter
from src.export.torchscript import TorchScriptExporter
from src.training.checkpoint import load_checkpoint
from omegaconf import OmegaConf
from src.models.zoo import BackboneFactory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--format", choices=["onnx", "torchscript"], default="onnx")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    ckpt = load_checkpoint(args.ckpt_path)
    config = OmegaConf.create(ckpt["config"])
    model = BackboneFactory.create(config.model.name, pretrained=False, num_classes=config.model.num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    sample = torch.randn(1, 3, 224, 224)
    output_path = args.output or f"export/model.{args.format}"
    if args.format == "onnx":
        ONNXExporter().export(model, sample, Path(output_path))
    else:
        TorchScriptExporter().export(model, sample, Path(output_path))
    print(f"Exported to {output_path}")


if __name__ == "__main__":
    main()
