#!/usr/bin/env python
import argparse

import torch

from src.training.checkpoint import load_checkpoint
from omegaconf import OmegaConf
from src.models.zoo import BackboneFactory
from src.profiling.benchmark import benchmark_inference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16,32")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    ckpt = load_checkpoint(args.ckpt_path)
    config = OmegaConf.create(ckpt["config"])
    model = BackboneFactory.create(config.model.name, pretrained=False, num_classes=config.model.num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device).eval()
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    input_shape = (3, config.get("dataset", {}).get("image_size", 224), config.get("dataset", {}).get("image_size", 224))
    df = benchmark_inference(model, input_shape, batch_sizes, args.device)
    print(df.to_string())


if __name__ == "__main__":
    main()
