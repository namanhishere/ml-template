#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.training.checkpoint import load_checkpoint
from src.training.evaluator import Evaluator
from src.utils.seed import set_seed
from src.utils.registry import EXPERIMENTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    ckpt = load_checkpoint(args.ckpt_path)
    config = OmegaConf.create(ckpt["config"])
    set_seed(ckpt["seed"])
    exp_cls = EXPERIMENTS.get(config.experiment.name)
    from src.models.zoo import BackboneFactory
    backbone = BackboneFactory.create(config.model.name, pretrained=False, num_classes=config.model.num_classes)
    model = backbone
    experiment = exp_cls(model, config)
    experiment.model.load_state_dict(ckpt["model_state"])
    evaluator = Evaluator(experiment, config)
    from src.data.datamodule import BaseDataModule
    dm = BaseDataModule(config)
    dm.setup()
    dl = dm.test_dataloader() if args.split == "test" else dm.val_dataloader()
    metrics = evaluator.evaluate(dl)
    print("Metrics:", metrics)
    evaluator.generate_report(metrics, Path("outputs/reports"))


if __name__ == "__main__":
    main()
