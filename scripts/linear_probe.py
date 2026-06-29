#!/usr/bin/env python
"""Linear probe: freeze backbone, train linear classifier."""
import hydra
from omegaconf import DictConfig
from src.training.trainer import Trainer
from src.tools.linear_probe import LinearProbe
from src.utils.distributed import setup_logging
from src.utils.seed import set_seed

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    setup_logging()
    set_seed(cfg.seed)
    trainer = Trainer.build_from_config(cfg)
    backbone = trainer.model
    num_classes = cfg.get("model", {}).get("num_classes", cfg.get("dataset", {}).get("num_classes", 10))
    tool_cfg = cfg.get("tools", {}).get("linear_probe", {})
    probe = LinearProbe(backbone=backbone, num_classes=num_classes, config=cfg)
    probe._validate_pretrained()
    train_dl = trainer.datamodule.train_dataloader()
    val_dl = trainer.datamodule.val_dataloader()
    results = probe.train(train_dl, val_dl, max_epochs=tool_cfg.get("max_epochs", 50), lr=tool_cfg.get("lr", 0.01))
    print(f"\nLinear Probe Results: train_acc={results.get('train/accuracy'):.4f}, val_acc={results.get('val/accuracy'):.4f}")

if __name__ == "__main__":
    main()
