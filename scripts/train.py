#!/usr/bin/env python
import hydra
from omegaconf import DictConfig, OmegaConf

from src.training.trainer import Trainer
from src.utils.distributed import setup_logging
from src.utils.seed import set_seed
from src.utils.config_validation import validate_train_config

import logging


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    setup_logging()
    set_seed(cfg.seed)
    warnings = validate_train_config(OmegaConf.to_container(cfg, resolve=True))
    for w in warnings:
        logging.getLogger("ai-ml-template").warning(w)
    trainer = Trainer.build_from_config(cfg)
    trainer.fit()


if __name__ == "__main__":
    main()
