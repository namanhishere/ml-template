#!/usr/bin/env python
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from src.training.trainer import Trainer
from src.utils.config_validation import validate_train_config
from src.utils.distributed import setup_logging
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config_demo")
def main(cfg: DictConfig):
    setup_logging()
    set_seed(cfg.seed)
    warnings = validate_train_config(OmegaConf.to_container(cfg, resolve=True))
    logger = logging.getLogger("ai-ml-template")
    for w in warnings:
        logger.warning(w)
    logger.info("Starting ConvDemo pipeline — conv-only model + synthetic data")
    trainer = Trainer.build_from_config(cfg)
    trainer.fit()
    logger.info("ConvDemo pipeline completed successfully")


if __name__ == "__main__":
    main()
