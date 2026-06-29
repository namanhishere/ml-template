#!/usr/bin/env python
"""Fine-tune with configurable strategy."""

import hydra
from omegaconf import DictConfig
from src.training.trainer import Trainer
from src.tools.fine_tuning import FineTuner
from src.utils.distributed import setup_logging
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    setup_logging()
    set_seed(cfg.seed)
    trainer = Trainer.build_from_config(cfg)
    tool_cfg = cfg.get("tools", {}).get("fine_tuning", {})
    finetuner = FineTuner(
        model=trainer.model,
        experiment_class=trainer.experiment.__class__,
        config=cfg,
    )
    results = finetuner.fine_tune(
        strategy_name=tool_cfg.get("strategy", "differential"),
        lr_multiplier=tool_cfg.get("lr_multiplier", 0.1),
        unfreeze_epochs=tool_cfg.get("unfreeze_epochs", 5),
    )
    print(f"Fine-tuning complete. Best val accuracy: {results.get('best_val_acc', 'N/A')}")


if __name__ == "__main__":
    main()
