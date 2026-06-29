#!/usr/bin/env python
"""Validate the training pipeline before full training: overfit + random label tests."""
import hydra
from omegaconf import DictConfig
from src.training.trainer import Trainer
from src.tools.feasibility import FeasibilityAnalyzer
from src.utils.distributed import setup_logging
from src.utils.seed import set_seed

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    setup_logging()
    set_seed(cfg.seed)
    trainer = Trainer.build_from_config(cfg)
    analyzer = FeasibilityAnalyzer(
        model=trainer.model,
        experiment_class=trainer.experiment.__class__,
        config=cfg,
        datamodule=trainer.datamodule if hasattr(trainer, '_datamodule') else None,
    )
    tool_cfg = cfg.get("tools", {}).get("feasibility", {})
    results = analyzer.run(tests=tool_cfg.get("tests", ["overfit", "random_label"]))
    print("\n" + "="*50)
    print("FEASIBILITY ANALYSIS RESULTS")
    print("="*50)
    for test_name, result in results.items():
        print(f"\n{test_name}:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    if "overfit" in results and not results["overfit"].get("success"):
        print("\n\u26a0 OVERFIT TEST FAILED \u2014 pipeline may be broken!")
    if "random_label" in results and not results["random_label"].get("can_memorize"):
        print("\n\u26a0 RANDOM LABEL TEST FAILED \u2014 model may lack capacity!")

if __name__ == "__main__":
    main()
