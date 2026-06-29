#!/usr/bin/env python
"""Few-shot evaluation with K samples per class."""
import hydra
from omegaconf import DictConfig
from src.training.trainer import Trainer
from src.tools.few_shot import FewShotEvaluator
from src.utils.distributed import setup_logging
from src.utils.seed import set_seed

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    setup_logging()
    set_seed(cfg.seed)
    trainer = Trainer.build_from_config(cfg)
    tool_cfg = cfg.get("tools", {}).get("few_shot", {})
    evaluator = FewShotEvaluator(
        model_factory=lambda: trainer.model.__class__(**{k: v for k, v in cfg.get("model", {}).items() if k != "name"}),
        experiment_class=trainer.experiment.__class__,
        config=cfg,
        datamodule=trainer.datamodule,
    )
    results = evaluator.evaluate(
        k_shots=tool_cfg.get("k_shots", [1, 5, 10, 20]),
        mode=tool_cfg.get("mode", "pretrained"),
        n_episodes=tool_cfg.get("n_episodes", 10),
    )
    print("\nFew-Shot Results:")
    for k, stats in results.items():
        print(f"  {k}: {stats['mean']:.4f} \u00b1 {stats['std']:.4f}")

if __name__ == "__main__":
    main()
