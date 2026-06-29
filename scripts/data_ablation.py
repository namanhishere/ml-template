#!/usr/bin/env python
"""Data ablation: measure impact of removing data portions."""

import hydra
from omegaconf import DictConfig
from pathlib import Path
from src.training.trainer import Trainer
from src.tools.data_ablation import DataAblator
from src.tools.reporting import ReportGenerator
from src.utils.distributed import setup_logging
from src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    setup_logging()
    set_seed(cfg.seed)
    trainer = Trainer.build_from_config(cfg)
    tool_cfg = cfg.get("tools", {}).get("data_ablation", {})
    ablator = DataAblator(
        model=trainer.model,
        experiment_class=trainer.experiment.__class__,
        config=cfg,
        datamodule=trainer.datamodule,
    )
    results = ablator.run_full_ablation()
    output_dir = Path("reports") / "ablation"
    reporter = ReportGenerator(output_dir=output_dir)
    report_path = reporter.generate_ablation_report(results, output_dir)
    print(f"\nAblation report saved to: {report_path}")
    if "influence" in results:
        n_important = min(10, len(results["influence"].get("forgetting_ranking", [])))
        print(f"Top {n_important} most forgettable samples: {results['influence']['forgetting_ranking'][:n_important]}")


if __name__ == "__main__":
    main()
