from __future__ import annotations

import logging
from typing import Any

from src.utils.distributed import is_main_process
from src.utils.registry import CALLBACKS

from .registry import Callback

logger = logging.getLogger("ai-ml-template")


@CALLBACKS.register("mlflow")
class MLflowCallback(Callback):
    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "default",
        run_name: str | None = None,
    ) -> None:
        super().__init__()
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._mlflow = None
        self._available = False

    def on_fit_start(self, trainer: Any) -> None:
        if not is_main_process():
            return

        try:
            import mlflow as mf
            self._mlflow = mf
        except ImportError:
            logger.warning("mlflow is not installed. MLflowCallback will be a no-op.")
            self._available = False
            return

        self._available = True

        if self.tracking_uri:
            self._mlflow.set_tracking_uri(self.tracking_uri)

        self._mlflow.set_experiment(self.experiment_name)

        run = self._mlflow.start_run(run_name=self.run_name)

        config = trainer.config
        flat_params = self._flatten_config(config)
        self._mlflow.log_params(flat_params)

        logger.info("MLflow run started: %s", run.info.run_id)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: dict[str, Any]) -> None:
        if not self._available or not is_main_process():
            return

        log_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_metrics[key] = float(value)
            elif isinstance(value, (list, tuple)):
                for i, v in enumerate(value):
                    if isinstance(v, (int, float)):
                        log_metrics[f"{key}_{i}"] = float(v)

        if log_metrics:
            self._mlflow.log_metrics(log_metrics, step=epoch)

    def on_fit_end(self, trainer: Any) -> None:
        if not self._available or not is_main_process():
            return

        self._mlflow.end_run()
        logger.info("MLflow run ended.")

    @staticmethod
    def _flatten_config(config: Any, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
        items: dict[str, Any] = {}
        if hasattr(config, "items"):
            for k, v in config.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
                if hasattr(v, "items"):
                    items.update(MLflowCallback._flatten_config(v, new_key, sep=sep))
                else:
                    items[new_key] = v
        return items
