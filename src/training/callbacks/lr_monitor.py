from __future__ import annotations

import logging
from typing import Any

from src.utils.distributed import is_main_process
from src.utils.registry import CALLBACKS

from .registry import Callback

logger = logging.getLogger("ai-ml-template")


@CALLBACKS.register("lr_monitor")
class LRMonitor(Callback):
    def __init__(self, log_every_n_steps: int = 100) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if not is_main_process():
            return
        if trainer._global_step % self.log_every_n_steps != 0:
            return

        optimizer = trainer.optimizer
        if optimizer is None:
            return

        lrs = []
        for param_group in optimizer.param_groups:
            lr = param_group.get("lr")
            if lr is not None:
                lrs.append(lr)

        if len(lrs) == 1:
            logger.info("step=%d lr=%.8f", trainer._global_step, lrs[0])
        else:
            lr_str = ", ".join(f"group{i}={lr:.8f}" for i, lr in enumerate(lrs))
            logger.info("step=%d lr=[%s]", trainer._global_step, lr_str)
