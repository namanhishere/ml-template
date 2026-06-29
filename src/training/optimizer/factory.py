from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("ai-ml-template")


def build_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    optimizer_cfg = cfg.get("optimizer", cfg)
    name = optimizer_cfg.get("name", "adamw").lower()
    params_cfg = optimizer_cfg.get("params", {})

    lr = float(params_cfg.get("lr", 1e-3))
    weight_decay = float(params_cfg.get("weight_decay", 0.0))

    if name in ("adam", "adamw"):
        betas = params_cfg.get("betas", (0.9, 0.999))
        eps = float(params_cfg.get("eps", 1e-8))
        amsgrad = bool(params_cfg.get("amsgrad", False))

        if name == "adamw":
            logger.info(
                "Building AdamW optimizer: lr=%.2e weight_decay=%.2e betas=%s",
                lr,
                weight_decay,
                betas,
            )
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )
        else:
            logger.info(
                "Building Adam optimizer: lr=%.2e weight_decay=%.2e betas=%s",
                lr,
                weight_decay,
                betas,
            )
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )

    elif name == "sgd":
        momentum = float(params_cfg.get("momentum", 0.9))
        dampening = float(params_cfg.get("dampening", 0.0))
        nesterov = bool(params_cfg.get("nesterov", False))
        logger.info(
            "Building SGD optimizer: lr=%.2e momentum=%.2f weight_decay=%.2e nesterov=%s",
            lr,
            momentum,
            weight_decay,
            nesterov,
        )
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )

    elif name == "rmsprop":
        alpha = float(params_cfg.get("alpha", 0.99))
        momentum = float(params_cfg.get("momentum", 0.0))
        eps = float(params_cfg.get("eps", 1e-8))
        centered = bool(params_cfg.get("centered", False))
        logger.info(
            "Building RMSprop optimizer: lr=%.2e alpha=%.2f momentum=%.2f weight_decay=%.2e",
            lr,
            alpha,
            momentum,
            weight_decay,
        )
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=alpha,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps,
            centered=centered,
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}. Available: adam, adamw, sgd, rmsprop")
