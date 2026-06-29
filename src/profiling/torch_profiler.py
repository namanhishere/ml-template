from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def profile_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        timings: list[float] = []
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.perf_counter()
            timings.append((end - start) * 1000)

    timings_arr = np.array(timings)

    return {
        "latency_p50_ms": float(np.percentile(timings_arr, 50)),
        "latency_p95_ms": float(np.percentile(timings_arr, 95)),
        "latency_p99_ms": float(np.percentile(timings_arr, 99)),
        "latency_mean_ms": float(np.mean(timings_arr)),
        "latency_std_ms": float(np.std(timings_arr)),
        "latency_min_ms": float(np.min(timings_arr)),
        "latency_max_ms": float(np.max(timings_arr)),
    }
