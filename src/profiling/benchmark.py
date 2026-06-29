from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def benchmark_inference(
    model: nn.Module,
    input_shape: tuple[int, int, int, int],
    batch_sizes: list[int],
    device: torch.device | str,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> pd.DataFrame:
    device = torch.device(device)
    model.to(device)
    model.eval()

    records: list[dict[str, Any]] = []

    for bs in batch_sizes:
        shape = (bs,) + input_shape[1:]
        dummy_input = torch.randn(*shape, device=device)

        with torch.no_grad():
            for _ in range(n_warmup):
                model(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

            timings: list[float] = []
            for _ in range(n_runs):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                model(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                end = time.perf_counter()
                timings.append((end - start) * 1000)

        timings_arr = np.array(timings)
        latency_mean = float(np.mean(timings_arr))
        throughput_fps = bs / (latency_mean / 1000.0)

        records.append({
            "batch_size": bs,
            "latency_mean_ms": latency_mean,
            "latency_std_ms": float(np.std(timings_arr)),
            "latency_p50_ms": float(np.percentile(timings_arr, 50)),
            "latency_p95_ms": float(np.percentile(timings_arr, 95)),
            "latency_p99_ms": float(np.percentile(timings_arr, 99)),
            "latency_min_ms": float(np.min(timings_arr)),
            "latency_max_ms": float(np.max(timings_arr)),
            "throughput_fps": throughput_fps,
        })

    return pd.DataFrame(records)
