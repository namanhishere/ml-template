from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def get_model_size_mb(model: nn.Module) -> float:
    total_params = sum(p.numel() for p in model.parameters())
    total_size_bytes = sum(
        p.numel() * p.element_size() for p in model.parameters()
    )
    return total_size_bytes / 1e6


def get_gpu_memory_usage() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    usage: dict[str, Any] = {}
    for i in range(torch.cuda.device_count()):
        device_key = f"cuda:{i}"
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1e6
        reserved = torch.cuda.memory_reserved(i) / 1e6
        total = props.total_memory / 1e6

        usage[device_key] = {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "capacity_mb": total,
            "free_mb": total - reserved,
            "name": props.name,
        }

    return usage


def estimate_max_batch_size(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device | str,
    target_memory_gb: float = 8.0,
    min_batch: int = 1,
    max_batch: int = 1024,
) -> int:
    device = torch.device(device)
    model.eval()
    model.to(device)
    target_memory_bytes = target_memory_gb * 1e9

    lo = min_batch
    hi = max_batch
    best = min_batch

    for _ in range(15):
        mid = (lo + hi) // 2
        try:
            example_input = torch.randn(mid, *input_shape[1:], device=device)
            with torch.no_grad():
                _ = model(example_input)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                if peak <= target_memory_bytes:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            else:
                best = mid
                lo = mid + 1

        except RuntimeError:
            hi = mid - 1

    return best
