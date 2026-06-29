from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    class_idx: int | None = None,
    save_path: str | Path | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None

    def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
        nonlocal activations
        activations = out.detach()

    def backward_hook(module: nn.Module, grad_in: Any, grad_out: Any) -> None:
        nonlocal gradients
        gradients = grad_out[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.eval()
    model.zero_grad()

    output = model(input_tensor)

    if class_idx is None:
        if isinstance(output, torch.Tensor):
            class_idx = output.argmax(dim=1).item()
        elif isinstance(output, dict):
            logits = next(v for v in output.values() if isinstance(v, torch.Tensor))
            class_idx = logits.argmax(dim=1).item()

    one_hot = torch.zeros_like(
        output if isinstance(output, torch.Tensor) else next(v for v in output.values() if isinstance(v, torch.Tensor))
    )
    one_hot[0, class_idx] = 1
    if isinstance(output, dict):
        logits = next(v for v in output.values() if isinstance(v, torch.Tensor))
        logits.backward(gradient=one_hot, retain_graph=True)
    else:
        output.backward(gradient=one_hot, retain_graph=True)

    forward_handle.remove()
    backward_handle.remove()

    if activations is None or gradients is None:
        raise RuntimeError("Failed to capture activations or gradients from target layer.")

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    cam = cam.squeeze(0).squeeze(0)
    cam = cam - cam.min()
    denom = cam.max()
    if denom > 0:
        cam = cam / denom

    cam = cam.cpu().numpy()

    cam_resized = _resize_cam(cam, input_tensor.shape[-2], input_tensor.shape[-1])

    input_img = input_tensor.detach().squeeze(0).cpu()
    if input_img.ndim == 3:
        input_img = input_img.permute(1, 2, 0).numpy()
        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)

    import matplotlib.pyplot as plt
    from matplotlib import cm

    cmap = cm.get_cmap("jet")
    heatmap = cmap(cam_resized)[:, :, :3]

    overlay = alpha * heatmap + (1 - alpha) * input_img
    overlay = np.clip(overlay, 0, 1)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.imshow(input_img)
        ax1.set_title("Original")
        ax1.axis("off")
        ax2.imshow(overlay)
        ax2.set_title(f"Grad-CAM (class {class_idx})")
        ax2.axis("off")
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    return overlay


def _resize_cam(cam: np.ndarray, h: int, w: int) -> np.ndarray:
    import cv2

    resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    return resized
