"""Grad-CAM and convolutional activation maps.

This is an upgraded version of the Grad-CAM in the original notebook. Three
things are fixed there: the heatmap is now bilinearly resized to the input
resolution before overlay (the original drew a 20x20 map on top of a 100x100
image, so the two were never spatially aligned), hooks are removed after use
instead of leaking, and the class activations are no longer mutated in place.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Gradient-weighted class activation mapping on a chosen conv layer.

    Usage::

        with GradCAM(model, model.gradcam_layer) as cam:
            heat = cam(x, class_idx)     # (H, W) in [0, 1], input resolution
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list = []

    def __enter__(self) -> "GradCAM":
        self._handles.append(self.layer.register_forward_hook(self._save_activation))
        self._handles.append(self.layer.register_full_backward_hook(self._save_gradient))
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _save_activation(self, module, inp, out) -> None:
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out) -> None:
        self.gradients = grad_out[0]

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, int]:
        """Return (heatmap at input resolution, class index actually used)."""
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        logits[0, class_idx].backward()

        assert self.activations is not None and self.gradients is not None
        acts = self.activations[0].detach()          # (C, h, w)
        grads = self.gradients[0].detach()           # (C, h, w)
        weights = grads.mean(dim=(1, 2))             # global-average-pooled gradients
        cam = (weights[:, None, None] * acts).sum(0)  # (h, w)
        cam = F.relu(cam)

        # Align to the input grid so the overlay is spatially meaningful.
        cam = F.interpolate(
            cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        cam = cam - cam.min()
        denom = cam.max()
        if denom > 0:
            cam = cam / denom
        return cam.cpu().numpy(), class_idx


@torch.no_grad()
def activation_maps(
    model: torch.nn.Module, x: torch.Tensor, layer: torch.nn.Module
) -> np.ndarray:
    """Return the (C, h, w) feature maps produced by ``layer`` for a single input."""
    store: dict[str, torch.Tensor] = {}

    def hook(module, inp, out):
        store["act"] = out.detach().cpu()

    h = layer.register_forward_hook(hook)
    try:
        model.eval()
        _ = model(x)
    finally:
        h.remove()
    return store["act"][0].numpy()
