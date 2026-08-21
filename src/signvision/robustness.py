"""Robustness evaluation under controlled, deterministic input perturbations.

Each perturbation is applied to the *same* held-out test images and evaluated
with the *same* trained checkpoint, so the only thing changing is the input
distribution. Severities are fixed (not sampled), which makes the numbers
repeatable rather than a different random draw on every run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from torch.utils.data import DataLoader

from .data import CachedASLDataset
from .evaluate import compute_metrics


@dataclass
class Perturbation:
    family: str
    severity: str
    fn: Callable[[torch.Tensor], torch.Tensor]

    @property
    def name(self) -> str:
        return f"{self.family}:{self.severity}"


def _rotate(deg: float) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda x: TF.rotate(x, deg)


def _brightness(factor: float) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda x: TF.adjust_brightness(x, factor)


def _blur(sigma: float) -> Callable[[torch.Tensor], torch.Tensor]:
    k = max(3, int(2 * round(3 * sigma) + 1))
    return lambda x: TF.gaussian_blur(x, kernel_size=[k, k], sigma=[sigma, sigma])


def _translate(frac: float) -> Callable[[torch.Tensor], torch.Tensor]:
    def f(x: torch.Tensor) -> torch.Tensor:
        px = int(round(frac * x.shape[-1]))
        return TF.affine(x, angle=0.0, translate=[px, px], scale=1.0, shear=[0.0, 0.0])

    return f


def default_suite() -> list[Perturbation]:
    """Four families x three severities. Deliberately small and interpretable."""
    out: list[Perturbation] = []
    for d in (5, 10, 20):
        out.append(Perturbation("rotation", f"{d}deg", _rotate(float(d))))
    for b in (0.6, 0.8, 1.4):
        out.append(Perturbation("brightness", f"x{b}", _brightness(b)))
    for s in (0.5, 1.0, 2.0):
        out.append(Perturbation("blur", f"sigma{s}", _blur(s)))
    for t in (0.05, 0.10, 0.15):
        out.append(Perturbation("translation", f"{int(t * 100)}pct", _translate(t)))
    return out


class PerturbedDataset(CachedASLDataset):
    """Applies the clean eval transform, then one fixed perturbation."""

    def __init__(self, *args, perturb: Callable[[torch.Tensor], torch.Tensor] | None = None, **kw):
        super().__init__(*args, **kw)
        self.perturb = perturb

    def __getitem__(self, i: int):
        img, label = super().__getitem__(i)
        if self.perturb is not None:
            img = self.perturb(img)
        return img, label


@torch.no_grad()
def evaluate_robustness(
    model: torch.nn.Module,
    cache_path: str,
    test_indices: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    eval_transform,
    device: torch.device,
    suite: list[Perturbation] | None = None,
    batch_size: int = 256,
    verbose: bool = True,
) -> dict:
    """Return clean metrics plus one metric block per perturbation, with deltas."""
    from .evaluate import predict

    suite = suite or default_suite()

    def _metrics(perturb) -> dict:
        ds = PerturbedDataset(cache_path, test_indices, labels, eval_transform, perturb=perturb)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        y, p, _ = predict(model, dl, device)
        return compute_metrics(y, p, classes)

    clean = _metrics(None)
    if verbose:
        print(f"[robust] clean  acc {clean['accuracy']:.4f}  macroF1 {clean['macro_f1']:.4f}")

    rows = []
    for pert in suite:
        m = _metrics(pert.fn)
        row = {
            "family": pert.family,
            "severity": pert.severity,
            "name": pert.name,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "delta_accuracy": m["accuracy"] - clean["accuracy"],
            "delta_macro_f1": m["macro_f1"] - clean["macro_f1"],
            "relative_accuracy_drop": (
                (clean["accuracy"] - m["accuracy"]) / clean["accuracy"]
                if clean["accuracy"] > 0
                else float("nan")
            ),
        }
        rows.append(row)
        if verbose:
            print(f"[robust] {pert.name:22s} acc {m['accuracy']:.4f} "
                  f"({row['delta_accuracy']:+.4f})  macroF1 {m['macro_f1']:.4f} "
                  f"({row['delta_macro_f1']:+.4f})")

    return {"clean": clean, "perturbations": rows}
