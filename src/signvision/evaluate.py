"""Prediction and the single metric definition used by every experiment."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """Return (y_true, y_pred, probs) as numpy arrays."""
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.softmax(logits.float(), dim=1)
        ps.append(logits.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
        ys.append(np.asarray(y))
    return (
        np.concatenate(ys),
        np.concatenate(ps),
        np.concatenate(probs),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    """Accuracy plus macro/weighted P/R/F1 and the full per-class breakdown.

    ``labels`` is pinned to the full class list so a class that never appears in
    a subset still shows up (with zeros) instead of silently shifting the macro
    average — the failure mode that made the original 28-image report misleading.
    """
    labels = list(range(len(classes)))
    acc = float((y_true == y_pred).mean())

    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    p_c, r_c, f_c, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    return {
        "n_samples": int(len(y_true)),
        "accuracy": acc,
        "precision_macro": float(p_ma),
        "recall_macro": float(r_ma),
        "macro_f1": float(f_ma),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "weighted_f1": float(f_w),
        "per_class": {
            classes[i]: {
                "precision": float(p_c[i]),
                "recall": float(r_c[i]),
                "f1": float(f_c[i]),
                "support": int(sup[i]),
            }
            for i in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classes": classes,
    }


def report_text(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> str:
    return classification_report(
        y_true,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
        digits=4,
    )


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(
        f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)
    )
