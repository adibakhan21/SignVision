"""Quantitative error analysis over a confusion matrix and raw predictions."""

from __future__ import annotations

import numpy as np


def confusion_pairs(cm: np.ndarray, classes: list[str], top_k: int = 10) -> list[dict]:
    """Ranked off-diagonal (true -> predicted) pairs with their share of all errors."""
    cm = np.asarray(cm)
    total_errors = int(cm.sum() - np.trace(cm))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1], off.shape))[0]

    out = []
    for i, j in order[:top_k]:
        n = int(off[i, j])
        if n == 0:
            break
        out.append(
            {
                "true": classes[i],
                "predicted": classes[j],
                "count": n,
                "share_of_all_errors": n / total_errors if total_errors else 0.0,
                "share_of_class_support": n / int(cm[i].sum()) if cm[i].sum() else 0.0,
            }
        )
    return out


def class_error_table(cm: np.ndarray, classes: list[str]) -> list[dict]:
    """Per-class support, recall, error count and error rate, sorted worst first."""
    cm = np.asarray(cm)
    rows = []
    for i, c in enumerate(classes):
        support = int(cm[i].sum())
        correct = int(cm[i, i])
        errors = support - correct
        rows.append(
            {
                "class": c,
                "support": support,
                "correct": correct,
                "errors": errors,
                "recall": correct / support if support else 0.0,
                "error_rate": errors / support if support else 0.0,
                "predicted_as": int(cm[:, i].sum()),
                "precision": correct / int(cm[:, i].sum()) if cm[:, i].sum() else 0.0,
            }
        )
    rows.sort(key=lambda r: (r["recall"], -r["errors"]))
    return rows


def error_concentration(cm: np.ndarray, classes: list[str], ks=(1, 3, 5, 10)) -> dict:
    """What fraction of all errors the top-k confusion pairs account for."""
    pairs = confusion_pairs(cm, classes, top_k=max(ks))
    cm = np.asarray(cm)
    total_errors = int(cm.sum() - np.trace(cm))
    cum = np.cumsum([p["count"] for p in pairs]) if pairs else np.array([0])
    out = {"total_errors": total_errors, "total_samples": int(cm.sum())}
    for k in ks:
        n = int(cum[min(k, len(cum)) - 1]) if len(pairs) else 0
        out[f"top_{k}_pairs_error_count"] = n
        out[f"top_{k}_pairs_error_share"] = n / total_errors if total_errors else 0.0
    return out


def example_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_prob: np.ndarray,
    n: int = 12,
) -> dict[str, list[int]]:
    """Positions (within the test set) of representative correct / incorrect cases.

    Incorrect examples are sorted by descending confidence: the model's *confident*
    mistakes are the diagnostically interesting ones.
    """
    y_true, y_pred, max_prob = map(np.asarray, (y_true, y_pred, max_prob))
    wrong = np.flatnonzero(y_true != y_pred)
    right = np.flatnonzero(y_true == y_pred)
    wrong_sorted = wrong[np.argsort(-max_prob[wrong])] if len(wrong) else wrong
    right_sorted = right[np.argsort(-max_prob[right])] if len(right) else right
    return {
        "confident_incorrect": wrong_sorted[:n].tolist(),
        "least_confident_incorrect": wrong_sorted[-n:][::-1].tolist() if len(wrong) else [],
        "confident_correct": right_sorted[:n].tolist(),
        "least_confident_correct": right_sorted[-n:][::-1].tolist() if len(right) else [],
    }


def analyse(y_true, y_pred, max_prob, cm, classes: list[str], top_k: int = 10) -> dict:
    return {
        "confusion_pairs": confusion_pairs(cm, classes, top_k=top_k),
        "class_table": class_error_table(cm, classes),
        "concentration": error_concentration(cm, classes),
        "examples": example_indices(y_true, y_pred, max_prob),
    }
