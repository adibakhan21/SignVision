"""All figure generation. Every plot is written from a saved metrics file."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"[plot] {path}")
    return path


def confusion_matrix_plot(cm, classes, path, title="Confusion matrix", normalize=False):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            cm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="BuGn", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(classes)), classes, rotation=90)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.grid(False)
    thresh = cm.max() / 2 if cm.max() else 0.5
    fmt = "{:.2f}" if normalize else "{:.0f}"
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] > 0:
                ax.text(
                    j, i, fmt.format(cm[i, j]), ha="center", va="center", fontsize=5.5,
                    color="white" if cm[i, j] > thresh else "black",
                )
    return _save(fig, path)


def training_curves(histories: dict[str, list[dict]], path, title="Training curves"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, h in histories.items():
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["train_loss"] for r in h], marker="o", ms=3, label=name)
        axes[1].plot(ep, [r["train_acc"] for r in h], marker="o", ms=3, label=name)
        axes[1].plot(ep, [r["val_acc"] for r in h], marker="s", ms=3, ls="--", label=f"{name} (val)")
        axes[2].plot(ep, [r["val_macro_f1"] for r in h], marker="o", ms=3, label=name)
    axes[0].set(xlabel="epoch", ylabel="train loss", title="Training loss")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy (solid=train, dashed=val)")
    axes[2].set(xlabel="epoch", ylabel="val macro F1", title="Validation macro F1")
    for a in axes:
        a.legend(fontsize=7)
    fig.suptitle(title)
    return _save(fig, path)


def per_class_recall_plot(per_class_by_arm: dict[str, dict], classes, path):
    fig, ax = plt.subplots(figsize=(13, 4.2))
    n = len(per_class_by_arm)
    w = 0.8 / n
    x = np.arange(len(classes))
    for k, (name, pc) in enumerate(per_class_by_arm.items()):
        vals = [pc[c]["recall"] for c in classes]
        ax.bar(x + k * w - 0.4 + w / 2, vals, width=w, label=name)
    ax.set_xticks(x, classes, rotation=90)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("recall")
    ax.set_title("Per-class recall on the held-out test set")
    ax.legend(fontsize=8)
    return _save(fig, path)


def robustness_plot(rows: list[dict], clean_acc: float, clean_f1: float, path):
    fams: dict[str, list[dict]] = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r)
    fig, axes = plt.subplots(1, len(fams), figsize=(4 * len(fams), 3.8), sharey=True)
    if len(fams) == 1:
        axes = [axes]
    for ax, (fam, rs) in zip(axes, fams.items()):
        labels = [r["severity"] for r in rs]
        ax.plot(labels, [r["accuracy"] for r in rs], marker="o", label="accuracy")
        ax.plot(labels, [r["macro_f1"] for r in rs], marker="s", label="macro F1")
        ax.axhline(clean_acc, color="gray", ls="--", lw=1, label="clean acc")
        ax.set_title(fam)
        ax.set_ylim(0, 1.02)
        ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("score")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Test performance under controlled input perturbations")
    return _save(fig, path)


def image_grid(images, titles, path, suptitle="", cols=6, cmap=None):
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i], cmap=cmap)
            ax.set_title(titles[i], fontsize=7)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def gradcam_panel(entries: list[dict], path, suptitle=""):
    """One row per example: original | heatmap | overlay."""
    n = len(entries)
    fig, axes = plt.subplots(n, 3, figsize=(6.6, 2.3 * n))
    axes = np.atleast_2d(axes)
    for r, e in enumerate(entries):
        img, heat = e["image"], e["heatmap"]
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"true={e['true']}  pred={e['pred']}\np={e['prob']:.2f}", fontsize=7.5,
                             color=("green" if e["true"] == e["pred"] else "crimson"))
        axes[r, 1].imshow(heat, cmap="jet")
        axes[r, 1].set_title("Grad-CAM", fontsize=7.5)
        axes[r, 2].imshow(img)
        axes[r, 2].imshow(heat, cmap="jet", alpha=0.45)
        axes[r, 2].set_title("overlay", fontsize=7.5)
        for c in range(3):
            axes[r, c].axis("off")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def error_concentration_plot(pairs: list[dict], total_errors: int, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{p['true']}→{p['predicted']}" for p in pairs]
    counts = [p["count"] for p in pairs]
    cum = np.cumsum(counts) / total_errors if total_errors else np.zeros(len(counts))
    ax.bar(labels, counts, color="#4C78A8")
    ax.set_ylabel("errors")
    ax.tick_params(axis="x", rotation=45)
    ax2 = ax.twinx()
    ax2.plot(labels, cum, color="crimson", marker="o", ms=4)
    ax2.set_ylabel("cumulative share of all errors", color="crimson")
    ax2.set_ylim(0, 1.02)
    ax2.grid(False)
    ax.set_title(f"Top confusion pairs (total errors = {total_errors})")
    return _save(fig, path)
