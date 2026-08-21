#!/usr/bin/env python3
"""Render the recurring confusion pairs side by side so the similarity is visible.

The error analysis reports that the model confuses M with N, G with H, U with R and
V with W. Those class names mean nothing without the handshapes, so this draws real
dataset examples of each pair together.

    python experiments/07_confusion_pairs_figure.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from _common import REPO, VIS

# Pairs that recur across seeds in results*/**/error_analysis.json, with the
# visual distinction that actually separates them.
PAIRS = [
    ("M", "N", "thumb under three fingers vs two"),
    ("G", "H", "one finger extended vs two together"),
    ("U", "R", "two fingers together vs crossed"),
    ("V", "W", "two fingers spread vs three"),
]
N_EX = 3


def main() -> int:
    root = REPO / "data" / "asl_alphabet_train"
    if not root.is_dir():
        print(f"dataset not found at {root}; run scripts/download_data.py")
        return 1

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(PAIRS), N_EX * 2,
                             figsize=(1.9 * N_EX * 2, 2.3 * len(PAIRS)))

    for r, (a, b, why) in enumerate(PAIRS):
        for c, cls in enumerate([a] * N_EX + [b] * N_EX):
            files = sorted((root / cls).glob("*.jpg"))
            f = files[int(rng.integers(0, len(files)))]
            ax = axes[r, c]
            ax.imshow(Image.open(f).convert("RGB").resize((100, 100)))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#4C78A8" if c < N_EX else "#E45756")
                s.set_linewidth(2.5)
            if c == 0:
                ax.set_ylabel(f"{a} vs {b}", fontsize=10, fontweight="bold")
            if c in (0, N_EX):
                ax.set_title(cls, fontsize=11, fontweight="bold",
                             color="#4C78A8" if c == 0 else "#E45756")
        axes[r, -1].text(1.06, 0.5, why, transform=axes[r, -1].transAxes,
                         fontsize=8.5, va="center", ha="left", style="italic")

    fig.suptitle("Recurring confusion pairs — handshapes differing only in finger count or separation",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.84, 0.96))
    out = VIS / "confusion_pairs.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
