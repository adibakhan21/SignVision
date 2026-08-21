#!/usr/bin/env python3
"""Cross-arm clean-vs-robust trade-off table.

Answers one question: what does augmentation buy, and what does it cost?
Reads only ``results/*/robustness.json`` written by 03_robustness.py.

    python experiments/06_tradeoff.py
"""

from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict

from _common import RESULTS, VIS

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def arm_of(run: str) -> str:
    return run.rsplit("_seed", 1)[0]


def main() -> int:
    per_arm: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for d in sorted(RESULTS.iterdir()):
        rp = d / "robustness.json"
        if not (d.is_dir() and rp.exists()):
            continue
        rob = json.loads(rp.read_text())
        arm = arm_of(d.name)
        per_arm[arm]["clean"].append(rob["clean"]["accuracy"])
        for p in rob["perturbations"]:
            per_arm[arm][p["name"]].append(p["accuracy"])

    if not per_arm:
        print("no robustness.json files found — run experiments/03_robustness.py first")
        return 1

    cols = ["clean", "rotation:10deg", "rotation:20deg", "translation:5pct",
            "translation:10pct", "blur:sigma2.0", "brightness:x0.6"]

    def fmt(vals: list[float]) -> str:
        if not vals:
            return "—"
        if len(vals) == 1:
            return f"{vals[0]:.4f}"
        return f"{st.mean(vals):.4f} ± {st.stdev(vals):.4f}"

    lines = ["# Clean-vs-robust trade-off", "",
             "Accuracy on the same held-out test set, mean ± sd across seeds.", "",
             "| Arm | Seeds | " + " | ".join(cols) + " |",
             "|---|--:|" + "--:|" * len(cols)]
    rows_csv = []
    for arm in sorted(per_arm):
        vals = per_arm[arm]
        n = len(vals["clean"])
        lines.append(f"| `{arm}` | {n} | " + " | ".join(fmt(vals[c]) for c in cols) + " |")
        rows_csv.append({"arm": arm, "seeds": n,
                         **{c: (st.mean(vals[c]) if vals[c] else None) for c in cols}})

    # Explicit paired deltas for the two augmentation contrasts that exist.
    pairs = [("baseline", "augmented"), ("aslnet_relu", "aslnet_relu_aug"),
             ("baseline_e30", "augmented_e30"), ("aslnet_relu_e30", "aslnet_relu_aug_e30")]
    made_header = False
    for a, b in pairs:
        if a not in per_arm or b not in per_arm:
            continue
        if not made_header:
            lines += ["", "## What augmentation costs and buys (Δ accuracy, percentage points)", "",
                      "| Contrast | " + " | ".join(cols) + " |",
                      "|---|" + "--:|" * len(cols)]
            made_header = True
        deltas = []
        for c in cols:
            if per_arm[a][c] and per_arm[b][c]:
                deltas.append(f"{100 * (st.mean(per_arm[b][c]) - st.mean(per_arm[a][c])):+.1f}")
            else:
                deltas.append("—")
        lines.append(f"| `{b}` − `{a}` | " + " | ".join(deltas) + " |")

    (RESULTS / "TRADEOFF.md").write_text("\n".join(lines) + "\n")
    with open(RESULTS / "tradeoff.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)

    # Clean accuracy vs mean geometric-perturbation accuracy.
    geo = ["rotation:10deg", "rotation:20deg", "translation:5pct", "translation:10pct"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for arm in sorted(per_arm):
        v = per_arm[arm]
        if not v["clean"]:
            continue
        x = st.mean(v["clean"])
        ys = [st.mean(v[g]) for g in geo if v[g]]
        if not ys:
            continue
        y = st.mean(ys)
        aug = "aug" in arm
        ax.scatter(x, y, s=90, marker="^" if aug else "o",
                   color="crimson" if aug else "#4C78A8")
        ax.annotate(arm, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("clean test accuracy")
    ax.set_ylabel("mean accuracy under geometric perturbation")
    ax.set_title("Augmentation trades clean accuracy for geometric robustness\n"
                 "(triangles = augmented, circles = not)")
    ax.grid(alpha=0.3)
    fig.savefig(VIS / "tradeoff.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("\n".join(lines))
    print(f"\n[report] wrote {RESULTS / 'TRADEOFF.md'}, {RESULTS / 'tradeoff.csv'}, "
          f"{VIS / 'tradeoff.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
