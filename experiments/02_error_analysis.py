#!/usr/bin/env python3
"""Quantitative error analysis for a completed run.

    python experiments/02_error_analysis.py --run baseline_seed42
"""

from __future__ import annotations

import argparse

import numpy as np
from _common import CACHE, RESULTS, run_dir, vis_dir

from signvision.config import ExperimentConfig
from signvision.data import prepare
from signvision.error_analysis import analyse
from signvision.plots import error_concentration_plot, image_grid
from signvision.utils import load_json, save_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--n-examples", type=int, default=12)
    a = ap.parse_args()

    res = load_json(RESULTS / a.run / "results.json")
    cfg = ExperimentConfig.from_dict(res["config"])
    cache_path, split = prepare(cfg.data, CACHE, verbose=False)
    classes = split.classes

    y_true = np.asarray(res["predictions"]["y_true"])
    y_pred = np.asarray(res["predictions"]["y_pred"])
    max_prob = np.asarray(res["predictions"]["max_prob"])
    cm = np.asarray(res["test_metrics"]["confusion_matrix"])

    report = analyse(y_true, y_pred, max_prob, cm, classes, top_k=a.top_k)
    out = run_dir(a.run)
    save_json(report, out / "error_analysis.json")

    conc = report["concentration"]
    print(f"\n=== Error analysis: {a.run} ===")
    print(f"test samples {conc['total_samples']}, errors {conc['total_errors']} "
          f"({conc['total_errors'] / max(conc['total_samples'], 1):.2%})")
    for k in (1, 3, 5, 10):
        print(f"  top-{k:<2d} confusion pairs account for "
              f"{conc[f'top_{k}_pairs_error_share']:.1%} of all errors "
              f"({conc[f'top_{k}_pairs_error_count']}/{conc['total_errors']})")
    print("\n  most confused pairs (true -> predicted):")
    for p in report["confusion_pairs"][:a.top_k]:
        print(f"    {p['true']:>7s} -> {p['predicted']:<7s} {p['count']:4d}  "
              f"{p['share_of_all_errors']:6.1%} of errors  "
              f"{p['share_of_class_support']:6.1%} of class support")
    print("\n  lowest-recall classes:")
    for r in report["class_table"][:8]:
        print(f"    {r['class']:>7s}  recall {r['recall']:.3f}  "
              f"errors {r['errors']:3d}/{r['support']:3d}  precision {r['precision']:.3f}")

    v = vis_dir(a.run)
    if report["confusion_pairs"]:
        error_concentration_plot(
            report["confusion_pairs"][:a.top_k], conc["total_errors"],
            v / "error_concentration.png",
        )

    # Representative misclassified and correctly classified images.
    arr = np.load(cache_path, mmap_mode="r")
    test_global = np.asarray(split.test)
    for key, title in (
        ("confident_incorrect", "Most confident misclassifications"),
        ("confident_correct", "Most confident correct predictions"),
    ):
        pos = report["examples"][key][: a.n_examples]
        if not pos:
            continue
        imgs = [np.array(arr[test_global[i]]) for i in pos]
        titles = [
            f"true={classes[y_true[i]]}  pred={classes[y_pred[i]]}\np={max_prob[i]:.2f}"
            for i in pos
        ]
        image_grid(imgs, titles, v / f"examples_{key}.png", suptitle=f"{a.run} — {title}")

    print(f"\n[done] wrote {out / 'error_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
