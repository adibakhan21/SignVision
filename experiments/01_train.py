#!/usr/bin/env python3
"""Train one experimental arm and write its metrics, checkpoint and curves.

Every arm goes through this same script; the only thing that differs is the
config file, which is what makes the comparisons controlled.

    python experiments/01_train.py --config configs/baseline.json
    python experiments/01_train.py --config configs/augmented.json --seed 1
"""

from __future__ import annotations

import argparse

from _common import CACHE, CKPT, run_dir, vis_dir

from signvision.config import ExperimentConfig
from signvision.data import prepare
from signvision.evaluate import report_text
from signvision.plots import confusion_matrix_plot, training_curves
from signvision.train import run_training
from signvision.utils import save_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="override config train.seed")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--tag", default=None, help="override run name")
    a = ap.parse_args()

    cfg = ExperimentConfig.load(a.config)
    if a.seed is not None:
        cfg.train.seed = a.seed
    if a.epochs is not None:
        cfg.train.epochs = a.epochs
    run_name = a.tag or f"{cfg.name}_seed{cfg.train.seed}"

    cache_path, split = prepare(cfg.data, CACHE)
    out = run_dir(run_name)
    res = run_training(cfg, cache_path, split, CKPT / f"{run_name}.pt")

    res["split_sizes"] = split.sizes()
    save_json(res, out / "results.json")
    cfg.save(out / "config.json")
    (out / "classification_report.txt").write_text(
        report_text(
            __import__("numpy").asarray(res["predictions"]["y_true"]),
            __import__("numpy").asarray(res["predictions"]["y_pred"]),
            split.classes,
        )
    )

    v = vis_dir(run_name)
    training_curves({run_name: res["history"]}, v / "training_curves.png", title=run_name)
    confusion_matrix_plot(
        res["test_metrics"]["confusion_matrix"],
        split.classes,
        v / "confusion_matrix.png",
        title=f"{run_name} — test confusion matrix",
    )
    confusion_matrix_plot(
        res["test_metrics"]["confusion_matrix"],
        split.classes,
        v / "confusion_matrix_normalized.png",
        title=f"{run_name} — test confusion matrix (row-normalised)",
        normalize=True,
    )
    print(f"\n[done] {run_name}: "
          f"test acc {res['test_metrics']['accuracy']:.4f} | "
          f"macro F1 {res['test_metrics']['macro_f1']:.4f} | "
          f"weighted F1 {res['test_metrics']['weighted_f1']:.4f}")
    print(f"[done] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
