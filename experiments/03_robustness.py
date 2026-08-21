#!/usr/bin/env python3
"""Robustness of a trained checkpoint under controlled input perturbations.

    python experiments/03_robustness.py --run baseline_seed42
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
from _common import CACHE, CKPT, RESULTS, run_dir, vis_dir

from signvision.config import ExperimentConfig
from signvision.data import make_transforms, prepare
from signvision.models import build_model
from signvision.plots import robustness_plot
from signvision.robustness import evaluate_robustness
from signvision.utils import get_device, load_json, save_json, set_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()

    res = load_json(RESULTS / a.run / "results.json")
    cfg = ExperimentConfig.from_dict(res["config"])
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)

    cache_path, split = prepare(cfg.data, CACHE, verbose=False)
    model = build_model(cfg.train.model, len(split.classes), cfg.data.image_size).to(device)
    state = torch.load(CKPT / f"{a.run}.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])

    print(f"=== Robustness: {a.run} ===")
    out = evaluate_robustness(
        model=model,
        cache_path=str(cache_path),
        test_indices=np.asarray(split.test),
        labels=split.labels,
        classes=split.classes,
        eval_transform=make_transforms(cfg.aug, train=False),
        device=device,
    )

    d = run_dir(a.run)
    save_json(out, d / "robustness.json")
    with open(d / "robustness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out["perturbations"][0].keys()))
        w.writeheader()
        w.writerows(out["perturbations"])

    robustness_plot(
        out["perturbations"],
        out["clean"]["accuracy"],
        out["clean"]["macro_f1"],
        vis_dir(a.run) / "robustness.png",
    )
    print(f"[done] wrote {d / 'robustness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
