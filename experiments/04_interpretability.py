#!/usr/bin/env python3
"""Grad-CAM panels and convolutional activation maps for a trained run.

    python experiments/04_interpretability.py --run baseline_seed42
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from _common import CACHE, CKPT, RESULTS, run_dir, vis_dir

from signvision.config import ExperimentConfig
from signvision.data import make_transforms, prepare
from signvision.interpretability import GradCAM, activation_maps
from signvision.models import build_model
from signvision.plots import gradcam_panel, image_grid
from signvision.utils import get_device, load_json, save_json, set_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=5, help="examples per panel")
    a = ap.parse_args()

    res = load_json(RESULTS / a.run / "results.json")
    cfg = ExperimentConfig.from_dict(res["config"])
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)

    cache_path, split = prepare(cfg.data, CACHE, verbose=False)
    classes = split.classes
    model = build_model(cfg.train.model, len(classes), cfg.data.image_size).to(device)
    state = torch.load(CKPT / f"{a.run}.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    y_true = np.asarray(res["predictions"]["y_true"])
    y_pred = np.asarray(res["predictions"]["y_pred"])
    max_prob = np.asarray(res["predictions"]["max_prob"])
    test_global = np.asarray(split.test)

    arr = np.load(cache_path, mmap_mode="r")
    tf = make_transforms(cfg.aug, train=False)

    def tensor_for(pos: int) -> torch.Tensor:
        img = torch.from_numpy(np.array(arr[test_global[pos]])).permute(2, 0, 1)
        return tf(img).unsqueeze(0).to(device)

    layer = getattr(model, "gradcam_layer")
    layer_name = "c2 (second conv layer, 27 channels)" if cfg.train.model.startswith("aslnet") \
        else "layer4[-1].conv2"

    wrong = np.flatnonzero(y_true != y_pred)
    right = np.flatnonzero(y_true == y_pred)
    wrong = wrong[np.argsort(-max_prob[wrong])][: a.n] if len(wrong) else wrong
    right = right[np.argsort(-max_prob[right])][: a.n]

    manifest = {"gradcam_layer": layer_name, "panels": {}}
    for key, sel, title in (
        ("correct", right, "Grad-CAM — correctly classified (most confident)"),
        ("incorrect", wrong, "Grad-CAM — misclassified (most confident errors)"),
    ):
        if len(sel) == 0:
            print(f"[interp] no {key} examples to plot")
            continue
        entries = []
        for pos in sel:
            x = tensor_for(int(pos))
            with GradCAM(model, layer) as cam:
                heat, used = cam(x, int(y_pred[pos]))
            entries.append(
                {
                    "image": np.array(arr[test_global[pos]]),
                    "heatmap": heat,
                    "true": classes[y_true[pos]],
                    "pred": classes[y_pred[pos]],
                    "prob": float(max_prob[pos]),
                }
            )
        p = gradcam_panel(entries, vis_dir(a.run) / f"gradcam_{key}.png",
                          suptitle=f"{a.run} — {title}\ntarget layer: {layer_name}")
        manifest["panels"][key] = str(p)

    # Activation maps for the same layer on one representative correct example.
    pos = int(right[0]) if len(right) else 0
    acts = activation_maps(model, tensor_for(pos), layer)
    manifest["activation_maps"] = {
        "layer": layer_name,
        "n_channels": int(acts.shape[0]),
        "spatial": list(acts.shape[1:]),
        "example_true": classes[y_true[pos]],
        "example_pred": classes[y_pred[pos]],
    }
    image_grid(
        [acts[i] for i in range(acts.shape[0])],
        [f"ch {i}" for i in range(acts.shape[0])],
        vis_dir(a.run) / "activation_maps.png",
        suptitle=(f"{a.run} — {acts.shape[0]} activation maps from {layer_name}\n"
                  f"input: true={classes[y_true[pos]]}, pred={classes[y_pred[pos]]}"),
        cols=6,
        cmap="viridis",
    )
    save_json(manifest, run_dir(a.run) / "interpretability.json")
    print(f"[interp] {acts.shape[0]} activation maps ({acts.shape[1]}x{acts.shape[2]}) from {layer_name}")
    print(f"[done] wrote {run_dir(a.run) / 'interpretability.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
