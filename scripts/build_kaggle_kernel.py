#!/usr/bin/env python3
"""Generate a self-contained Kaggle notebook that runs the full experiment suite.

The notebook embeds the current contents of ``src/signvision`` so the Kaggle run
executes exactly the code in this repository — no drift between local and remote.

    python scripts/build_kaggle_kernel.py
    kaggle kernels push -p kaggle
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "signvision"
OUT = REPO / "kaggle"
USERNAME = "adibakhan23"
SLUG = "signvision-experiments"

MODULES = ["__init__", "utils", "config", "data", "models", "evaluate",
           "train", "robustness", "error_analysis", "interpretability", "plots"]


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": src.splitlines(keepends=True),
            "execution_count": None, "outputs": []}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def build() -> None:
    OUT.mkdir(exist_ok=True)
    cells = [md("# SignVision — full experiment suite\n\n"
                "Runs every arm at 10 and 30 epochs, 3 seeds each, plus the robustness sweep\n"
                "and error analysis. Generated from the local repo by\n"
                "`scripts/build_kaggle_kernel.py` — do not edit here.\n")]

    writer = ["import pathlib, sys",
              "PKG = pathlib.Path('/kaggle/working/src/signvision')",
              "PKG.mkdir(parents=True, exist_ok=True)",
              "FILES = {}"]
    for m in MODULES:
        body = (SRC / f"{m}.py").read_text()
        assert "'''" not in body, f"{m}.py contains a triple quote that breaks embedding"
        writer.append(f"FILES[{m!r}] = r'''{body}'''")
    writer += ["for name, body in FILES.items():",
               "    (PKG / f'{name}.py').write_text(body)",
               "sys.path.insert(0, '/kaggle/working/src')",
               "print('wrote', len(FILES), 'modules')"]
    cells.append(code("\n".join(writer)))

    cells.append(code('''import pathlib

def find_root(base="/kaggle/input"):
    """Locate the ImageFolder root by content, not by a guessed mount path.

    Kaggle's mount layout varies between datasets and over time, so search for the
    directory whose immediate children are the 29 ASL class folders.
    """
    want = {"A", "B", "C", "nothing", "space", "del"}
    best = None
    for d in pathlib.Path(base).rglob("*"):
        if not d.is_dir():
            continue
        names = {c.name for c in d.iterdir() if c.is_dir()}
        if want <= names:
            n = len(names)
            if best is None or n > best[0]:
                best = (n, d)
    if best is None:
        listing = sorted(str(x) for x in pathlib.Path(base).rglob("*") if x.is_dir())[:40]
        raise FileNotFoundError("no ASL ImageFolder root under %s; saw %s" % (base, listing))
    return best[1]

ROOT = find_root()
classes = sorted(d.name for d in ROOT.iterdir() if d.is_dir())
n = sum(len(list((ROOT / c).glob("*.jpg"))) for c in classes)
print("root:", ROOT)
print(len(classes), "classes,", n, "images")'''))

    cells.append(code('''from signvision.config import ExperimentConfig

ARMS = [("baseline", "aslnet_original", False), ("augmented", "aslnet_original", True),
        ("aslnet_relu", "aslnet_relu", False), ("aslnet_relu_aug", "aslnet_relu", True)]
CFG = {}
for epochs in (10, 30):
    for base, model, aug in ARMS:
        name = base if epochs == 10 else base + "_e30"
        c = ExperimentConfig(name=name)
        c.data.root = str(ROOT)
        c.train.model, c.train.epochs = model, epochs
        c.aug.enabled = aug
        c.train.num_workers = 4
        CFG[name] = c
print(len(CFG), "arms:", sorted(CFG))'''))

    cells.append(code('''import torch, time
from signvision.data import prepare
from signvision.utils import get_device

def working_device():
    """Return a device that can actually run our ops.

    Kaggle sometimes allocates a P100 (sm_60), for which the preinstalled torch
    build ships no kernels. Probing costs seconds; discovering it after the cache
    build costs minutes.
    """
    d = get_device()
    if d.type != "cuda":
        return d
    try:
        m = torch.nn.Conv2d(3, 4, 3).to(d)
        m(torch.zeros(1, 3, 16, 16, device=d)).sum().backward()
        torch.cuda.synchronize()
        print("GPU OK:", torch.cuda.get_device_name(0),
              "sm_%d%d" % torch.cuda.get_device_capability(0))
        return d
    except Exception as e:
        # Deliberately fatal. A silent CPU fallback turned a 20-minute job into a
        # 12-hour one that could not finish, and Kaggle hides logs until a run
        # ends, so the degradation was invisible while it burned quota.
        msg = ("GPU %s (sm_%d%d) cannot run this build of torch: %s"
               % (torch.cuda.get_device_name(0), *torch.cuda.get_device_capability(0), e))
        print(msg)
        print("Re-push with: kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4")
        raise RuntimeError(msg)

DEVICE = working_device()
for c in CFG.values():
    c.train.device = str(DEVICE)
print("device:", DEVICE)
t0 = time.time()
CACHE, SPLIT = prepare(list(CFG.values())[0].data, "/tmp/cache")
print("cache built in %.0fs | split %s" % (time.time() - t0, SPLIT.sizes()))'''))

    cells.append(code('''import numpy as np
from signvision.train import run_training
from signvision.evaluate import report_text
from signvision.utils import save_json

RESULTS = pathlib.Path("/kaggle/working/results"); RESULTS.mkdir(exist_ok=True)
CKPT = pathlib.Path("/kaggle/working/checkpoints"); CKPT.mkdir(exist_ok=True)
SEEDS = [42, 43, 44]

for name, base_cfg in CFG.items():
    for seed in SEEDS:
        run = "%s_seed%d" % (name, seed)
        out = RESULTS / run
        if (out / "results.json").exists():
            print("skip", run); continue
        cfg = ExperimentConfig.from_dict(base_cfg.to_dict()); cfg.train.seed = seed
        res = run_training(cfg, CACHE, SPLIT, CKPT / (run + ".pt"), verbose=False)
        res["split_sizes"] = SPLIT.sizes()
        out.mkdir(parents=True, exist_ok=True)
        save_json(res, out / "results.json"); cfg.save(out / "config.json")
        (out / "classification_report.txt").write_text(report_text(
            np.asarray(res["predictions"]["y_true"]),
            np.asarray(res["predictions"]["y_pred"]), SPLIT.classes))
        t = res["test_metrics"]
        print("%-32s acc %.4f  macroF1 %.4f  (%.0fs)" % (
            run, t["accuracy"], t["macro_f1"], res["wall_seconds"]))
print("TRAINING COMPLETE")'''))

    cells.append(code('''import csv
from signvision.data import make_transforms
from signvision.models import build_model
from signvision.robustness import evaluate_robustness
from signvision.error_analysis import analyse
from signvision.utils import load_json

device = DEVICE
for out in sorted(RESULTS.iterdir()):
    res = load_json(out / "results.json")
    cfg = ExperimentConfig.from_dict(res["config"])
    run = out.name
    if not (out / "robustness.json").exists():
        model = build_model(cfg.train.model, len(SPLIT.classes), cfg.data.image_size).to(device)
        model.load_state_dict(torch.load(CKPT / (run + ".pt"), map_location=device,
                                         weights_only=False)["model_state"])
        rob = evaluate_robustness(model, str(CACHE), np.asarray(SPLIT.test), SPLIT.labels,
                                  SPLIT.classes, make_transforms(cfg.aug, train=False),
                                  device, verbose=False)
        save_json(rob, out / "robustness.json")
        with open(out / "robustness.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rob["perturbations"][0].keys()))
            w.writeheader(); w.writerows(rob["perturbations"])
    if not (out / "error_analysis.json").exists():
        p = res["predictions"]
        save_json(analyse(np.asarray(p["y_true"]), np.asarray(p["y_pred"]),
                          np.asarray(p["max_prob"]),
                          np.asarray(res["test_metrics"]["confusion_matrix"]),
                          SPLIT.classes), out / "error_analysis.json")
    print("analysed", run)
print("ANALYSIS COMPLETE")'''))

    cells.append(code('''import statistics as st
from collections import defaultdict
arms = defaultdict(list)
for out in sorted(RESULTS.iterdir()):
    r = load_json(out / "results.json")
    arms[out.name.rsplit("_seed", 1)[0]].append(r["test_metrics"])
def ms(v):
    return "%.4f +/- %.4f" % (st.mean(v), st.stdev(v)) if len(v) > 1 else "%.4f" % v[0]
print("%-26s %2s %18s %18s" % ("arm", "n", "accuracy", "macro F1"))
for a in sorted(arms):
    print("%-26s %2d %18s %18s" % (a, len(arms[a]),
          ms([m["accuracy"] for m in arms[a]]), ms([m["macro_f1"] for m in arms[a]])))'''))

    cells.append(code('''import shutil
shutil.rmtree("/kaggle/working/src", ignore_errors=True)
print("kept:", sorted(p.name for p in pathlib.Path("/kaggle/working").iterdir()))'''))

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        try:
            compile("".join(c["source"]), "cell%d" % i, "exec")
        except SyntaxError as e:
            raise SystemExit("cell %d has a syntax error on line %s: %s\n  %s"
                             % (i, e.lineno, e.msg, e.text))
    print("all %d code cells compile" % sum(c["cell_type"] == "code" for c in cells))
    (OUT / (SLUG + ".ipynb")).write_text(json.dumps(nb, indent=1))

    meta = {"id": USERNAME + "/" + SLUG,
            "title": "SignVision Experiments",
            "code_file": SLUG + ".ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "accelerator": "NvidiaTeslaT4",
            "enable_internet": False,
            "dataset_sources": ["grassknoted/asl-alphabet"],
            "competition_sources": [],
            "kernel_sources": []}
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote %s (%d cells)" % (OUT / (SLUG + ".ipynb"), len(cells)))
    print("private=%s gpu=%s internet=%s" % (meta["is_private"], meta["enable_gpu"],
                                             meta["enable_internet"]))


if __name__ == "__main__":
    build()
