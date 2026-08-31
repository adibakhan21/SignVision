#!/usr/bin/env python3
"""Fill the results tables in README.md from the saved experiment JSON.

Every number between a pair of ``<!-- AUTO:x -->`` / ``<!-- /AUTO:x -->`` markers
is generated here, read from ``results/*/``. Nothing in those blocks is typed by
hand, so the README cannot drift from the experiments or carry a placeholder.

    python scripts/make_readme.py
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
README = REPO / "README.md"


def load_runs() -> dict[str, dict]:
    out = {}
    for d in sorted(RESULTS.iterdir()):
        f = d / "results.json"
        if d.is_dir() and f.exists():
            out[d.name] = json.loads(f.read_text())
    return out


def arm_of(run: str) -> str:
    return run.rsplit("_seed", 1)[0]


def ms(vals: list[float], pct: bool = False) -> str:
    if not vals:
        return "—"
    m = st.mean(vals)
    if len(vals) == 1:
        return f"{100 * m:.2f}%" if pct else f"{m:.4f}"
    s = st.stdev(vals)
    return f"{100 * m:.2f}% ± {100 * s:.2f}" if pct else f"{m:.4f} ± {s:.4f}"


LABEL = {
    "baseline": "A · Baseline (original ASLNet, no aug)",
    "augmented": "B · Baseline + augmentation",
    "aslnet_relu": "C · ASLNet-ReLU (FC non-linearity restored)",
    "aslnet_relu_aug": "D · ASLNet-ReLU + augmentation",
    "baseline_e30": "A30 · Baseline, 30 epochs",
    "augmented_e30": "B30 · Baseline + augmentation, 30 epochs",
    "aslnet_relu_e30": "C30 · ASLNet-ReLU, 30 epochs",
    "aslnet_relu_aug_e30": "D30 · ASLNet-ReLU + augmentation, 30 epochs",
    "aslnet_gap_e30": "F30 · ASLNet-GAP (flatten replaced by pooling), 30 epochs",
    "resnet18_ft": "E · ResNet-18 fine-tuned (ImageNet init)",
}
ORDER = list(LABEL)


def block_main(runs) -> str:
    arms = defaultdict(list)
    for name, r in runs.items():
        arms[arm_of(name)].append(r)
    L = ["| Experiment | Epochs | Seeds | Accuracy | Macro F1 | Weighted F1 |",
         "|---|--:|--:|--:|--:|--:|"]
    for arm in [a for a in ORDER if a in arms] + [a for a in sorted(arms) if a not in ORDER]:
        rs = arms[arm]
        ep = rs[0]["config"]["train"]["epochs"]
        L.append(
            f"| {LABEL.get(arm, arm)} | {ep} | {len(rs)} | "
            f"{ms([x['test_metrics']['accuracy'] for x in rs], pct=True)} | "
            f"{ms([x['test_metrics']['macro_f1'] for x in rs])} | "
            f"{ms([x['test_metrics']['weighted_f1'] for x in rs])} |"
        )
    return "\n".join(L)


def block_aug(runs) -> str:
    arms = defaultdict(list)
    for name, r in runs.items():
        arms[arm_of(name)].append(r)
    L = ["| Contrast | Budget | Clean accuracy | Macro F1 | Δ accuracy |", "|---|--:|--:|--:|--:|"]
    for a, b, ep in [("baseline", "augmented", "10 ep"),
                     ("aslnet_relu", "aslnet_relu_aug", "10 ep"),
                     ("baseline_e30", "augmented_e30", "30 ep"),
                     ("aslnet_relu_e30", "aslnet_relu_aug_e30", "30 ep")]:
        if a not in arms or b not in arms:
            continue
        ma = st.mean([x["test_metrics"]["accuracy"] for x in arms[a]])
        mb = st.mean([x["test_metrics"]["accuracy"] for x in arms[b]])
        L.append(f"| {LABEL.get(b, b)} vs {LABEL.get(a, a)} | {ep} | "
                 f"{100 * ma:.2f}% → {100 * mb:.2f}% | "
                 f"{ms([x['test_metrics']['macro_f1'] for x in arms[a]])} → "
                 f"{ms([x['test_metrics']['macro_f1'] for x in arms[b]])} | "
                 f"{100 * (mb - ma):+.1f} pp |")
    return "\n".join(L)


def block_robust() -> str:
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for d in sorted(RESULTS.iterdir()):
        f = d / "robustness.json"
        if not (d.is_dir() and f.exists()):
            continue
        rob = json.loads(f.read_text())
        a = arm_of(d.name)
        per[a]["clean"].append(rob["clean"]["accuracy"])
        for p in rob["perturbations"]:
            per[a][p["name"]].append(p["accuracy"])
    cols = ["clean", "rotation:10deg", "rotation:20deg",
            "translation:5pct", "translation:10pct", "blur:sigma2.0", "brightness:x0.6"]
    head = ["Clean", "Rot 10°", "Rot 20°", "Shift 5%", "Shift 10%", "Blur σ2", "Bright ×0.6"]
    L = ["| Arm | Seeds | " + " | ".join(head) + " |", "|---|--:|" + "--:|" * len(head)]
    for arm in [a for a in ORDER if a in per] + [a for a in sorted(per) if a not in ORDER]:
        v = per[arm]
        L.append(f"| {LABEL.get(arm, arm)} | {len(v['clean'])} | "
                 + " | ".join(ms(v[c], pct=True) for c in cols) + " |")
    return "\n".join(L)


def block_tradeoff() -> str:
    per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for d in sorted(RESULTS.iterdir()):
        f = d / "robustness.json"
        if not (d.is_dir() and f.exists()):
            continue
        rob = json.loads(f.read_text())
        a = arm_of(d.name)
        per[a]["clean"].append(rob["clean"]["accuracy"])
        for p in rob["perturbations"]:
            per[a][p["name"]].append(p["accuracy"])
    cols = ["clean", "rotation:10deg", "rotation:20deg",
            "translation:5pct", "translation:10pct", "blur:sigma2.0"]
    head = ["Clean", "Rot 10°", "Rot 20°", "Shift 5%", "Shift 10%", "Blur σ2"]
    L = ["| Contrast | " + " | ".join(head) + " |", "|---|" + "--:|" * len(head)]
    for a, b in [("baseline", "augmented"), ("aslnet_relu", "aslnet_relu_aug"),
                 ("baseline_e30", "augmented_e30"),
                 ("aslnet_relu_e30", "aslnet_relu_aug_e30")]:
        if a not in per or b not in per:
            continue
        ds = []
        for c in cols:
            if per[a][c] and per[b][c]:
                ds.append(f"{100 * (st.mean(per[b][c]) - st.mean(per[a][c])):+.1f}")
            else:
                ds.append("—")
        ds[0] = f"**{ds[0]}**"
        L.append(f"| {LABEL.get(b, b)}<br>− {LABEL.get(a, a)} | " + " | ".join(ds) + " |")
    return "\n".join(L)


def block_errors(runs) -> str:
    best = max(runs.items(), key=lambda kv: kv[1]["test_metrics"]["accuracy"])[0]
    f = RESULTS / best / "error_analysis.json"
    if not f.exists():
        return "_error analysis not generated_"
    ea = json.loads(f.read_text())
    c = ea["concentration"]
    L = [f"Best run: `{best}` — {c['total_errors']} errors on {c['total_samples']} test images "
         f"({c['total_errors'] / c['total_samples']:.2%} error rate).", "",
         "| Rank | True → Predicted | Errors | % of all errors | % of that class |",
         "|--:|---|--:|--:|--:|"]
    for i, p in enumerate(ea["confusion_pairs"][:8], 1):
        L.append(f"| {i} | {p['true']} → {p['predicted']} | {p['count']} | "
                 f"{p['share_of_all_errors']:.1%} | {p['share_of_class_support']:.1%} |")
    L += ["", "| Concentration | Share of all errors |", "|---|--:|"]
    for k in (1, 3, 5, 10):
        L.append(f"| Top-{k} confusion pairs | **{c[f'top_{k}_pairs_error_share']:.1%}** "
                 f"({c[f'top_{k}_pairs_error_count']}/{c['total_errors']}) |")
    L += ["", "Lowest-recall classes:", "",
          "| Class | Support | Recall | Precision |", "|---|--:|--:|--:|"]
    for r in ea["class_table"][:6]:
        L.append(f"| {r['class']} | {r['support']} | {r['recall']:.3f} | {r['precision']:.3f} |")
    return "\n".join(L)


def block_setup(runs) -> str:
    r = next(iter(runs.values()))
    d, t = r["config"]["data"], r["config"]["train"]
    n_tr, n_va, n_te = (r["split_sizes"][k] for k in ("train", "val", "test"))
    return "\n".join([
        "| Item | Value |", "|---|---|",
        f"| Classes | {len(r['test_metrics']['classes'])} (A–Z, `del`, `nothing`, `space`) |",
        f"| Images used | {n_tr + n_va + n_te:,} ({d['images_per_class']}/class, "
        f"stratified from 87,000) |",
        f"| Split | {n_tr:,} train / {n_va:,} val / {n_te:,} test "
        f"({d['train_frac']:.0%}/{d['val_frac']:.0%}/{d['test_frac']:.0%}, "
        f"seed {d['split_seed']}) |",
        f"| Input | {d['image_size']}×{d['image_size']} RGB, scaled to [0,1] |",
        f"| Optimiser | {t['optimizer'].title()}, lr {t['lr']}, batch {t['batch_size']} |",
        f"| Model selection | best epoch by `{t['select_on']}` on the validation set |",
        f"| Parameters | {r['params']['total']:,} |",
    ])


def main() -> int:
    runs = load_runs()
    if not runs:
        print("no runs found", file=sys.stderr)
        return 1
    blocks = {
        "setup": block_setup(runs),
        "main": block_main(runs),
        "aug": block_aug(runs),
        "robust": block_robust(),
        "tradeoff": block_tradeoff(),
        "errors": block_errors(runs),
    }
    text = README.read_text()
    for key, body in blocks.items():
        pat = re.compile(rf"(<!-- AUTO:{key} -->)(.*?)(<!-- /AUTO:{key} -->)", re.S)
        if not pat.search(text):
            print(f"warning: no marker for '{key}' in README", file=sys.stderr)
            continue
        text = pat.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)
    README.write_text(text)
    print(f"[readme] filled {len(blocks)} blocks from {len(runs)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
