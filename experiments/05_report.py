#!/usr/bin/env python3
"""Aggregate every completed run into results/RESULTS.md and results/summary.csv.

Reads only files written by the other experiment scripts, so the report can
never contain a number that was not produced by a run.

    python experiments/05_report.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from _common import REPO, RESULTS, VIS

from signvision.plots import per_class_recall_plot, training_curves


def load_runs() -> dict[str, dict]:
    runs = {}
    for d in sorted(RESULTS.iterdir()):
        f = d / "results.json"
        if d.is_dir() and f.exists():
            runs[d.name] = json.loads(f.read_text())
    return runs


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> int:
    runs = load_runs()
    if not runs:
        print("no runs found under results/")
        return 1

    rows = []
    for name, r in runs.items():
        t, v = r["test_metrics"], r["val_metrics"]
        rows.append(
            {
                "run": name,
                "model": r["config"]["train"]["model"],
                "augmentation": r["config"]["aug"]["enabled"],
                "seed": r["config"]["train"]["seed"],
                "params": r["params"]["total"],
                "best_epoch": r["best_epoch"],
                "val_accuracy": v["accuracy"],
                "val_macro_f1": v["macro_f1"],
                "test_accuracy": t["accuracy"],
                "test_precision_macro": t["precision_macro"],
                "test_recall_macro": t["recall_macro"],
                "test_macro_f1": t["macro_f1"],
                "test_weighted_f1": t["weighted_f1"],
                "test_n": t["n_samples"],
                "wall_seconds": round(r["wall_seconds"], 1),
            }
        )
    rows.sort(key=lambda r: (r["model"], r["augmentation"], r["seed"]))

    with open(RESULTS / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Group seeds of the same arm so mean +/- sd is reported honestly.
    arms: dict[str, list[dict]] = {}
    for r in rows:
        arms.setdefault(f"{r['model']}|aug={r['augmentation']}", []).append(r)

    L: list[str] = ["# SignVision — verified experimental results", ""]
    L.append("Every number below was written by a script in `experiments/` and read back from")
    L.append("`results/*/results.json`. Regenerate with `python experiments/05_report.py`.")
    L.append("")
    L.append("## Per-run results (held-out test set)")
    L.append("")
    L.append("| Run | Model | Aug | Seed | Params | Best ep | Test acc | Macro F1 | Weighted F1 | Macro P | Macro R |")
    L.append("|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        L.append(
            f"| `{r['run']}` | {r['model']} | {'yes' if r['augmentation'] else 'no'} | "
            f"{r['seed']} | {r['params']:,} | {r['best_epoch']} | {_fmt(r['test_accuracy'])} | "
            f"{_fmt(r['test_macro_f1'])} | {_fmt(r['test_weighted_f1'])} | "
            f"{_fmt(r['test_precision_macro'])} | {_fmt(r['test_recall_macro'])} |"
        )

    L += ["", "## Arm summary (mean ± sd across seeds)", ""]
    L.append("| Arm | Seeds | Test acc | Macro F1 | Weighted F1 |")
    L.append("|---|--:|--:|--:|--:|")
    for arm, rs in sorted(arms.items()):
        def ms(key: str) -> str:
            vals = [x[key] for x in rs]
            if len(vals) == 1:
                return _fmt(vals[0])
            return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"
        L.append(f"| {arm} | {len(rs)} | {ms('test_accuracy')} | {ms('test_macro_f1')} | "
                 f"{ms('test_weighted_f1')} |")

    # Augmentation ablation delta, computed only from arms that actually exist.
    base = arms.get("aslnet_original|aug=False")
    aug = arms.get("aslnet_original|aug=True")
    if base and aug:
        L += ["", "## Augmentation ablation (identical split, architecture, optimiser, budget)", ""]
        L.append("| Metric | Baseline | + Augmentation | Δ (abs) | Δ (rel) |")
        L.append("|---|--:|--:|--:|--:|")
        for key, label in (
            ("test_accuracy", "Accuracy"),
            ("test_macro_f1", "Macro F1"),
            ("test_weighted_f1", "Weighted F1"),
        ):
            b = statistics.mean([x[key] for x in base])
            a = statistics.mean([x[key] for x in aug])
            rel = (a - b) / b if b else float("nan")
            L.append(f"| {label} | {_fmt(b)} | {_fmt(a)} | {a - b:+.4f} | {rel:+.2%} |")
        ns = min(len(base), len(aug))
        L.append("")
        L.append(f"Averaged over {ns} seed(s) per arm. "
                 + ("With fewer than 3 seeds this is a point comparison, not a significance test."
                    if ns < 3 else
                    "No significance test was run; treat the direction, not the magnitude, as the finding."))

    # Robustness, error analysis and interpretability blocks per run.
    for name in runs:
        rp = RESULTS / name / "robustness.json"
        if not rp.exists():
            continue
        rob = json.loads(rp.read_text())
        L += ["", f"## Robustness — `{name}`", ""]
        L.append(f"Clean test accuracy {_fmt(rob['clean']['accuracy'])}, "
                 f"macro F1 {_fmt(rob['clean']['macro_f1'])}.")
        L.append("")
        L.append("| Perturbation | Accuracy | Δ acc | Macro F1 | Δ macro F1 | Rel. acc drop |")
        L.append("|---|--:|--:|--:|--:|--:|")
        for p in rob["perturbations"]:
            L.append(f"| {p['family']} {p['severity']} | {_fmt(p['accuracy'])} | "
                     f"{p['delta_accuracy']:+.4f} | {_fmt(p['macro_f1'])} | "
                     f"{p['delta_macro_f1']:+.4f} | {p['relative_accuracy_drop']:.2%} |")

    for name in runs:
        ep = RESULTS / name / "error_analysis.json"
        if not ep.exists():
            continue
        ea = json.loads(ep.read_text())
        c = ea["concentration"]
        L += ["", f"## Error analysis — `{name}`", ""]
        L.append(f"{c['total_errors']} errors out of {c['total_samples']} test images "
                 f"({c['total_errors'] / max(c['total_samples'], 1):.2%} error rate).")
        L.append("")
        for k in (1, 3, 5, 10):
            L.append(f"* Top-{k} confusion pairs account for "
                     f"**{c[f'top_{k}_pairs_error_share']:.1%}** of all errors "
                     f"({c[f'top_{k}_pairs_error_count']}/{c['total_errors']}).")
        if ea["confusion_pairs"]:
            L += ["", "| True | Predicted | Count | % of all errors | % of class support |",
                  "|---|---|--:|--:|--:|"]
            for p in ea["confusion_pairs"][:10]:
                L.append(f"| {p['true']} | {p['predicted']} | {p['count']} | "
                         f"{p['share_of_all_errors']:.1%} | {p['share_of_class_support']:.1%} |")
        L += ["", "Lowest-recall classes:", "",
              "| Class | Support | Recall | Precision | Errors |", "|---|--:|--:|--:|--:|"]
        for r in ea["class_table"][:8]:
            L.append(f"| {r['class']} | {r['support']} | {_fmt(r['recall'])} | "
                     f"{_fmt(r['precision'])} | {r['errors']} |")

    (RESULTS / "RESULTS.md").write_text("\n".join(L) + "\n")
    print(f"[report] wrote {RESULTS / 'RESULTS.md'} and {RESULTS / 'summary.csv'} "
          f"({len(runs)} runs)")

    # Cross-arm figures.
    training_curves({n: r["history"] for n, r in runs.items()},
                    VIS / "all_training_curves.png", title="Training curves — all arms")
    first = next(iter(runs.values()))
    per_class_recall_plot(
        {n: r["test_metrics"]["per_class"] for n, r in runs.items()},
        first["test_metrics"]["classes"],
        VIS / "per_class_recall.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
