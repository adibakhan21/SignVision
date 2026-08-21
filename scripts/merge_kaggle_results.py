#!/usr/bin/env python3
"""Import the Kaggle run's results and compare them against the local run.

The Kaggle notebook trains the same 8 arms x 3 seeds on a different device
(T4 / CUDA) than the local run (Apple MPS). Identical code, identical split,
identical seeds — but a different RNG stream, so weight init and shuffle order
diverge. Comparing the two is a free measurement of how much of the between-arm
gap is real and how much is environment noise.

    python scripts/merge_kaggle_results.py            # download + merge + report
    python scripts/merge_kaggle_results.py --skip-download
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics as st
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "results"
KAGGLE = REPO / "results_kaggle"
SLUG = "adibakhan23/signvision-experiments"


def download(dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[merge] downloading {SLUG} output -> {dest}")
    r = subprocess.run(["kaggle", "kernels", "output", SLUG, "-p", str(dest)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
        return False
    return True


def collect(root: Path) -> dict[str, dict]:
    """Map run-name -> results.json contents for every complete run under root."""
    out = {}
    for d in sorted(root.rglob("results.json")):
        try:
            out[d.parent.name] = json.loads(d.read_text())
        except json.JSONDecodeError:
            print(f"[merge] skipping unreadable {d}")
    return out


def arm_of(run: str) -> str:
    return run.rsplit("_seed", 1)[0]


def summarise(runs: dict[str, dict]) -> dict[str, list[float]]:
    arms = defaultdict(list)
    for name, r in runs.items():
        arms[arm_of(name)].append(r["test_metrics"]["accuracy"])
    return arms


def fmt(vals: list[float]) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{st.mean(vals):.4f} ± {st.stdev(vals):.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()

    if not a.skip_download:
        tmp = Path(tempfile.mkdtemp(prefix="kaggle_out_"))
        if not download(tmp):
            print("[merge] download failed — is the run finished?", file=sys.stderr)
            return 1
        src = tmp / "results"
        if not src.is_dir():
            print(f"[merge] no results/ in kernel output; got: "
                  f"{sorted(p.name for p in tmp.iterdir())}", file=sys.stderr)
            return 2
        if KAGGLE.exists():
            shutil.rmtree(KAGGLE)
        shutil.copytree(src, KAGGLE)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[merge] imported {len(list(KAGGLE.iterdir()))} run dirs -> {KAGGLE}")

    local, kag = collect(LOCAL), collect(KAGGLE)
    if not kag:
        print("[merge] no Kaggle results found", file=sys.stderr)
        return 3
    la, ka = summarise(local), summarise(kag)

    L = ["# Cross-environment replication", "",
         "The same code, split and seeds were run in two environments:",
         "",
         "| | local | Kaggle |",
         "|---|---|---|",
         "| device | Apple MPS | NVIDIA T4 (CUDA) |",
         f"| runs | {len(local)} | {len(kag)} |",
         "",
         "The RNG stream differs between backends, so weight initialisation and shuffle",
         "order diverge. Any gap below is environment + seed noise, not a code difference.",
         "",
         "## Test accuracy by arm", "",
         "| Arm | local (n) | Kaggle (n) | Δ mean |", "|---|--:|--:|--:|"]

    for arm in sorted(set(la) | set(ka)):
        lv, kv = la.get(arm, []), ka.get(arm, [])
        d = (st.mean(kv) - st.mean(lv)) if lv and kv else None
        L.append(f"| `{arm}` | {fmt(lv)} ({len(lv)}) | {fmt(kv)} ({len(kv)}) | "
                 f"{d:+.4f} |" if d is not None else
                 f"| `{arm}` | {fmt(lv)} ({len(lv)}) | {fmt(kv)} ({len(kv)}) | — |")

    # The headline question: does each finding survive the environment change?
    L += ["", "## Do the findings survive?", ""]
    checks = [
        ("ReLU fix (aslnet_relu − baseline)", "baseline", "aslnet_relu"),
        ("Augmentation cost, 10 ep (augmented − baseline)", "baseline", "augmented"),
        ("Augmentation cost, 30 ep (augmented_e30 − baseline_e30)",
         "baseline_e30", "augmented_e30"),
    ]
    L += ["| Contrast | local Δ | Kaggle Δ | same sign? |", "|---|--:|--:|:--:|"]
    for label, a_arm, b_arm in checks:
        def delta(src):
            if src.get(a_arm) and src.get(b_arm):
                return st.mean(src[b_arm]) - st.mean(src[a_arm])
            return None
        dl, dk = delta(la), delta(ka)
        if dl is None or dk is None:
            L.append(f"| {label} | — | — | — |")
            continue
        same = "yes" if (dl > 0) == (dk > 0) else "**no**"
        L.append(f"| {label} | {dl:+.4f} | {dk:+.4f} | {same} |")

    L += ["", "A contrast that keeps its sign across two independent environments is a finding.",
          "One that flips is within noise and must not be claimed.", ""]

    out = LOCAL / "CROSS_ENV.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[merge] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
