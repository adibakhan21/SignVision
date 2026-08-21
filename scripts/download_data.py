#!/usr/bin/env python3
"""Fetch the ASL Alphabet dataset into ./data.

Primary source is Kaggle (`grassknoted/asl-alphabet`), which is the dataset the
original SignVision notebook used. It requires authentication:

    kaggle auth login          # browser OAuth, or
    # place an API token at ~/.kaggle/kaggle.json

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --check     # report what is already present
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SLUG = "grassknoted/asl-alphabet"
TRAIN = DATA / "asl_alphabet_train"
TEST = DATA / "asl_alphabet_test"


def report() -> bool:
    ok = True
    for name, p in (("train", TRAIN), ("test", TEST)):
        if p.exists():
            subdirs = sorted(d for d in p.iterdir() if d.is_dir())
            if subdirs:
                n = sum(len(list(d.glob("*.jpg"))) for d in subdirs)
                print(f"  {name}: {p}  ({len(subdirs)} classes, {n} jpg)")
            else:
                n = len(list(p.glob("*.jpg")))
                print(f"  {name}: {p}  (flat, {n} jpg)")
        else:
            print(f"  {name}: MISSING ({p})")
            ok = False
    return ok


def _flatten(root: Path, target_name: str) -> None:
    """Kaggle nests the folders one level deep; normalise to data/<target_name>."""
    nested = root / target_name / target_name
    if nested.is_dir():
        tmp = root / f"{target_name}__tmp"
        shutil.move(str(nested), str(tmp))
        shutil.rmtree(root / target_name, ignore_errors=True)
        shutil.move(str(tmp), str(root / target_name))


def download() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    if shutil.which("kaggle") is None:
        print("error: the `kaggle` CLI is not installed (pip install kaggle)", file=sys.stderr)
        return 2
    print(f"[data] downloading {SLUG} -> {DATA} (~1.1 GB)")
    r = subprocess.run(
        ["kaggle", "datasets", "download", "-d", SLUG, "-p", str(DATA), "--force"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        print("\nKaggle authentication is required. Run `kaggle auth login`.", file=sys.stderr)
        return r.returncode

    zips = sorted(DATA.glob("*.zip"))
    if not zips:
        print("error: no zip downloaded", file=sys.stderr)
        return 3
    z = zips[-1]
    print(f"[data] extracting {z.name}")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(DATA)
    z.unlink()
    _flatten(DATA, "asl_alphabet_train")
    _flatten(DATA, "asl_alphabet_test")
    print("[data] done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report what is present")
    a = ap.parse_args()
    print(f"[data] directory: {DATA}")
    present = report()
    if a.check:
        return 0 if present else 1
    if present:
        print("[data] already present; nothing to do")
        return 0
    return download()


if __name__ == "__main__":
    raise SystemExit(main())
