"""Shared bootstrap for experiment scripts: path setup and run directories."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

RESULTS = REPO / "results"
VIS = REPO / "visualizations"
CKPT = REPO / "checkpoints"
CACHE = REPO / "data" / "cache"


def run_dir(name: str) -> Path:
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def vis_dir(name: str) -> Path:
    d = VIS / name
    d.mkdir(parents=True, exist_ok=True)
    return d
