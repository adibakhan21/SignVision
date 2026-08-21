"""Experiment configuration.

Every hyperparameter that affects a result lives here so a run is fully described
by one serialisable object, which is written next to the metrics it produced.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
VIS_DIR = REPO_ROOT / "visualizations"
CKPT_DIR = REPO_ROOT / "checkpoints"


@dataclass
class DataConfig:
    """Dataset location, sampling and the deterministic split."""

    # Relative to the repository root, so configs are portable.
    root: str = "data/asl_alphabet_train"
    # Kaggle's official 28-image test folder, kept as a secondary sanity set.
    official_test_root: str = "data/asl_alphabet_test"
    image_size: int = 100
    # Stratified cap per class. The full dataset is 3000/class; capping keeps the
    # whole experiment suite tractable on one laptop GPU and is applied
    # identically to every arm, so it is never the experimental variable.
    images_per_class: int | None = 1000
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    split_seed: int = 1234  # fixed independently of the training seed


@dataclass
class AugConfig:
    """Augmentation switch and its parameters (Experiment B only)."""

    enabled: bool = False
    rotation_deg: float = 12.0
    translate: float = 0.10
    scale_min: float = 0.90
    scale_max: float = 1.10
    brightness: float = 0.25
    contrast: float = 0.25
    saturation: float = 0.15
    hue: float = 0.02
    # Horizontal flip is deliberately NOT used: ASL handshapes are chiral and a
    # mirrored 'D' is not a valid 'D'. See README, Experiments section.
    horizontal_flip: bool = False


@dataclass
class TrainConfig:
    model: str = "aslnet_original"
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    # Model selection: keep the checkpoint with the best validation macro F1
    # rather than the last epoch. Applied identically to every arm.
    select_on: str = "val_macro_f1"


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        return cls(
            name=d.get("name", "baseline"),
            data=DataConfig(**d.get("data", {})),
            aug=AugConfig(**d.get("aug", {})),
            train=TrainConfig(**d.get("train", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))
