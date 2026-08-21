"""Dataset discovery, deterministic splitting, caching and transforms.

Design notes
------------
* The split is computed from a *fixed* ``split_seed`` that is independent of the
  training seed, so every experimental arm sees byte-identical train/val/test
  partitions regardless of how the model is seeded.
* Images are decoded and resized once into a uint8 cache. Every arm then reads
  the same pixels, which removes JPEG-decode nondeterminism and makes the
  augmentation ablation a clean single-variable comparison.
* Augmentation is applied on-the-fly to training samples only. Validation and
  test are always evaluated with the identical deterministic transform.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .config import REPO_ROOT, AugConfig, DataConfig

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class SplitIndex:
    """Index arrays into the cached image array, plus the class vocabulary."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    labels: np.ndarray
    classes: list[str]
    files: list[str]

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def resolve_root(root: str | Path) -> Path:
    """Interpret a config path relative to the repository root when not absolute."""
    p = Path(root)
    return p if p.is_absolute() else (REPO_ROOT / p)


def scan_imagefolder(root: str | Path) -> tuple[list[str], np.ndarray, list[str]]:
    """Return (relative file paths, integer labels, class names) sorted deterministically."""
    root = resolve_root(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not classes:
        raise RuntimeError(f"no class subdirectories under {root}")
    files: list[str] = []
    labels: list[int] = []
    for ci, cname in enumerate(classes):
        cdir = root / cname
        names = sorted(p.name for p in cdir.iterdir() if p.suffix.lower() in IMG_EXTS)
        files.extend(f"{cname}/{n}" for n in names)
        labels.extend([ci] * len(names))
    return files, np.asarray(labels, dtype=np.int64), classes


def select_subset(labels: np.ndarray, cfg: DataConfig) -> np.ndarray:
    """Deterministically choose which images are used at all.

    Applied before caching so only the images an experiment can actually see are
    decoded and stored. Driven by ``split_seed``, so the subset is identical for
    every arm and every training seed.
    """
    rng = np.random.default_rng(cfg.split_seed)
    keep = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        if cfg.images_per_class is not None:
            idx = idx[: cfg.images_per_class]
        keep.append(np.sort(idx))
    return np.concatenate(keep)


def stratified_split(
    labels: np.ndarray,
    cfg: DataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic per-class shuffle then contiguous train/val/test slices.

    ``labels`` are the labels of the *selected* images and the returned arrays are
    positions into that selection. Stratified, so every class keeps the same
    proportions in all three partitions; driven by ``cfg.split_seed`` only, which
    is why the split is stable across arms and across training seeds.
    """
    rng = np.random.default_rng(cfg.split_seed + 1)
    tr, va, te = [], [], []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(cfg.train_frac * n))
        n_va = int(round(cfg.val_frac * n))
        tr.append(idx[:n_tr])
        va.append(idx[n_tr : n_tr + n_va])
        te.append(idx[n_tr + n_va :])
    return (
        np.sort(np.concatenate(tr)),
        np.sort(np.concatenate(va)),
        np.sort(np.concatenate(te)),
    )


def _cache_key(root: str, size: int, n_files: int) -> str:
    h = hashlib.sha1(f"{resolve_root(root).resolve()}|{size}|{n_files}".encode()).hexdigest()[:12]
    return f"cache_{size}px_{n_files}n_{h}.npy"


def build_cache(cfg: DataConfig, cache_dir: str | Path, verbose: bool = True) -> tuple[Path, list[str], np.ndarray, list[str]]:
    """Decode + resize the selected images once into a uint8 array on disk.

    Returns (cache_path, selected_files, selected_labels, classes). Cache row *i*
    corresponds to ``selected_files[i]``.
    """
    all_files, all_labels, classes = scan_imagefolder(cfg.root)
    sel = select_subset(all_labels, cfg)
    files = [all_files[i] for i in sel]
    labels = all_labels[sel]

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _cache_key(cfg.root, cfg.image_size, len(files))

    if cache_path.exists():
        if verbose:
            print(f"[data] using existing cache {cache_path.name}")
        return cache_path, files, labels, classes

    s_ = cfg.image_size
    tmp = cache_path.with_suffix(".partial.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(files), s_, s_, 3))
    root = resolve_root(cfg.root)
    for i, rel in enumerate(files):
        with Image.open(root / rel) as im:
            im = im.convert("RGB").resize((s_, s_), Image.BILINEAR)
            arr[i] = np.asarray(im, dtype=np.uint8)
        if verbose and (i + 1) % 5000 == 0:
            print(f"[data] cached {i + 1}/{len(files)}")
    arr.flush()
    del arr
    tmp.rename(cache_path)  # atomic: a partial cache is never mistaken for a complete one
    if verbose:
        print(f"[data] wrote {cache_path.name} "
              f"({cache_path.stat().st_size / 1e9:.2f} GB, {len(files)} images)")
    return cache_path, files, labels, classes


def make_transforms(cfg: AugConfig, train: bool) -> v2.Transform:
    """Build the transform pipeline.

    The evaluation transform is a pure uint8->float32 [0,1] conversion, matching
    the original project's ``T.ToTensor()``. No normalisation is introduced, so
    the augmentation ablation isolates augmentation alone.
    """
    ops: list[v2.Transform] = []
    if train and cfg.enabled:
        ops.append(
            v2.RandomAffine(
                degrees=cfg.rotation_deg,
                translate=(cfg.translate, cfg.translate),
                scale=(cfg.scale_min, cfg.scale_max),
            )
        )
        ops.append(
            v2.ColorJitter(
                brightness=cfg.brightness,
                contrast=cfg.contrast,
                saturation=cfg.saturation,
                hue=cfg.hue,
            )
        )
        if cfg.horizontal_flip:
            ops.append(v2.RandomHorizontalFlip(p=0.5))
    ops.append(v2.ToDtype(torch.float32, scale=True))
    return v2.Compose(ops)


class CachedASLDataset(Dataset):
    """Reads uint8 HWC images from the memmapped cache and applies transforms."""

    def __init__(
        self,
        cache_path: str | Path,
        indices: np.ndarray,
        labels: np.ndarray,
        transform: v2.Transform | None = None,
    ) -> None:
        self.cache_path = str(cache_path)
        self.indices = np.asarray(indices)
        self.labels = np.asarray(labels)
        self.transform = transform
        self._arr: np.ndarray | None = None

    def _array(self) -> np.ndarray:
        # Opened lazily so the memmap survives DataLoader worker forking.
        if self._arr is None:
            self._arr = np.load(self.cache_path, mmap_mode="r")
        return self._arr

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        gi = int(self.indices[i])
        img = torch.from_numpy(np.array(self._array()[gi])).permute(2, 0, 1)  # CHW uint8
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[gi])


def prepare(cfg: DataConfig, cache_dir: str | Path, verbose: bool = True) -> tuple[Path, SplitIndex]:
    """One-call entry point: cache the images and compute the fixed split."""
    cache_path, files, labels, classes = build_cache(cfg, cache_dir, verbose=verbose)
    tr, va, te = stratified_split(labels, cfg)
    split = SplitIndex(train=tr, val=va, test=te, labels=labels, classes=classes, files=files)
    if verbose:
        print(f"[data] {len(classes)} classes, {len(files)} images available")
        print(f"[data] split (seed={cfg.split_seed}): {split.sizes()}")
    return cache_path, split
