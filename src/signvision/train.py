"""Training loop: one function, shared by every experimental arm.

Everything that could differ between arms (augmentation, architecture, seed) is
passed in through ``ExperimentConfig``; the loop itself is identical, which is
what makes the comparisons controlled.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import CachedASLDataset, SplitIndex, make_transforms
from .evaluate import compute_metrics, macro_f1, predict
from .models import build_model
from .utils import count_parameters, get_device, seed_worker, set_seed


def build_loaders(
    cache_path: str | Path, split: SplitIndex, cfg: ExperimentConfig
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tf = make_transforms(cfg.aug, train=True)
    eval_tf = make_transforms(cfg.aug, train=False)

    ds_tr = CachedASLDataset(cache_path, split.train, split.labels, train_tf)
    ds_va = CachedASLDataset(cache_path, split.val, split.labels, eval_tf)
    ds_te = CachedASLDataset(cache_path, split.test, split.labels, eval_tf)

    g = torch.Generator()
    g.manual_seed(cfg.train.seed)
    common = dict(num_workers=cfg.train.num_workers, worker_init_fn=seed_worker)
    return (
        DataLoader(ds_tr, batch_size=cfg.train.batch_size, shuffle=True, generator=g, **common),
        DataLoader(ds_va, batch_size=256, shuffle=False, **common),
        DataLoader(ds_te, batch_size=256, shuffle=False, **common),
    )


def run_training(
    cfg: ExperimentConfig,
    cache_path: str | Path,
    split: SplitIndex,
    ckpt_path: str | Path,
    verbose: bool = True,
) -> dict:
    """Train, select the best epoch on validation macro F1, return history + test metrics."""
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    classes = split.classes
    n_classes = len(classes)

    tr_dl, va_dl, te_dl = build_loaders(cache_path, split, cfg)
    model = build_model(cfg.train.model, n_classes, cfg.data.image_size).to(device)
    loss_fn = nn.CrossEntropyLoss()
    if cfg.train.optimizer.lower() == "adam":
        opt = torch.optim.Adam(
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
    elif cfg.train.optimizer.lower() == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.weight_decay
        )
    else:
        raise ValueError(f"unknown optimizer {cfg.train.optimizer}")

    params = count_parameters(model)
    if verbose:
        print(f"[train] {cfg.name} | model={cfg.train.model} | aug={cfg.aug.enabled} "
              f"| seed={cfg.train.seed} | device={device} | params={params['total']:,}")

    history: list[dict] = []
    best = {"score": -1.0, "epoch": -1}
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for ep in range(1, cfg.train.epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        ep_t0 = time.time()
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = run_loss / total
        train_acc = correct / total

        yv, pv, _ = predict(model, va_dl, device)
        val_acc = float((yv == pv).mean())
        val_f1 = macro_f1(yv, pv, n_classes)

        rec = {
            "epoch": ep,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "seconds": time.time() - ep_t0,
        }
        history.append(rec)
        if verbose:
            print(f"  ep {ep:2d}  loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
                  f"val_acc {val_acc:.4f}  val_macroF1 {val_f1:.4f}  ({rec['seconds']:.1f}s)")

        score = val_f1 if cfg.train.select_on == "val_macro_f1" else val_acc
        if score > best["score"]:
            best = {"score": score, "epoch": ep}
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.to_dict(),
                    "classes": classes,
                    "epoch": ep,
                    "val_macro_f1": val_f1,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )

    # Restore the selected epoch before touching the test set.
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])

    yt, pt, probs = predict(model, te_dl, device)
    test_metrics = compute_metrics(yt, pt, classes)
    yv, pv, _ = predict(model, va_dl, device)
    val_metrics = compute_metrics(yv, pv, classes)

    if verbose:
        print(f"[train] best epoch {best['epoch']} (val macroF1 {best['score']:.4f}) | "
              f"TEST acc {test_metrics['accuracy']:.4f} macroF1 {test_metrics['macro_f1']:.4f} "
              f"| total {time.time() - t0:.1f}s")

    return {
        "name": cfg.name,
        "config": cfg.to_dict(),
        "params": params,
        "device": str(device),
        "history": history,
        "best_epoch": best["epoch"],
        "best_val_score": best["score"],
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "checkpoint": str(ckpt_path),
        "wall_seconds": time.time() - t0,
        "predictions": {
            "y_true": yt.tolist(),
            "y_pred": pt.tolist(),
            "max_prob": probs.max(axis=1).tolist(),
            "test_indices": np.asarray(split.test).tolist(),
        },
    }
