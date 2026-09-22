"""Experiment 4 training: 5 seeds (5100-5104), spec-exact configuration.

Default training: Adam lr 1e-3, batch 128, CE loss, max 100 epochs, early
stopping on classifier-validation loss with patience 10, restore the
lowest-val-loss checkpoint. No ablation result may influence checkpoint
selection. Test accuracy below 97% is a failure FLAG (not silent retraining).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..common import seeds
from ..common.config import CHECKPOINTS_DIR
from .classifier import MLP
from .data import load_splits

TRAIN_CFG = {
    "lr": 1e-3,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 10,
    "dropout_p": 0.5,
    "acc_failure_threshold": 0.97,
}


def checkpoint_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def train_one_model(
    root_seed: int,
    splits: dict,
    cfg: dict | None = None,
    device: str | None = None,
    max_epochs_override: int | None = None,
    save: bool = True,
) -> dict:
    """Train one classifier. Returns metadata + the model in eval mode."""
    cfg = {**TRAIN_CFG, **(cfg or {})}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(root_seed)

    model = MLP(dropout_p=cfg["dropout_p"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    lossf = torch.nn.CrossEntropyLoss()

    tx = splits["train_x"].to(device)
    ty = splits["train_y"].to(device)
    vx = splits["val_x"].to(device)
    vy = splits["val_y"].to(device)

    max_epochs = max_epochs_override or cfg["max_epochs"]
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    epochs_no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        # deterministic minibatch order derived from (seed, epoch)
        g = torch.Generator(device="cpu").manual_seed(
            seeds.derive_seed(root_seed, "minibatch", epoch))
        perm = torch.randperm(len(tx), generator=g).to(device)
        total_loss = 0.0
        for i in range(0, len(tx), cfg["batch_size"]):
            idx = perm[i : i + cfg["batch_size"]]
            logits = model(tx[idx])
            loss = lossf(logits, ty[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        train_loss = total_loss / len(tx)

        model.eval()
        with torch.no_grad():
            val_logits = model(vx)
            val_loss = lossf(val_logits, vy).item()
            val_acc = (val_logits.argmax(1) == vy).float().mean().item()
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_acc": val_acc})
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()
    runtime = time.time() - t0

    # held-out test metrics (recorded; never used for selection)
    tex, tey = splits["test_x"].to(device), splits["test_y"].to(device)
    with torch.no_grad():
        test_logits = model(tex)
        test_acc = (test_logits.argmax(1) == tey).float().mean().item()
        test_loss = lossf(test_logits, tey).item()

    meta = {
        "root_seed": root_seed,
        "train_cfg": cfg,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "runtime_s": runtime,
        "acc_flag_below_97": test_acc < cfg["acc_failure_threshold"],
        "n_epochs_run": len(history),
    }
    if save:
        ckpt_dir = CHECKPOINTS_DIR / "mnist"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"mnist_seed{root_seed}.pt"
        torch.save(best_state, path)
        meta["checkpoint"] = str(path)
        meta["checkpoint_hash"] = checkpoint_hash(path)
        with open(ckpt_dir / f"mnist_seed{root_seed}_meta.json", "w") as f:
            json.dump({k: v for k, v in meta.items() if k != "train_cfg"}, f,
                      indent=2, default=str)
    return {"model": model, "meta": meta, "history": history}


def load_model(root_seed: int, device: str | None = None) -> MLP:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = CHECKPOINTS_DIR / "mnist" / f"mnist_seed{root_seed}.pt"
    model = MLP(dropout_p=0.5)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.to(device).eval()
    return model


def main(seeds_list: list[int], cfg: dict | None = None,
         max_epochs_override: int | None = None) -> list[dict]:
    splits = load_splits(save_dir=CHECKPOINTS_DIR / "mnist")
    results = []
    for root_seed in seeds_list:
        out = train_one_model(root_seed, splits, cfg,
                              max_epochs_override=max_epochs_override)
        m = out["meta"]
        print(f"seed {root_seed}: best epoch {m['best_epoch']} "
              f"val {m['best_val_loss']:.4f} test acc {m['test_acc']:.4f} "
              f"({m['runtime_s']:.0f}s)"
              + (" [FLAG: acc<0.97]" if m["acc_flag_below_97"] else ""))
        results.append(m)
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[5100, 5101, 5102, 5103, 5104])
    ap.add_argument("--epochs", type=int, default=None,
                    help="override max epochs (smoke training only)")
    args = ap.parse_args()
    main(args.seeds, max_epochs_override=args.epochs)