"""Experiment 5 data: ImageNet split, cached-feature verification, clustering.

Split rule (agreed deviation): the HF imagenet-1k validation set provides no
file paths, so images are sorted by dataset row order within each class:
first 25 -> discovery, last 25 -> evaluation. The dataset fingerprint is
recorded so the split is reproducible.

The existing 28 GB cache (resnet_100_feature_*.pkl) contains 50k x 512 pooled
features, 50k x 1000 logits, and labels from IMAGENET1K_V1 weights. The cache
is verified against fresh ResNet-18 forward passes on a sample of decoded
images before being trusted (verify_cache).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_STEM = "resnet_100_feature"

# ImageNet class mapping (HF index -> (wnid, name)); loaded from the HF repo
AFRICAN_GREY_HF_IDX = 87
AFRICAN_GREY_WNID = "n01817953"


def imagenet_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class row-order split: first 25 discovery, last 25 evaluation.

    Returns (is_discovery, is_evaluation) boolean arrays.
    """
    labels = np.asarray(labels)
    is_disc = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        assert len(idx) == 50, f"class {c} has {len(idx)} images, expected 50"
        is_disc[idx[:25]] = True
    return is_disc, ~is_disc


def load_cached(cache_dir: Path | str | None = None) -> dict:
    """Load the existing ResNet-18 feature cache (50k images)."""
    root = Path(cache_dir) if cache_dir else REPO_ROOT
    with open(root / f"{CACHE_STEM}_features.pkl", "rb") as f:
        features = pickle.load(f)
    with open(root / f"{CACHE_STEM}_preds.pkl", "rb") as f:
        preds = pickle.load(f)
    with open(root / f"{CACHE_STEM}_labels.pkl", "rb") as f:
        labels = pickle.load(f)
    import torch

    features = features if isinstance(features, np.ndarray) else features.numpy()
    preds = preds if isinstance(preds, np.ndarray) else preds.numpy()
    labels = np.asarray(labels if not isinstance(labels, torch.Tensor)
                        else labels.numpy())
    return {"features": features, "preds": preds, "labels": labels}


def load_cached_inputs_meta(cache_dir: Path | str | None = None) -> dict | None:
    """Optional: metadata for the cached input tensors (not loaded fully)."""
    return None  # 28 GB inputs are not loaded; cache verification decodes
                 # fresh images from the HF dataset instead


def imagenet_dataloader(batch_size=256):
    """Fresh DataLoader over the HF imagenet-1k validation split with the
    canonical IMAGENET1K_V1 preprocessing (Resize 256, CenterCrop 224,
    Normalize)."""
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    ds = load_dataset("imagenet-1k", split="validation")
    ds = ds.with_transform(lambda b, tf=tf: {
        "pixel_values": [tf(img.convert("RGB")) for img in b["image"]],
        "label": b["label"],
    })
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=8), ds


def resnet18_model(device: str = "cuda"):
    """ResNet-18 with IMAGENET1K_V1 weights, eval mode, on device."""
    import torch
    from torchvision import models

    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.to(device).eval()
    return m


def verify_cache(n_check: int = 256, device: str = "cuda", seed: int = 7002) -> dict:
    """Verify cached features/logits against fresh forward passes.

    Samples n_check validation images (deterministic), runs the backbone,
    compares pooled features and logits to the cache. Returns a report dict.
    """
    import torch

    cache = load_cached()
    dl, _ = imagenet_dataloader()
    model = resnet18_model(device)
    rng = np.random.default_rng(seed)
    check_idx = np.sort(rng.choice(50_000, size=n_check, replace=False))

    # stream the loader once, keeping only the sampled indices
    kept_x, kept_y, row = [], [], 0
    with torch.no_grad():
        for batch in dl:
            b = len(batch["pixel_values"])
            sel = check_idx[(check_idx >= row) & (check_idx < row + b)] - row
            if len(sel):
                kept_x.append(batch["pixel_values"][sel].to(device))
                kept_y.append(batch["label"][sel].numpy())
            row += b
            if row > check_idx.max():
                break
    x = torch.cat(kept_x)
    y = np.concatenate(kept_y)
    with torch.no_grad():
        feats = model(x)  # resnet18 forward gives logits; need pooled feats

        # hooks
        pooled = {}
        def hook(mod, inp, out):
            pooled["h"] = inp[0].detach()
        model.fc.register_forward_hook(hook)
        logits = model(x)
        h = pooled["h"].flatten(1)

    feats_np = h.cpu().numpy()
    logits_np = logits.cpu().numpy()
    feat_diff = np.abs(feats_np - cache["features"][check_idx]).max()
    logit_diff = np.abs(logits_np - cache["preds"][check_idx]).max()
    label_match = float((y == cache["labels"][check_idx]).mean())
    return {
        "n_check": n_check,
        "max_feature_absdiff": float(feat_diff),
        "max_logit_absdiff": float(logit_diff),
        "label_match_rate": label_match,
        "verified": bool(feat_diff < 1e-3 and logit_diff < 1e-3 and label_match == 1.0),
    }