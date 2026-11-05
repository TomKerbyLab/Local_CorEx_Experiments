"""Experiment 5 data: HF imagenet-1k loading, fresh ResNet-18 feature cache,
demonstrated cache verification, per-class 25/25 split.

Ported from tnnls_experiments/exp5_imagenet/data.py. Deviations from the
reference implementation (recorded per the protocol freeze):
  - The old resnet_100_feature_*.pkl exploration cache is NOT reused; the
    cache is regenerated fresh under the new pipeline (build_cache), then
    demonstrated by verify_cache() on >= 1000 stratified freshly-decoded
    images agreeing to atol 1e-4 on cached logits (freeze F1).
  - The split is by HF dataset row order within each class (the HF set
    exposes no canonical file paths; agreed design deviation in F1).

Everything downstream consumes the cache; the convolutional backbone is
never rerun during clustering/fits/ablations.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

IMAGENET_HF_CACHE_DIR = Path("/yunity/tkerby2/.cache/huggingface/hub/datasets--imagenet-1k")
AFRICAN_GREY_HF_IDX = 87
AFRICAN_GREY_WNID = "n01817953"

PREPROCESS = {
    "resize": 256,
    "center_crop": 224,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "interpolation": "bilinear (torchvision default for Resize)",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_validation_split():
    """HF imagenet-1k validation split, loaded from the local cache only."""
    from datasets import load_dataset

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    ds = load_dataset("imagenet-1k", split="validation")
    if len(ds) != 50_000:
        raise RuntimeError(f"expected 50,000 validation images, got {len(ds)}")
    return ds


def imagenet_dataloader(batch_size: int = 256, num_workers: int = 16):
    """DataLoader over the HF validation split with the canonical
    IMAGENET1K_V1 preprocessing (Resize 256, CenterCrop 224, Normalize)."""
    from torch.utils.data import DataLoader
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=PREPROCESS["mean"], std=PREPROCESS["std"]),
    ])
    ds = load_validation_split()
    ds = ds.with_transform(lambda b, tf=tf: {
        "pixel_values": [tf(img.convert("RGB")) for img in b["image"]],
        "label": b["label"],
    })
    return (DataLoader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers), ds)


def class_mapping() -> list[dict]:
    """Class index -> {wnid, name}. Source: the HF dataset's own classes.py
    (the label feature names are the same strings)."""
    scripts = sorted(IMAGENET_HF_CACHE_DIR.glob("snapshots/*/classes.py"))
    if not scripts:
        raise RuntimeError("HF imagenet-1k classes.py not found in the local cache")
    spec = importlib.util.spec_from_file_location("exp5_imagenet_classes", scripts[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # plain OrderedDict of wnid -> name
    rows = []
    for i, (wnid, name) in enumerate(mod.IMAGENET2012_CLASSES.items()):
        rows.append({"class_index": i, "wnid": wnid, "class_name": name})
    assert AFRICAN_GREY_WNID == rows[AFRICAN_GREY_HF_IDX]["wnid"], \
        "African-grey index/wnid mismatch against the HF class mapping"
    return rows


def weights_checkpoint_hash() -> str:
    """sha256 of the local torchvision ResNet-18 IMAGENET1K_V1 weights file."""
    path = Path.home() / ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def imagenet_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class row-order split: first 25 discovery, last 25 evaluation.

    Returns (is_discovery, is_evaluation) boolean arrays. Fails loudly if
    any class does not have exactly 50 images (no silent rebalancing).
    """
    labels = np.asarray(labels)
    is_disc = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        assert len(idx) == 50, f"class {c} has {len(idx)} images, expected 50"
        is_disc[idx[:25]] = True
    return is_disc, ~is_disc


def dataset_fingerprint(labels: np.ndarray, ds=None) -> dict:
    """Fingerprint of the split source: dataset hash, label-array hash."""
    labels = np.asarray(labels)
    fp = {
        "n_images": int(len(labels)),
        "n_classes": int(np.unique(labels).size),
        "labels_sha256": hashlib.sha256(labels.astype(np.int64).tobytes()).hexdigest(),
        "hf_cache_dir": str(IMAGENET_HF_CACHE_DIR),
    }
    if ds is not None and hasattr(ds, "_fingerprint"):
        fp["ds_fingerprint"] = str(ds._fingerprint)
    return fp


def per_class_integrity(labels: np.ndarray) -> dict:
    """Every class must have exactly 50 valid images (spec hard stop)."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=1000)
    bad = np.flatnonzero(counts != 50)
    report = {
        "n_images": int(len(labels)),
        "n_classes": int((counts > 0).sum()),
        "n_classes_wrong_count": int(len(bad)),
        "bad_classes": bad.tolist()[:50],
    }
    if len(bad):
        raise AssertionError(f"ImageNet integrity check failed: "
                             f"{len(bad)} classes without exactly 50 images "
                             f"(first: {bad[:10].tolist()})")
    return report


# ---------------------------------------------------------------------------
# Fresh cache build + F1 demonstrated verification
# ---------------------------------------------------------------------------

def resnet18_model(device: str = "cuda:0"):
    """ResNet-18 with IMAGENET1K_V1 weights, eval mode, on device."""
    import torch
    from torchvision import models

    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    m.to(device).eval()
    return m


def build_cache(cache_path: Path, device: str = "cuda:0", batch_size: int = 256,
                num_workers: int = 16) -> dict:
    """Forward pass over all 50k validation images; cache pooled 512-d
    features (input of the final fc), 1000-d logits, and labels."""
    import torch

    dl, ds = imagenet_dataloader(batch_size=batch_size, num_workers=num_workers)
    model = resnet18_model(device)
    feats = np.zeros((50_000, 512), dtype=np.float32)
    logits = np.zeros((50_000, 1000), dtype=np.float32)
    labels = np.zeros(50_000, dtype=np.int64)
    pooled = {}

    def _hook(mod, inp, out):
        pooled["h"] = inp[0].detach()

    handle = model.fc.register_forward_hook(_hook)
    row = 0
    t0 = time.time()
    with torch.no_grad():
        for batch in dl:
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = np.asarray(batch["label"])
            out = model(x)          # logits computed by the full forward pass
            h = pooled["h"].flatten(1)
            logits[row:row + len(y)] = out.cpu().numpy()
            feats[row:row + len(y)] = h.cpu().numpy()
            labels[row:row + len(y)] = y
            row += len(y)
    handle.remove()
    assert row == 50_000
    runtime = time.time() - t0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=feats, logits=logits, labels=labels)
    meta = {
        "cache_path": str(cache_path),
        "n_images": int(row),
        "model": "torchvision.models.resnet18",
        "weights": "IMAGENET1K_V1",
        "weights_sha256": weights_checkpoint_hash(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "device": device,
        "batch_size": batch_size,
        "preprocess": PREPROCESS,
        "forward_runtime_s": runtime,
        "dataset_fingerprint": dataset_fingerprint(labels, ds),
        "integrity": per_class_integrity(labels),
    }
    with open(cache_path.with_suffix(".cache_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_cache(cache_path: Path) -> dict:
    with np.load(cache_path) as z:
        return {"features": z["features"], "logits": z["logits"],
                "labels": z["labels"]}


def cache_split(cache_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    cache = load_cache(cache_path)
    is_disc, is_eval = imagenet_split(cache["labels"])
    return is_disc, is_eval, cache


def verify_cache(cache_path: Path, n_per_class: int = 2, device: str = "cuda:0",
                 seed: int = 7002, atol: float = 1e-4) -> dict:
    """F1 demonstrated cache verification.

    Re-decodes and re-forwards a stratified sample of n_per_class images per
    class (2000 total >= 1000) through the backbone and compares pooled
    features and logits against the cache. Verified requires
    max |logits_fresh - logits_cached| <= atol and label agreement.
    Returns the agreement statistics (callers write them to the run log).
    """
    import torch
    from torchvision import transforms

    cache = load_cache(cache_path)
    rng = np.random.default_rng(seed)
    check_idx_list = []
    for c in range(1000):
        idx_c = np.flatnonzero(cache["labels"] == c)
        assert len(idx_c) == 50
        check_idx_list.append(rng.choice(idx_c, size=n_per_class, replace=False))
    check_idx = np.sort(np.concatenate(check_idx_list))
    assert len(check_idx) >= 1000

    # fresh decode + transform of exactly the sampled rows
    ds = load_validation_split()
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=PREPROCESS["mean"], std=PREPROCESS["std"]),
    ])
    model = resnet18_model(device)
    pooled = {}

    def _hook(mod, inp, out):
        pooled["h"] = inp[0].detach()

    handle = model.fc.register_forward_hook(_hook)
    feats_f = np.zeros((len(check_idx), 512), dtype=np.float32)
    logits_f = np.zeros((len(check_idx), 1000), dtype=np.float32)
    labels_f = np.zeros(len(check_idx), dtype=np.int64)
    t0 = time.time()
    with torch.no_grad():
        for i, idx in enumerate(check_idx):
            img = ds[int(idx)]["image"].convert("RGB")
            x = tf(img)[None].to(device)
            out = model(x)
            feats_f[i] = pooled["h"].flatten(1).cpu().numpy()
            logits_f[i] = out.cpu().numpy()
            labels_f[i] = cache["labels"][idx]
    handle.remove()
    runtime = time.time() - t0

    feat_diff = np.abs(feats_f - cache["features"][check_idx])
    logit_diff = np.abs(logits_f - cache["logits"][check_idx])
    n_label_match = int((labels_f == cache["labels"][check_idx]).sum())
    report = {
        "n_check": int(len(check_idx)),
        "n_per_class": n_per_class,
        "atol": atol,
        "max_feature_absdiff": float(feat_diff.max()),
        "mean_feature_absdiff": float(feat_diff.mean()),
        "max_logit_absdiff": float(logit_diff.max()),
        "mean_logit_absdiff": float(logit_diff.mean()),
        "frac_logit_within_atol": float((logit_diff <= atol).mean()),
        "frac_feature_within_atol": float((feat_diff <= atol).mean()),
        "label_match_rate": n_label_match / len(check_idx),
        "verified": bool(logit_diff.max() <= atol and n_label_match == len(check_idx)),
        "runtime_s": runtime,
    }
    return report
