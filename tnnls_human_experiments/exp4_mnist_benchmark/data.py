"""Experiment 4 MNIST data splits (fixed, stratified, seed 5000).

Canonical MNIST: 60,000 train / 10,000 test.
Fixed stratified split of the training set:
  - 45,000 classifier-training
  -  5,000 classifier-validation (checkpoint selection only)
  - 10,000 discovery
The 10,000 canonical test images are untouched for final intervention
evaluation. Exact observation indices are saved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SPLIT_SEED = 5000
N_TRAIN = 45_000
N_VAL = 5_000
N_DISCOVERY = 10_000


def mnist_arrays(data_root: str | Path | None = None):
    """Load MNIST via torchvision; return (train_x, train_y, test_x, test_y)
    as float tensors scaled to [0,1], shape (N, 784)."""
    import torch
    from torchvision import datasets

    if data_root is None:
        # decision #6: MNIST under the repository-relative data/ directory
        data_root = (Path(__file__).resolve().parents[2]
                     / "data" / "mnist_classifier")
    root = str(data_root)
    train = datasets.MNIST(root=root, train=True, download=True)
    test = datasets.MNIST(root=root, train=False, download=True)
    tx = torch.from_numpy(train.data.numpy().astype(np.float32) / 255.0).reshape(-1, 784)
    ty = torch.tensor(train.targets, dtype=torch.long)
    vx = torch.from_numpy(test.data.numpy().astype(np.float32) / 255.0).reshape(-1, 784)
    vy = torch.tensor(test.targets, dtype=torch.long)
    return tx, ty, vx, vy


def stratified_indices(labels: np.ndarray, sizes: dict[str, int],
                       seed: int) -> dict[str, np.ndarray]:
    """Stratified split preserving digit proportions as closely as possible.

    sizes maps split name -> requested count; the total must equal len(labels).
    Deterministic given (labels, sizes, seed).
    """
    from sklearn.model_selection import train_test_split

    labels = np.asarray(labels)
    total = sum(sizes.values())
    assert total == len(labels), f"sizes sum {total} != {len(labels)}"
    remaining_labels = labels.copy()
    remaining_idx = np.arange(len(labels))
    out: dict[str, np.ndarray] = {}
    names = list(sizes)
    # peel off splits in order, each stratified on the remaining pool
    for name in names[:-1]:
        n = sizes[name]
        idx_out, idx_keep = train_test_split(
            remaining_idx, train_size=n, stratify=remaining_labels,
            random_state=seed)
        out[name] = np.sort(idx_out)
        keep_mask = np.ones(len(remaining_idx), dtype=bool)
        keep_mask[np.isin(remaining_idx, idx_out)] = False
        remaining_labels = remaining_labels[keep_mask]
        remaining_idx = idx_keep
    out[names[-1]] = np.sort(remaining_idx)
    return out


def make_splits(data_root=None, save_dir=None) -> dict:
    """Build (or load) the fixed splits; save exact indices."""
    import json

    import torch

    tx, ty, vx, vy = mnist_arrays(data_root)
    splits = stratified_indices(
        ty.numpy(), {"train": N_TRAIN, "val": N_VAL, "discovery": N_DISCOVERY},
        seed=SPLIT_SEED)
    result = {
        "train": splits["train"],
        "val": splits["val"],
        "discovery": splits["discovery"],
        "test": np.arange(len(vy)),
        "train_x": tx, "train_y": ty,
        "val_x": tx[torch.tensor(splits["val"])], "val_y": ty[torch.tensor(splits["val"])],
        "discovery_x": tx[torch.tensor(splits["discovery"])],
        "discovery_y": ty[torch.tensor(splits["discovery"])],
        "test_x": vx, "test_y": vy,
    }
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        meta = {
            "split_seed": SPLIT_SEED,
            "sizes": {k: int(len(v)) for k, v in splits.items()},
            "class_counts": {
                name: np.bincount(result[f"{name}_y"].numpy(), minlength=10).tolist()
                for name in ("train", "val", "discovery", "test")
            },
        }
        np.savez(
            Path(save_dir) / "mnist_splits.npz",
            **{k: (v.numpy() if torch.is_tensor(v) else v) for k, v in result.items()},
        )
        with open(Path(save_dir) / "mnist_splits_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    return result


def load_splits(data_root=None, save_dir=None) -> dict:
    """Load fixed splits from disk if available, else build and save."""
    p = Path(save_dir) / "mnist_splits.npz" if save_dir else None
    if p and p.exists():
        z = np.load(p, allow_pickle=True)
        out = {k: z[k] for k in z.files}
        # tensors for convenience
        import torch

        for k in ("train_x", "val_x", "discovery_x", "test_x"):
            out[k] = torch.from_numpy(out[k]).float()
        for k in ("train_y", "val_y", "discovery_y", "test_y"):
            out[k] = torch.from_numpy(out[k]).long()
        return out
    return make_splits(data_root, save_dir)