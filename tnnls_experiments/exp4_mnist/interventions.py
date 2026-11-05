"""Experiment 4 intervention evaluation.

Semantics (spec): activation masking at the layer's BN+ReLU output, model in
eval mode, evaluation on all 10,000 untouched test images. For true digit y,
mean baseline and ablated probability vectors give the signed shift matrix
Delta_{y,c} = pbar^{(A)}_{y,c} - pbar^{(0)}_{y,c}. Metrics: T_y, T_max,
T_weighted, H_input, D_input, H_output/D_output at the most affected class,
directional concordance, accuracy changes.
"""

from __future__ import annotations

import numpy as np
import torch

from .classifier import MLP

N_CLASSES = 10


@torch.no_grad()
def predict_proba(model: MLP, x: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    """Softmax probabilities, eval mode, masks applied as set on the model."""
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        logits = model(x[i : i + batch_size])
        out.append(torch.softmax(logits, dim=1))
    return torch.cat(out)


def baseline_cache(model: MLP, x: torch.Tensor) -> dict:
    """Baseline (all-ones mask) predictions, cached once per model seed."""
    model.clear_masks()
    model.eval()
    with torch.no_grad():
        logits = predict_logits(model, x)
        probs = torch.softmax(logits, dim=1)
    return {"logits": logits, "probs": probs}


@torch.no_grad()
def predict_logits(model: MLP, x: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(x), batch_size):
        out.append(model(x[i : i + batch_size]))
    return torch.cat(out)


def classwise_shift_matrix(
    probs_abl: np.ndarray, probs_base: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """10x10 signed shift matrix Delta_{y,c}."""
    delta = np.zeros((N_CLASSES, N_CLASSES))
    for y in range(N_CLASSES):
        mask = labels == y
        if mask.sum() == 0:
            continue
        delta[y] = probs_abl[mask].mean(axis=0) - probs_base[mask].mean(axis=0)
    return delta


def classwise_mean_probs(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """10x10 matrix of mean probabilities pbar_{y,c}: class-mean probability
    vector for each true digit y. Complement of classwise_shift_matrix."""
    out = np.zeros((N_CLASSES, N_CLASSES))
    for y in range(N_CLASSES):
        mask = labels == y
        if mask.sum() == 0:
            continue
        out[y] = probs[mask].mean(axis=0)
    return out


def effect_metrics(delta: np.ndarray, n_per_class: np.ndarray) -> dict:
    """All aggregate effect/specificity metrics from the shift matrix."""
    T = 0.5 * np.abs(delta).sum(axis=1)              # T_y
    total = T.sum()
    if total > 0:
        r = T / total
        H_input = -(r * np.log(r, where=r > 0)).sum() / np.log(N_CLASSES)
        H_input = float(H_input)
        D_input = float(r.max())
    else:
        H_input, D_input = float("nan"), float("nan")
    n = n_per_class.sum()
    T_weighted = float((n_per_class / n * T).sum()) if n else float("nan")
    T_max = float(T.max()) if len(T) else float("nan")

    y_star = int(np.argmax(T))
    drow = delta[y_star]
    denom = np.abs(drow).sum()
    if denom > 0:
        q = np.abs(drow) / denom
        H_output = float(-(q * np.log(q, where=q > 0)).sum() / np.log(N_CLASSES))
        D_output = float(q.max())
    else:
        H_output, D_output = float("nan"), float("nan")
    c_minus = int(np.argmin(drow))
    c_plus = int(np.argmax(drow))
    return {
        "T_max": T_max,
        "T_weighted": T_weighted,
        "H_input": H_input,
        "D_input": D_input,
        "H_output": H_output,
        "D_output": D_output,
        "y_star": y_star,
        "c_minus": c_minus,
        "c_plus": c_plus,
    }


def substantial_class_concentrated(T_max: float, H_input: float,
                                   T_thresh: float = 0.10,
                                   H_thresh: float = 0.50) -> bool:
    """Predeclared criterion: T_max >= 0.10 and H_input <= 0.50."""
    if not np.isfinite(T_max) or not np.isfinite(H_input):
        return False
    return bool(T_max >= T_thresh and H_input <= H_thresh)


def sensitivity_table(T_max: float, H_input: float) -> dict[str, bool]:
    """Predeclared supplementary sensitivity grid."""
    out = {}
    for Tt in (0.05, 0.10, 0.15):
        for Ht in (0.40, 0.50, 0.60):
            out[f"T{Tt:.2f}_H{Ht:.2f}"] = substantial_class_concentrated(
                T_max, H_input, Tt, Ht)
    return out


def accuracy_metrics(probs_abl: np.ndarray, probs_base: np.ndarray,
                     labels: np.ndarray) -> dict:
    pred_a = probs_abl.argmax(1)
    pred_0 = probs_base.argmax(1)
    acc_a = (pred_a == labels).mean()
    acc_0 = (pred_0 == labels).mean()
    # per-class recall change
    recall_change = np.zeros(N_CLASSES)
    for y in range(N_CLASSES):
        m = labels == y
        ra = (pred_a[m] == y).mean() if m.sum() else np.nan
        r0 = (pred_0[m] == y).mean() if m.sum() else np.nan
        recall_change[y] = ra - r0
    # mean cross-entropy change
    eps = 1e-12
    ce_a = -np.log(np.clip(probs_abl[np.arange(len(labels)), labels], eps, 1)).mean()
    ce_0 = -np.log(np.clip(probs_base[np.arange(len(labels)), labels], eps, 1)).mean()
    # confusion-matrix change (10x10x10: too large to store per intervention;
    # store the L1 norm of the change)
    from sklearn.metrics import confusion_matrix

    cm_a = confusion_matrix(labels, pred_a, labels=range(N_CLASSES))
    cm_0 = confusion_matrix(labels, pred_0, labels=range(N_CLASSES))
    return {
        "accuracy_change": float(acc_a - acc_0),
        "accuracy_ablated": float(acc_a),
        "accuracy_baseline": float(acc_0),
        "mean_ce_change": float(ce_a - ce_0),
        "cm_l1_change": float(np.abs(cm_a - cm_0).sum()),
        "recall_change_per_class": recall_change.tolist(),
    }


def evaluate_intervention(
    model: MLP,
    x_test: torch.Tensor,
    labels: np.ndarray,
    baseline: dict,
    layer: int,
    support_indices: list[int],
    batch_size: int = 2048,
) -> dict:
    """One intervention: mask, evaluate on the full test set, all metrics.

    Returns the shift matrix + metrics; the caller restores the mask.
    """
    model.set_mask(layer, support_indices)
    try:
        probs_abl = predict_proba(model, x_test, batch_size).cpu().numpy()
    finally:
        model.clear_masks()
    probs_base = baseline["probs"].cpu().numpy()
    delta = classwise_shift_matrix(probs_abl, probs_base, labels)
    n_per_class = np.bincount(labels, minlength=N_CLASSES).astype(float)
    metrics = effect_metrics(delta, n_per_class)
    acc = accuracy_metrics(probs_abl, probs_base, labels)
    return {
        "delta": delta,
        "metrics": {**metrics, **acc},
        "probs_abl": probs_abl,
    }


def individual_effect_metrics(
    probs_abl: np.ndarray, probs_base: np.ndarray, labels: np.ndarray
) -> dict:
    """Cancel-free companions to the class-mean metrics.

    U_y = mean over images of true class y of 0.5*||p_i^A - p_i^0||_1.
    U_y >= T_y always (triangle inequality): a small T_y with a large U_y
    means substantial individual effects whose directions cancel within the
    class, which the class-mean shift cannot reveal. Also the prediction-flip
    rate (argmax change), overall and per true class.
    """
    d = np.abs(probs_abl - probs_base).sum(axis=1) / 2.0  # per image, (n,)
    flips = probs_abl.argmax(1) != probs_base.argmax(1)
    U = np.zeros(N_CLASSES)
    flip_y = np.zeros(N_CLASSES)
    for y in range(N_CLASSES):
        m = labels == y
        if m.sum():
            U[y] = d[m].mean()
            flip_y[y] = flips[m].mean()
    n = labels.shape[0]
    return {
        "U_max": float(U.max()),
        "U_weighted": float((np.bincount(labels, minlength=N_CLASSES) / n * U).sum())
        if n else float("nan"),
        "y_U_star": int(np.argmax(U)),
        "U_per_class": [float(v) for v in U],
        "flip_rate": float(flips.mean()),
        "flip_rate_per_class": [float(v) for v in flip_y],
    }


def random_control_probability(stat_lc: float, stats_random: list[float],
                               larger_is_extreme: bool = True) -> float:
    """One-sided empirical random-control probability (R=20 by spec)."""
    stats_random = np.asarray(stats_random, dtype=float)
    if larger_is_extreme:
        ge = (stats_random >= stat_lc).sum()
    else:
        ge = (stats_random <= stat_lc).sum()
    return float((1 + ge) / (len(stats_random) + 1))


def concordance(c_star: int, c_minus: int) -> bool:
    """Directional concordance: factor-associated class is most reduced."""
    return bool(c_star == c_minus)