"""Shared metrics: cumulative-MI supports, Hungarian matching, recovery metrics,
partition metrics. Implements the definitions in 00_README_experiment_program.md."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, average_precision_score

DEGENERATE_MI_SUM = 1e-12


# ---------------------------------------------------------------------------
# Cumulative-MI supports
# ---------------------------------------------------------------------------

@dataclass
class CumulativeSupport:
    """Result of cumulative-MI support construction for one factor."""

    requested_fraction: float
    achieved_fraction: float
    size: int
    density: float
    indices: tuple[int, ...]  # ordered hidden-unit indices, best first
    scores: tuple[float, ...]  # MI scores in the same order
    degenerate: bool


def cumulative_mi_support(
    scores: np.ndarray,
    fraction: float,
    valid_mask: np.ndarray | None = None,
) -> CumulativeSupport:
    """Cumulative-MI support per the README definition.

    scores: association scores a_{ji} for one factor across all variables.
    valid_mask: True for coordinates eligible for selection (e.g. nondegenerate
        hidden units). If None, all coordinates are eligible.

    Sort eligible scores descending; take the smallest prefix achieving the
    requested cumulative fraction. Degenerate factors (total MI <= 1e-12) are
    flagged and get an empty support.
    """
    scores = np.asarray(scores, dtype=float)
    p = scores.shape[0]
    if valid_mask is None:
        valid_mask = np.ones(p, dtype=bool)
    total = float(scores[valid_mask].sum())
    if total <= DEGENERATE_MI_SUM or not np.isfinite(total):
        return CumulativeSupport(
            requested_fraction=fraction,
            achieved_fraction=0.0,
            size=0,
            density=0.0,
            indices=(),
            scores=(),
            degenerate=True,
        )
    order = np.argsort(-scores[valid_mask], kind="stable")
    valid_idx = np.flatnonzero(valid_mask)
    sorted_scores = scores[valid_idx][order]
    sorted_idx = valid_idx[order]
    cum = np.cumsum(sorted_scores) / total
    k = int(np.searchsorted(cum, fraction - 1e-12, side="left")) + 1
    k = max(1, min(k, len(sorted_scores)))
    achieved = float(cum[k - 1])
    return CumulativeSupport(
        requested_fraction=float(fraction),
        achieved_fraction=achieved,
        size=int(k),
        density=float(k / p),
        indices=tuple(int(i) for i in sorted_idx[:k]),
        scores=tuple(float(s) for s in sorted_scores[:k]),
        degenerate=False,
    )


def nested_supports(scores: np.ndarray, fractions: list[float], valid_mask=None) -> list[CumulativeSupport]:
    """Supports at multiple thresholds; validated to be nested."""
    sups = [cumulative_mi_support(scores, q, valid_mask) for q in fractions]
    for a, b in zip(sups, sups[1:]):
        if not a.degenerate and not b.degenerate:
            assert set(a.indices) <= set(b.indices), "cumulative-MI supports must be nested"
    return sups


# ---------------------------------------------------------------------------
# Hungarian factor matching
# ---------------------------------------------------------------------------

def hungarian_match(score_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one assignment maximizing total score.

    score_matrix: shape (n_learned, n_true). Returns
    (learned_idx, true_idx, values) for the matched pairs. Assignment is
    optimal even when the matrix is not square (rows may exceed columns or
    vice versa); each learned factor is used at most once and each true group
    at most once.
    """
    score_matrix = np.asarray(score_matrix, dtype=float)
    r, c = linear_sum_assignment(-score_matrix)  # maximize
    return r, c, score_matrix[r, c]


# ---------------------------------------------------------------------------
# Recovery metrics
# ---------------------------------------------------------------------------

def auprc(scores: np.ndarray, truth: np.ndarray) -> float:
    """AUPRC using scores as ranking and truth as binary labels."""
    truth = np.asarray(truth, dtype=float)
    if truth.sum() == 0 or len(np.unique(truth)) < 2:
        return float("nan")
    return float(average_precision_score(truth, scores))


def topk_metrics(scores: np.ndarray, truth: np.ndarray, k: int) -> dict:
    """Precision/recall/F1/Jaccard for the top-k scoring coordinates."""
    truth = np.asarray(truth, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    k = min(k, len(scores))
    # ties broken by lower index, matching np.argsort(-scores, kind="stable")
    top = np.argsort(-scores, kind="stable")[:k]
    pred = np.zeros_like(truth)
    pred[top] = True
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    union = int((pred | truth).sum())
    jaccard = tp / union if union else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "support_size": k,
        "n_true_positives": int(truth.sum()),
    }


def false_positive_proportion(pred_indices, excluded: np.ndarray) -> float:
    """Proportion of predicted support that falls in the excluded index set."""
    pred = np.asarray(list(pred_indices), dtype=int)
    if len(pred) == 0:
        return 0.0
    return float(np.isin(pred, np.flatnonzero(excluded)).mean())


def support_vs_truth_metrics(support_indices, truth: np.ndarray) -> dict:
    """P/R/F1/Jaccard of an index support against a binary truth vector.

    Used for cumulative-MI supports whose size is not fixed at the true
    group size.
    """
    truth = np.asarray(truth, dtype=bool)
    pred = np.zeros_like(truth)
    pred[np.asarray(list(support_indices), dtype=int)] = True
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    union = int((pred | truth).sum())
    jaccard = tp / union if union else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "support_size": int(pred.sum()),
        "n_true_positives": int(truth.sum()),
    }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def match_factors_to_groups(
    assoc_matrix: np.ndarray,  # (n_learned, p) association scores
    true_groups: np.ndarray,  # (n_true, p) binary vectors
    top_k: int | None = None,
) -> list[dict]:
    """Hungarian-match learned factors to true groups and compute all metrics.

    The assignment maximizes cosine similarity and is held fixed for all
    metrics (per the specs). Returns one record per matched (factor, group)
    pair. top_k defaults to the size of each true group; callers may override
    (e.g. top-4 for Experiment 1's size-4 groups).
    """
    assoc = np.asarray(assoc_matrix, dtype=float)
    groups = np.asarray(true_groups, dtype=float)
    n_learned, p = assoc.shape
    n_true = groups.shape[0]

    # cosine similarity matrix (n_learned x n_true)
    cos = np.zeros((n_learned, n_true))
    for j in range(n_learned):
        for g in range(n_true):
            cos[j, g] = cosine_similarity(assoc[j], groups[g])
    learned_idx, true_idx, cos_vals = hungarian_match(cos)

    records = []
    for j, g, cv in zip(learned_idx, true_idx, cos_vals):
        truth_vec = groups[g] > 0
        k = int(top_k) if top_k is not None else int(truth_vec.sum())
        rec = {
            "learned_factor": int(j),
            "true_group": int(g),
            "cosine": float(cv),
            "auprc": auprc(assoc[j], truth_vec.astype(float)),
        }
        rec.update({f"topk_{key}": v for key, v in topk_metrics(assoc[j], truth_vec, k=k).items()})
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Partition metrics
# ---------------------------------------------------------------------------

def contingency(labels_true: np.ndarray, labels_est: np.ndarray) -> np.ndarray:
    """Contingency matrix, rows=true, cols=estimated."""
    t, e = np.asarray(labels_true), np.asarray(labels_est)
    t_ids, t_inv = np.unique(t, return_inverse=True)
    e_ids, e_inv = np.unique(e, return_inverse=True)
    mat = np.zeros((len(t_ids), len(e_ids)), dtype=np.int64)
    np.add.at(mat, (t_inv, e_inv), 1)
    return mat


def align_partition(labels_est: np.ndarray, labels_true: np.ndarray) -> tuple[np.ndarray, float]:
    """Hungarian align estimated 2-cluster labels to true regions.

    Requires both label sets to contain exactly two distinct values (the
    one-to-one map is only forced for k=2, per the spec). Returns
    (aligned labels, aligned accuracy). Invariant to label swaps.
    """
    if len(np.unique(labels_est)) != 2 or len(np.unique(labels_true)) != 2:
        raise ValueError("align_partition requires exactly two clusters in both label sets")
    mat = contingency(labels_true, labels_est)
    r, c = linear_sum_assignment(-mat)
    mapping = {int(est): int(tru) for tru, est in zip(r, c)}
    aligned = np.array([mapping[int(v)] for v in labels_est])
    acc = float((aligned == labels_true).mean())
    return aligned, acc


def partition_metrics(labels_true: np.ndarray, labels_est: np.ndarray) -> dict:
    """ARI, purity, aligned accuracy (NaN unless k=2), cluster sizes (JSON)."""
    import json

    labels_true = np.asarray(labels_true)
    labels_est = np.asarray(labels_est)
    mat = contingency(labels_true, labels_est)
    purity = float(mat.max(axis=0).sum() / mat.sum())
    if len(np.unique(labels_true)) == 2 and len(np.unique(labels_est)) == 2:
        _, acc = align_partition(labels_est, labels_true)
    else:
        acc = float("nan")
    sizes = {str(k): int(v) for k, v in zip(*np.unique(labels_est, return_counts=True))}
    return {
        "ari": float(adjusted_rand_score(labels_true, labels_est)),
        "purity": purity,
        "aligned_accuracy": acc,
        "cluster_sizes": json.dumps(sizes),
    }


# ---------------------------------------------------------------------------
# Statistical summaries
# ---------------------------------------------------------------------------

def summarize_replicates(values: np.ndarray, confidence: float = 0.95) -> dict:
    """Mean, sd, se, and normal-approx CI for replicate-level values."""
    from scipy import stats

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"mean": float("nan"), "sd": float("nan"), "se": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n_valid": 0, "n_failed": 0}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n)
    if n > 1:
        crit = stats.t.ppf(1 - (1 - confidence) / 2, n - 1)
        lo, hi = mean - crit * se, mean + crit * se
    else:
        lo = hi = mean
    return {"mean": mean, "sd": sd, "se": se, "ci_low": float(lo), "ci_high": float(hi),
            "n_valid": n, "n_failed": int(len(values) - n) if hasattr(values, "__len__") else 0}


def paired_difference_summary(x: np.ndarray, y: np.ndarray, confidence: float = 0.95) -> dict:
    """Paired differences x - y with mean, se, and CI over paired replicates."""
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return {"mean_diff": float("nan"), "se_diff": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"), "n_pairs": 0}
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if n > 1:
        crit = stats.t.ppf(1 - (1 - confidence) / 2, n - 1)
        lo, hi = mean - crit * se, mean + crit * se
    else:
        lo = hi = mean
    return {"mean_diff": mean, "se_diff": se, "ci_low": float(lo), "ci_high": float(hi), "n_pairs": n}