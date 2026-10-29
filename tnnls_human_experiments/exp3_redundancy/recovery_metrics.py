"""Recovery metrics for Experiment 3: neutral-tie top-d and supports.

Adapted from ../AI_implemented/exp2_nonlinear/recovery_metrics.py — the
program's decided tie handling (Appendix B: score ties are never resolved
in favor of the signal-column layout) and support definitions. In this
experiment the true group always occupies the FIRST d columns, with the
nuisance columns appended after, so the legacy lower-index stable
tie-break would systematically favor the true group on ties; the neutral
seeded permutation removes that bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score

DEFAULT_TIE_SEED = 0    # fixed neutral tie order: reproducible, layout-free
SUPPORT_FRACTION = 0.80
DEGENERATE_MI_SUM = 1e-12


@dataclass
class CumulativeSupport:
    """80% cumulative-MI support for one factor."""
    achieved_fraction: float
    size: int
    density: float
    indexes: tuple
    degenerate: bool


def _descending_order(scores, tie_seed=DEFAULT_TIE_SEED):
    """Column indexes by descending score with neutral tie handling.

    Score ties are broken by a seeded permutation of the columns before the
    stable descending sort, so the tie order has no relation to the variable
    layout. ``tie_seed=None`` keeps the lower-index stable rule (legacy
    behavior; not used by the runner).
    """
    scores = np.asarray(scores, dtype=float)
    if tie_seed is None:
        return np.argsort(-scores, kind='stable')
    rng = np.random.default_rng(tie_seed)
    column_order = rng.permutation(len(scores))
    ordered_positions = np.argsort(-scores[column_order], kind='stable')
    return column_order[ordered_positions]


# public name: factor ranking (by explained TC) uses the same neutral rule
descending_order = _descending_order


def cumulative_mi_support(scores, fraction=SUPPORT_FRACTION,
                          tie_seed=DEFAULT_TIE_SEED):
    """Smallest score prefix reaching the requested share of total MI.

    The cumulative share is measured against the sum over ALL variables of
    the nonnegative scores. A factor with total (nonnegative) MI at or
    below DEGENERATE_MI_SUM is degenerate: empty support, and the caller
    scores it as zeros.
    """
    scores = np.maximum(np.asarray(scores, dtype=float), 0.0)
    total = float(scores.sum())
    if total <= DEGENERATE_MI_SUM or not np.isfinite(total):
        return CumulativeSupport(
            achieved_fraction=0.0, size=0, density=0.0, indexes=(),
            degenerate=True,
        )
    order = _descending_order(scores, tie_seed)
    sorted_scores = scores[order]
    cumulative = np.cumsum(sorted_scores) / total
    prefix = int(np.searchsorted(cumulative, fraction - 1e-12,
                                 side='left')) + 1
    prefix = max(1, min(prefix, len(sorted_scores)))
    return CumulativeSupport(
        achieved_fraction=float(cumulative[prefix - 1]),
        size=prefix,
        density=float(prefix / len(scores)),
        indexes=tuple(int(index) for index in order[:prefix]),
        degenerate=False,
    )


def top_d_metrics(scores, truth, d, tie_seed=DEFAULT_TIE_SEED):
    """Precision/recall/F1/Jaccard of the top-d scoring variables."""
    truth = np.asarray(truth, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    d = min(d, len(scores))
    top_indexes = _descending_order(scores, tie_seed)[:d]
    predicted = np.zeros_like(truth)
    predicted[top_indexes] = True
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (
        precision + recall) else 0.0
    union = int((predicted | truth).sum())
    jaccard = tp / union if union else 1.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard,
        'top_indexes': tuple(int(i) for i in top_indexes),
        'support_size': d,
    }


def auprc(scores, truth):
    """AUPRC using scores as the ranking and truth as binary labels."""
    truth = np.asarray(truth, dtype=float)
    if truth.sum() == 0 or len(np.unique(truth)) < 2:
        return float('nan')
    return float(average_precision_score(truth, np.asarray(scores, float)))


def support_vs_truth_metrics(support_indexes, truth):
    """P/R/F1/Jaccard of an index support against a binary truth vector."""
    truth = np.asarray(truth, dtype=bool)
    predicted = np.zeros_like(truth)
    predicted[list(support_indexes)] = True
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (
        precision + recall) else 0.0
    union = int((predicted | truth).sum())
    jaccard = tp / union if union else 1.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard,
    }


def nuisance_false_positive(pred_indices, nuisance_mask):
    """Proportion of the predicted support that falls in nuisance columns."""
    pred = np.asarray(list(pred_indices), dtype=int)
    if len(pred) == 0:
        return 0.0
    return float(np.isin(
        pred, np.flatnonzero(np.asarray(nuisance_mask, bool))).mean())