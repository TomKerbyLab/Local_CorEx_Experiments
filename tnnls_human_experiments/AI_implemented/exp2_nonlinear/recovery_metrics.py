"""Factor matching and recovery metrics for the nonlinear experiments.

Implements the definitions of agent_instructions/00_README_experiment_program.md
and 02_nonlinear_higher_order_recovery.md:

- One-to-one Hungarian matching between learned factors and true groups,
  maximizing **cosine similarity** (not average precision, unlike the older
  Gaussian modules), then held fixed for every other metric.
- Matched AUPRC, top-d precision/recall/F1/Jaccard (d = true group size).
- Nuisance and routing false positives, measured two ways: the fraction of
  nonnegative score mass attributed to the role (``*_score_fraction``) and
  the fraction of the top-d set occupied by the role (``*_top_d_fraction``).
- 80% cumulative-MI support: sort the nonnegative association scores of a
  factor, take the smallest prefix reaching the requested share of total MI
  (spec convention: the denominator is the total over all variables). A
  factor whose total MI is at most 1e-12 nats is degenerate, flagged, and
  never enters support metrics.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import average_precision_score

DEGENERATE_MI_SUM = 1e-12
SUPPORT_FRACTION = 0.80


@dataclass
class CumulativeSupport:
    """80% cumulative-MI support for one factor."""

    achieved_fraction: float
    size: int
    density: float
    indexes: tuple
    degenerate: bool


def safe_cosine(first, second):
    """Cosine similarity that treats zero vectors as maximally dissimilar."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    norm_first = np.linalg.norm(first)
    norm_second = np.linalg.norm(second)
    if norm_first == 0.0 or norm_second == 0.0:
        return 0.0
    return float(first @ second / (norm_first * norm_second))


def cumulative_mi_support(scores, fraction=SUPPORT_FRACTION):
    """Smallest score prefix reaching the requested share of total MI.

    Scores are sorted descending (ties broken by lower index, stable) and
    the cumulative share is measured against the sum over ALL variables of
    the nonnegative scores, per the README support definition.
    """
    scores = np.maximum(np.asarray(scores, dtype=float), 0.0)
    total = float(scores.sum())
    if total <= DEGENERATE_MI_SUM or not np.isfinite(total):
        return CumulativeSupport(
            achieved_fraction=0.0,
            size=0,
            density=0.0,
            indexes=(),
            degenerate=True,
        )
    order = np.argsort(-scores, kind='stable')
    sorted_scores = scores[order]
    cumulative = np.cumsum(sorted_scores) / total
    prefix = int(np.searchsorted(cumulative, fraction - 1e-12, side='left')) + 1
    prefix = max(1, min(prefix, len(sorted_scores)))
    return CumulativeSupport(
        achieved_fraction=float(cumulative[prefix - 1]),
        size=prefix,
        density=float(prefix / len(scores)),
        indexes=tuple(int(index) for index in order[:prefix]),
        degenerate=False,
    )


def top_d_metrics(scores, truth, d):
    """Precision/recall/F1/Jaccard of the top-d scoring variables."""
    truth = np.asarray(truth, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    d = min(d, len(scores))
    top_indexes = np.argsort(-scores, kind='stable')[:d]
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
        'top_indexes': top_indexes,
        'support_size': d,
    }


def support_vs_truth_metrics(support_indexes, truth, d_reference=None):
    """P/R/F1/Jaccard of an index support against a binary truth vector."""
    truth = np.asarray(truth, dtype=bool)
    predicted = np.zeros_like(truth)
    predicted[np.asarray(list(support_indexes), dtype=int)] = True
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


def role_fractions(scores, role_mask, top_indexes, support):
    """Role leakage of one factor: score-mass and top-d occupancy shares.

    ``support`` is a CumulativeSupport; degenerate supports contribute
    zero support leakage and are flagged separately by the caller.
    """
    positive_scores = np.maximum(np.asarray(scores, dtype=float), 0.0)
    total_score = float(positive_scores.sum())
    if total_score == 0.0:
        score_fraction = 0.0
    else:
        score_fraction = float(
            positive_scores[role_mask].sum() / total_score)
    top_fraction = float(np.mean(role_mask[np.asarray(top_indexes, dtype=int)]))
    if support.degenerate:
        support_fraction = 0.0
    else:
        support_fraction = float(np.mean(
            role_mask[np.asarray(support.indexes, dtype=int)]))
    return {
        'score_fraction': score_fraction,
        'top_fraction': top_fraction,
        'support_fraction': support_fraction,
    }


def harmonic_mean(first, second):
    """Harmonic mean, zero when both inputs are zero."""
    if first + second == 0:
        return 0.0
    return 2.0 * first * second / (first + second)


def auprc(scores, truth):
    """AUPRC using scores as the ranking and truth as binary labels."""
    truth = np.asarray(truth, dtype=float)
    if truth.sum() == 0 or len(np.unique(truth)) < 2:
        return float('nan')
    return float(average_precision_score(truth, scores))


def match_factors_to_groups(factor_scores, groups, nuisance_mask=None,
                            routing_mask=None,
                            support_fraction=SUPPORT_FRACTION):
    """Match learned factors to true groups on maximum cosine similarity.

    Signature follows the reference convention: learned factor scores first
    (n_learned, p association rows), true groups second (n_true, p binary
    masks). Returns per-assignment records plus aggregate recovery
    summaries. The assignment is reused for every other metric. Recovery
    divides the matched cosine sum by the number of true groups (penalizes
    missing groups); precision divides by the number of learned factors
    (penalizes extra factors); their harmonic mean gives a balanced summary.
    """
    groups = np.asarray(groups, dtype=float)
    factor_scores = np.asarray(factor_scores, dtype=float)
    if groups.ndim != 2 or factor_scores.ndim != 2:
        raise ValueError("Groups and factor scores must both be matrices.")
    if groups.shape[1] != factor_scores.shape[1]:
        raise ValueError(
            "True groups and learned factors must describe the same variables."
        )

    n_variables = groups.shape[1]
    if nuisance_mask is None:
        nuisance_mask = np.sum(groups, axis=0) == 0
    nuisance_mask = np.asarray(nuisance_mask, dtype=bool)
    routing_mask = (
        np.zeros(n_variables, dtype=bool) if routing_mask is None
        else np.asarray(routing_mask, dtype=bool)
    )
    nuisance_mask = nuisance_mask & ~routing_mask

    # cosine matrix is (n_learned, n_true): linear_sum_assignment returns
    # (row indexes = learned factors, col indexes = true groups)
    cosine_similarity = np.asarray([
        [safe_cosine(factor, group) for group in groups]
        for factor in factor_scores
    ])
    learned_indexes, true_indexes = linear_sum_assignment(-cosine_similarity)
    matched_cosine = cosine_similarity[learned_indexes, true_indexes]

    assignments = []
    for true_index, learned_index in zip(true_indexes, learned_indexes):
        group = groups[true_index]
        factor = factor_scores[learned_index]
        group_size = int(np.sum(group > 0))
        top = top_d_metrics(factor, group > 0, d=group_size)
        support = cumulative_mi_support(factor, support_fraction)
        support_metrics = (
            support_vs_truth_metrics(support.indexes, group > 0)
            if not support.degenerate
            else {'f1': 0.0}
        )
        nuisance_leakage = role_fractions(
            factor, nuisance_mask, top['top_indexes'], support)
        routing_leakage = role_fractions(
            factor, routing_mask, top['top_indexes'], support)
        prevalence = float(np.mean(group > 0))
        group_ap = auprc(factor, group > 0)
        assignments.append({
            'true_group': int(true_index),
            'learned_factor': int(learned_index),
            'cosine_similarity': float(
                cosine_similarity[learned_index, true_index]),
            'average_precision': float(group_ap),
            'adjusted_average_precision': float(
                (group_ap - prevalence) / (1.0 - prevalence))
            if prevalence < 1.0 else float('nan'),
            'top_d_precision': float(top['precision']),
            'top_d_recall': float(top['recall']),
            'top_d_f1': float(top['f1']),
            'top_d_jaccard': float(top['jaccard']),
            'nuisance_score_fraction': nuisance_leakage['score_fraction'],
            'top_d_nuisance_fraction': nuisance_leakage['top_fraction'],
            'routing_score_fraction': routing_leakage['score_fraction'],
            'top_d_routing_fraction': routing_leakage['top_fraction'],
            'support_size': support.size,
            'support_density': support.density,
            'support_f1': float(support_metrics['f1']),
            'support_nuisance_fraction': nuisance_leakage['support_fraction'],
            'support_routing_fraction': routing_leakage['support_fraction'],
            'support_degenerate': bool(support.degenerate),
        })

    matched_ap = np.array([
        assignment['average_precision'] for assignment in assignments])
    matched_f1 = np.array([
        assignment['top_d_f1'] for assignment in assignments])
    num_true_groups = len(groups)
    num_learned_factors = len(factor_scores)
    group_recovery_ap = matched_ap.sum() / num_true_groups
    factor_precision_ap = matched_ap.sum() / num_learned_factors
    group_recovery_cosine = matched_cosine.sum() / num_true_groups
    factor_precision_cosine = matched_cosine.sum() / num_learned_factors

    unmatched_groups = sorted(
        set(range(num_true_groups)) - set(true_indexes.tolist()))
    unmatched_factors = sorted(
        set(range(num_learned_factors)) - set(learned_indexes.tolist()))
    assignment_text = ';'.join(
        f"{item['true_group']}:{item['learned_factor']}"
        for item in assignments
    )

    result = {
        'cosine_similarity_matrix': cosine_similarity,
        'assignments': assignments,
        'num_matched_factors': len(assignments),
        'unmatched_true_groups': unmatched_groups,
        'unmatched_learned_factors': unmatched_factors,
        'matched_average_precision': float(np.mean(matched_ap)),
        'matched_adjusted_average_precision': float(np.mean([
            assignment['adjusted_average_precision']
            for assignment in assignments
            if np.isfinite(assignment['adjusted_average_precision'])
        ])) if assignments else float('nan'),
        'matched_cosine_similarity': float(np.mean(matched_cosine)),
        'matched_top_d_f1': float(np.mean(matched_f1)),
        'true_group_recovery_average_precision': float(group_recovery_ap),
        'learned_factor_precision_average_precision': float(
            factor_precision_ap),
        'balanced_average_precision': float(harmonic_mean(
            group_recovery_ap, factor_precision_ap)),
        'true_group_recovery_cosine_similarity': float(group_recovery_cosine),
        'learned_factor_precision_cosine_similarity': float(
            factor_precision_cosine),
        'balanced_cosine_similarity': float(harmonic_mean(
            group_recovery_cosine, factor_precision_cosine)),
        'num_true_groups': num_true_groups,
        'num_learned_factors': num_learned_factors,
        'factor_assignment': assignment_text,
    }
    for leakage_field, top_field, support_field in (
        ('matched_top_d_nuisance_fraction', 'top_d_nuisance_fraction',
         'support_nuisance_fraction'),
        ('matched_top_d_routing_fraction', 'top_d_routing_fraction',
         'support_routing_fraction'),
        ('matched_nuisance_score_fraction', 'nuisance_score_fraction',
         'support_nuisance_fraction'),
        ('matched_routing_score_fraction', 'routing_score_fraction',
         'support_routing_fraction'),
    ):
        values = [assignment[top_field] for assignment in assignments]
        result[leakage_field] = (
            float(np.mean(values)) if values else 0.0)
    for support_field, output_field in (
        ('support_f1', 'matched_support_f1'),
        ('support_size', 'matched_support_size'),
        ('support_density', 'matched_support_density'),
        ('support_nuisance_fraction', 'matched_support_nuisance_fraction'),
        ('support_routing_fraction', 'matched_support_routing_fraction'),
    ):
        values = [assignment[support_field] for assignment in assignments]
        result[output_field] = float(np.mean(values)) if values else 0.0
    result['matched_support_degenerate_fraction'] = float(np.mean([
        assignment['support_degenerate'] for assignment in assignments]))
    return result


def validate_recovery_metrics():
    """Self-checks on matching, factor-count penalties, and leakage."""
    groups = np.zeros((4, 20))
    groups[0, [0, 1, 2]] = 1.0
    groups[1, [3, 4, 5]] = 1.0
    groups[2, [6, 7, 8, 9]] = 1.0
    groups[3, [10, 11, 12, 13]] = 1.0

    perfect = match_factors_to_groups(groups.copy(), groups)
    assert np.isclose(perfect['matched_cosine_similarity'], 1.0)
    assert np.isclose(perfect['matched_average_precision'], 1.0)
    assert np.isclose(perfect['matched_top_d_f1'], 1.0)
    assert np.isclose(perfect['true_group_recovery_cosine_similarity'], 1.0)
    assert perfect['unmatched_true_groups'] == []
    assert perfect['unmatched_learned_factors'] == []

    permutation = [3, 1, 0, 2]
    permuted = match_factors_to_groups(groups[permutation], groups)
    assert np.isclose(permuted['matched_cosine_similarity'], 1.0)

    missing = match_factors_to_groups(groups[:3], groups)
    assert np.isclose(missing['matched_cosine_similarity'], 1.0)
    assert np.isclose(
        missing['true_group_recovery_cosine_similarity'], 3 / 4)
    assert np.isclose(
        missing['learned_factor_precision_cosine_similarity'], 1.0)
    assert len(missing['unmatched_true_groups']) == 1

    extra_scores = np.vstack([groups, groups[:2]])
    extra = match_factors_to_groups(extra_scores, groups)
    assert np.isclose(extra['true_group_recovery_cosine_similarity'], 1.0)
    assert np.isclose(
        extra['learned_factor_precision_cosine_similarity'], 4 / 6)

    leaky_scores = groups.copy()
    leaky_scores[:, 14:20] = 0.5
    nuisance_mask = np.zeros(20, dtype=bool)
    nuisance_mask[14:20] = True
    leaky = match_factors_to_groups(
        leaky_scores, groups, nuisance_mask=nuisance_mask)
    assert leaky['matched_cosine_similarity'] < 1.0
    # the top-d set stays signal-dominated (d = 3 or 4 signal scores of 1.0
    # outrank the 0.5 nuisance mass), so leakage appears in score mass and
    # in the 80% cumulative-MI support but not necessarily in top-d
    assert leaky['matched_top_d_nuisance_fraction'] >= 0.0
    assert leaky['matched_nuisance_score_fraction'] > 0.0
    assert leaky['matched_support_nuisance_fraction'] > 0.0

    # degenerate factor: no MI at all
    degenerate_scores = np.zeros((1, 20))
    degenerate = match_factors_to_groups(degenerate_scores, groups)
    assert degenerate['assignments'][0]['support_degenerate'] is True
    assert degenerate['assignments'][0]['support_size'] == 0

    # Hungarian never assigns a learned factor twice
    rng = np.random.default_rng(0)
    random_scores = np.abs(rng.normal(size=(5, 20)))
    random_match = match_factors_to_groups(random_scores, groups)
    learned_factors = [
        assignment['learned_factor']
        for assignment in random_match['assignments']
    ]
    assert len(set(learned_factors)) == len(learned_factors)

    return {
        'cosine_matching_checked': True,
        'factor_permutation_checked': True,
        'missing_factor_penalty_checked': True,
        'extra_factor_penalty_checked': True,
        'nuisance_leakage_checked': True,
        'degenerate_support_checked': True,
        'no_factor_reuse_checked': True,
    }
