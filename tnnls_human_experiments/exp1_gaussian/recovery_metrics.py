"""Factor matching and recovery metrics for Local CorEx experiments."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from sklearn.metrics import average_precision_score

from gaussian_data import CLUSTER_1_GROUPS, groups_to_indicator_vectors


RECOVERY_COLUMNS = [
    'matched_average_precision',
    'matched_adjusted_average_precision',
    'matched_cosine_distance',
    'matched_cosine_similarity',
    'matched_top_k_f1',
    'matched_nuisance_score_fraction',
    'matched_top_k_nuisance_fraction',
    'true_group_recovery_average_precision',
    'learned_factor_precision_average_precision',
    'balanced_average_precision',
    'true_group_recovery_cosine_similarity',
    'learned_factor_precision_cosine_similarity',
    'balanced_cosine_similarity',
    'num_matched_factors',
    'num_unmatched_true_groups',
    'num_unmatched_learned_factors',
    'group_best_average_precision',
    'group_best_cosine_distance',
    'factor_best_average_precision',
    'factor_best_cosine_distance',
    'factor_assignment',
]


def get_factor_variable_scores(corex_model, linear=True):
    """Extract a factors-by-variables score matrix from either estimator."""
    if linear:
        scores = corex_model.moments['MI']
    else:
        scores = corex_model.mis
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("Factor-variable scores must be a two-dimensional array.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Factor-variable scores contain non-finite values.")
    return scores


def safe_cosine_distance(first, second):
    """Return worst-case distance when either vector has zero magnitude."""
    if np.linalg.norm(first) == 0 or np.linalg.norm(second) == 0:
        return 1.0
    return float(cosine(first, second))


def create_factor_score_matrices(groups, factor_scores):
    """Score every true-group and learned-factor pairing."""
    groups = np.asarray(groups, dtype=float)
    factor_scores = np.asarray(factor_scores, dtype=float)
    if groups.ndim != 2 or factor_scores.ndim != 2:
        raise ValueError("Groups and factor scores must both be matrices.")
    if groups.shape[1] != factor_scores.shape[1]:
        raise ValueError(
            "True groups and learned factors must describe the same variables."
        )

    average_precision = np.zeros((len(groups), len(factor_scores)))
    adjusted_average_precision = np.zeros_like(average_precision)
    cosine_distance = np.zeros_like(average_precision)

    for group_index, group in enumerate(groups):
        prevalence = np.mean(group)
        for factor_index, factor in enumerate(factor_scores):
            ap_score = average_precision_score(group, factor)
            average_precision[group_index, factor_index] = ap_score
            adjusted_average_precision[group_index, factor_index] = (
                (ap_score - prevalence) / (1.0 - prevalence)
            )
            cosine_distance[group_index, factor_index] = safe_cosine_distance(
                group,
                factor,
            )

    return average_precision, adjusted_average_precision, cosine_distance


def matched_support_diagnostics(groups, factor_scores,
                                group_indexes, factor_indexes):
    """Measure support recovery and nuisance leakage for matched pairs."""
    groups = np.asarray(groups, dtype=float)
    factor_scores = np.asarray(factor_scores, dtype=float)
    nuisance_mask = np.sum(groups, axis=0) == 0

    top_k_f1 = []
    nuisance_score_fraction = []
    top_k_nuisance_fraction = []

    for group_index, factor_index in zip(group_indexes, factor_indexes):
        group = groups[group_index]
        factor = factor_scores[factor_index]
        group_size = int(np.sum(group > 0))
        top_indexes = np.argsort(-factor, kind='stable')[:group_size]
        recovered_count = np.sum(group[top_indexes] > 0)

        # Both the true and estimated supports contain k variables, so
        # precision, recall, and F1 all equal recovered_count / group_size.
        top_k_f1.append(float(recovered_count / group_size))
        top_k_nuisance_fraction.append(float(
            np.mean(nuisance_mask[top_indexes])
        ))

        nonnegative_scores = np.maximum(factor, 0.0)
        total_score = np.sum(nonnegative_scores)
        if total_score == 0:
            nuisance_score_fraction.append(0.0)
        else:
            nuisance_score_fraction.append(float(
                np.sum(nonnegative_scores[nuisance_mask]) / total_score
            ))

    return {
        'top_k_f1': np.asarray(top_k_f1),
        'nuisance_score_fraction': np.asarray(nuisance_score_fraction),
        'top_k_nuisance_fraction': np.asarray(top_k_nuisance_fraction),
    }


def harmonic_mean(first, second):
    """Calculate a harmonic mean, returning zero when both inputs are zero."""
    if first + second == 0:
        return 0.0
    return 2 * first * second / (first + second)


def match_factors_to_groups(groups, factor_scores):
    """Match factors to groups once using maximum average precision.

    The assignment is reused for every other metric. Recovery divides the
    matched total by the number of true groups, which penalizes missing groups.
    Precision divides by the number of learned factors, which penalizes extra
    factors. Their harmonic mean gives a balanced summary.
    """
    (average_precision,
     adjusted_average_precision,
     cosine_distance) = create_factor_score_matrices(groups, factor_scores)

    group_indexes, factor_indexes = linear_sum_assignment(-average_precision)
    matched_average_precision = average_precision[group_indexes, factor_indexes]
    matched_adjusted_average_precision = adjusted_average_precision[
        group_indexes,
        factor_indexes,
    ]
    matched_cosine_distance = cosine_distance[group_indexes, factor_indexes]
    matched_cosine_similarity = np.maximum(0.0, 1.0 - matched_cosine_distance)
    support_diagnostics = matched_support_diagnostics(
        groups,
        factor_scores,
        group_indexes,
        factor_indexes,
    )

    num_true_groups = len(groups)
    num_learned_factors = len(factor_scores)
    true_group_ap = matched_average_precision.sum() / num_true_groups
    learned_factor_ap = matched_average_precision.sum() / num_learned_factors
    true_group_cosine = matched_cosine_similarity.sum() / num_true_groups
    learned_factor_cosine = matched_cosine_similarity.sum() / num_learned_factors

    unmatched_groups = sorted(
        set(range(num_true_groups)) - set(group_indexes.tolist())
    )
    unmatched_factors = sorted(
        set(range(num_learned_factors)) - set(factor_indexes.tolist())
    )
    assignments = [
        {
            'true_group': int(group_index),
            'learned_factor': int(factor_index),
            'average_precision': float(ap_score),
            'adjusted_average_precision': float(adjusted_ap_score),
            'cosine_distance': float(cosine_score),
            'top_k_f1': float(top_k_f1),
            'nuisance_score_fraction': float(nuisance_fraction),
            'top_k_nuisance_fraction': float(top_k_nuisance),
        }
        for (group_index, factor_index, ap_score, adjusted_ap_score,
             cosine_score, top_k_f1, nuisance_fraction,
             top_k_nuisance) in zip(
            group_indexes,
            factor_indexes,
            matched_average_precision,
            matched_adjusted_average_precision,
            matched_cosine_distance,
            support_diagnostics['top_k_f1'],
            support_diagnostics['nuisance_score_fraction'],
            support_diagnostics['top_k_nuisance_fraction'],
        )
    ]

    return {
        'average_precision_matrix': average_precision,
        'adjusted_average_precision_matrix': adjusted_average_precision,
        'cosine_distance_matrix': cosine_distance,
        'assignments': assignments,
        'matched_average_precision': float(np.mean(matched_average_precision)),
        'matched_adjusted_average_precision': float(np.mean(
            matched_adjusted_average_precision
        )),
        'matched_cosine_distance': float(np.mean(matched_cosine_distance)),
        'matched_cosine_similarity': float(np.mean(matched_cosine_similarity)),
        'matched_top_k_f1': float(np.mean(support_diagnostics['top_k_f1'])),
        'matched_nuisance_score_fraction': float(np.mean(
            support_diagnostics['nuisance_score_fraction']
        )),
        'matched_top_k_nuisance_fraction': float(np.mean(
            support_diagnostics['top_k_nuisance_fraction']
        )),
        'true_group_recovery_average_precision': float(true_group_ap),
        'learned_factor_precision_average_precision': float(learned_factor_ap),
        'balanced_average_precision': float(harmonic_mean(
            true_group_ap,
            learned_factor_ap,
        )),
        'true_group_recovery_cosine_similarity': float(true_group_cosine),
        'learned_factor_precision_cosine_similarity': float(
            learned_factor_cosine
        ),
        'balanced_cosine_similarity': float(harmonic_mean(
            true_group_cosine,
            learned_factor_cosine,
        )),
        'num_matched_factors': len(group_indexes),
        'unmatched_true_groups': unmatched_groups,
        'unmatched_learned_factors': unmatched_factors,
    }


def evaluate_factor_recovery(groups, corex_model, linear=True,
                             verbose=False, plot=False):
    """Calculate matched primary metrics and directional diagnostics."""
    factor_scores = get_factor_variable_scores(corex_model, linear=linear)
    matched = match_factors_to_groups(groups, factor_scores)
    average_precision = matched['average_precision_matrix']
    cosine_distance = matched['cosine_distance_matrix']

    matched['group_best_average_precision'] = float(
        np.mean(np.max(average_precision, axis=1))
    )
    matched['group_best_cosine_distance'] = float(
        np.mean(np.min(cosine_distance, axis=1))
    )
    matched['factor_best_average_precision'] = float(
        np.mean(np.max(average_precision, axis=0))
    )
    matched['factor_best_cosine_distance'] = float(
        np.mean(np.min(cosine_distance, axis=0))
    )

    if verbose:
        for assignment in matched['assignments']:
            print(
                "Group {true_group} matched factor {learned_factor}: "
                "AP={average_precision:.3f}, cosine distance={cosine_distance:.3f}"
                .format(**assignment)
            )

    if plot:
        num_matches = len(matched['assignments'])
        _, axes = plt.subplots(num_matches, 1, figsize=(8, num_matches))
        axes = np.atleast_1d(axes)
        for axis, assignment in zip(axes, matched['assignments']):
            group = groups[assignment['true_group']]
            factor = factor_scores[assignment['learned_factor']]
            group_total = np.sum(group)
            factor_total = np.sum(factor)
            normalized_group = group / group_total
            normalized_factor = (
                factor / factor_total
                if factor_total != 0
                else np.zeros_like(factor)
            )
            axis.imshow(np.vstack([normalized_group, normalized_factor]))
            axis.set_ylabel(
                f"G{assignment['true_group']} / F{assignment['learned_factor']}"
            )
        plt.show()

    return matched


def factor_recovery_output_values(recovery):
    """Return recovery results in the same order as RECOVERY_COLUMNS."""
    assignment_text = ';'.join(
        f"{item['true_group']}:{item['learned_factor']}"
        for item in recovery['assignments']
    )
    return [
        recovery['matched_average_precision'],
        recovery['matched_adjusted_average_precision'],
        recovery['matched_cosine_distance'],
        recovery['matched_cosine_similarity'],
        recovery['matched_top_k_f1'],
        recovery['matched_nuisance_score_fraction'],
        recovery['matched_top_k_nuisance_fraction'],
        recovery['true_group_recovery_average_precision'],
        recovery['learned_factor_precision_average_precision'],
        recovery['balanced_average_precision'],
        recovery['true_group_recovery_cosine_similarity'],
        recovery['learned_factor_precision_cosine_similarity'],
        recovery['balanced_cosine_similarity'],
        recovery['num_matched_factors'],
        len(recovery['unmatched_true_groups']),
        len(recovery['unmatched_learned_factors']),
        recovery['group_best_average_precision'],
        recovery['group_best_cosine_distance'],
        recovery['factor_best_average_precision'],
        recovery['factor_best_cosine_distance'],
        assignment_text,
    ]


def recovery_result_fields(recovery):
    """Convert recovery results to named output fields."""
    values = factor_recovery_output_values(recovery)
    return dict(zip(RECOVERY_COLUMNS, values))


def validate_factor_matching():
    """Check matching, factor-count penalties, and leakage diagnostics."""
    groups = np.asarray(groups_to_indicator_vectors(CLUSTER_1_GROUPS))

    perfect = match_factors_to_groups(groups, groups.copy())
    assert np.isclose(perfect['matched_average_precision'], 1.0)
    assert np.isclose(perfect['matched_adjusted_average_precision'], 1.0)
    assert np.isclose(perfect['balanced_average_precision'], 1.0)
    assert np.isclose(perfect['balanced_cosine_similarity'], 1.0)
    assert np.isclose(perfect['matched_top_k_f1'], 1.0)
    assert np.isclose(perfect['matched_nuisance_score_fraction'], 0.0)
    assert np.isclose(perfect['matched_top_k_nuisance_fraction'], 0.0)
    assert perfect['unmatched_true_groups'] == []
    assert perfect['unmatched_learned_factors'] == []

    permutation = [3, 1, 4, 0, 2]
    permuted = match_factors_to_groups(groups, groups[permutation])
    assert np.isclose(permuted['matched_average_precision'], 1.0)
    assert np.isclose(permuted['balanced_average_precision'], 1.0)

    missing = match_factors_to_groups(groups, groups[:3])
    assert np.isclose(missing['matched_average_precision'], 1.0)
    assert np.isclose(
        missing['true_group_recovery_average_precision'],
        3 / 5,
    )
    assert np.isclose(
        missing['learned_factor_precision_average_precision'],
        1.0,
    )
    assert len(missing['unmatched_true_groups']) == 2
    assert missing['unmatched_learned_factors'] == []

    extra_scores = np.vstack([groups, groups[:2]])
    extra = match_factors_to_groups(groups, extra_scores)
    assert np.isclose(extra['true_group_recovery_average_precision'], 1.0)
    assert np.isclose(
        extra['learned_factor_precision_average_precision'],
        5 / 7,
    )
    assert extra['unmatched_true_groups'] == []
    assert len(extra['unmatched_learned_factors']) == 2

    duplicate_scores = np.repeat(groups[[0]], 5, axis=0)
    duplicated = match_factors_to_groups(groups, duplicate_scores)
    assert duplicated['balanced_average_precision'] < 1.0
    assert duplicated['true_group_recovery_average_precision'] < 1.0

    extended_groups = np.asarray(groups_to_indicator_vectors(
        CLUSTER_1_GROUPS,
        n_variables=50,
    ))
    extended_perfect = match_factors_to_groups(
        extended_groups,
        extended_groups.copy(),
    )
    assert np.isclose(
        extended_perfect['matched_adjusted_average_precision'],
        1.0,
    )
    assert np.isclose(extended_perfect['matched_top_k_f1'], 1.0)
    assert np.isclose(
        extended_perfect['matched_nuisance_score_fraction'],
        0.0,
    )

    leaky_scores = extended_groups.copy()
    leaky_scores[:, 25:] = 0.5
    leaky = match_factors_to_groups(extended_groups, leaky_scores)
    assert np.isclose(leaky['matched_average_precision'], 1.0)
    assert leaky['matched_nuisance_score_fraction'] > 0.0
    assert np.isclose(leaky['matched_top_k_nuisance_fraction'], 0.0)

    badly_leaked_scores = extended_groups.copy()
    badly_leaked_scores[:, 25:30] = 2.0
    badly_leaked = match_factors_to_groups(
        extended_groups,
        badly_leaked_scores,
    )
    assert badly_leaked['matched_top_k_nuisance_fraction'] > 0.0
    assert badly_leaked['matched_top_k_f1'] < 1.0

    return {
        'perfect_matching_checked': True,
        'factor_permutation_checked': True,
        'missing_factor_penalty_checked': True,
        'extra_factor_penalty_checked': True,
        'duplicate_factor_penalty_checked': True,
        'adjusted_average_precision_checked': True,
        'nuisance_score_leakage_checked': True,
        'top_k_nuisance_leakage_checked': True,
    }
