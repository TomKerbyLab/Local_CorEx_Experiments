"""Factor matching and recovery metrics for Local CorEx experiments.

Matching objective: Hungarian assignment on maximum COSINE similarity
(the decided primary rule, harmonized with the nonlinear experiments);
AUPRC stays computed and reported at the matched pairs as the
experiment-specific ranking metric. Degenerate factors (total nonnegative
score <= DEGENERATE_MI_SUM) score an empty top-k support: top-k
precision/recall/F1-like metrics 0, leakage fractions 0, cosine 0, raw
AUPRC left at the group prevalence, adjusted AUPRC 0. They enter the
primary support-recovery averages as zeros; conditional-on-valid averages
are reported alongside (num_top_k_valid / matched_top_k_f1_valid).
Remaining score ties are broken by neutral tie handling (a seeded column
permutation) instead of the stable sort's lower-index rule, which would
award an all-zero factor a perfect top-k result on the first true group.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from sklearn.metrics import average_precision_score

DEGENERATE_MI_SUM = 1e-12
DEFAULT_TIE_SEED = 0  # fixed neutral tie order: reproducible, layout-free

RECOVERY_COLUMNS = [
    'matched_average_precision',
    'matched_adjusted_average_precision',
    'matched_cosine_distance',
    'matched_cosine_similarity',
    'matched_top_k_f1',
    'matched_top_k_f1_valid',
    'num_top_k_valid',
    'matched_top_k_degenerate_fraction',
    'matched_nuisance_score_fraction',
    'matched_top_k_nuisance_fraction',
    'matched_routing_score_fraction',
    'matched_top_k_routing_fraction',
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


def _descending_order(scores, tie_seed=DEFAULT_TIE_SEED):
    """Indexes by descending score with neutral tie handling.

    Score ties are broken by a seeded permutation of the columns before
    the stable descending sort, so the tie order has no relation to the
    variable layout (the lower-index stable tie-break instead hands an
    all-tied — e.g. all-zero — factor the leading true group as its top-k
    set). ``tie_seed=None`` keeps the lower-index stable rule.
    """
    scores = np.asarray(scores, dtype=float)
    if tie_seed is None:
        return np.argsort(-scores, kind='stable')
    rng = np.random.default_rng(tie_seed)
    column_order = rng.permutation(len(scores))
    ordered_positions = np.argsort(
        -scores[column_order], kind='stable')
    return column_order[ordered_positions]


def _pair_tie_seed(tie_seed, group_index, factor_index):
    """Deterministic per-assignment tie seed (varies across pairs)."""
    if tie_seed is None:
        return None
    return int(np.random.default_rng(
        (int(tie_seed), int(group_index), int(factor_index))
    ).integers(0, 2**31 - 1))


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


def _validate_role_mask(mask, n_variables, name):
    """Return a Boolean variable-role mask with the expected length."""
    if mask is None:
        return np.zeros(n_variables, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (n_variables,):
        raise ValueError(
            f"{name} must have shape ({n_variables},), found {mask.shape}."
        )
    return mask


def matched_support_diagnostics(groups, factor_scores,
                                group_indexes, factor_indexes,
                                nuisance_mask=None, routing_mask=None,
                                tie_seed=DEFAULT_TIE_SEED):
    """Measure support recovery and role-specific leakage for matched pairs.

    A factor whose total nonnegative score is at most DEGENERATE_MI_SUM is
    degenerate: its predicted top-k support is EMPTY and every top-k and
    leakage quantity is 0 (cosine is 0 upstream, raw AUPRC stays at the
    group prevalence). Non-degenerate top-k sets use the neutral tie order.
    """
    groups = np.asarray(groups, dtype=float)
    factor_scores = np.asarray(factor_scores, dtype=float)
    if groups.ndim != 2 or factor_scores.ndim != 2:
        raise ValueError("Groups and factor scores must both be matrices.")
    n_variables = groups.shape[1]
    if factor_scores.shape[1] != n_variables:
        raise ValueError("Groups and factor scores must have matching lengths.")

    if nuisance_mask is None:
        nuisance_mask = np.sum(groups, axis=0) == 0
    else:
        nuisance_mask = _validate_role_mask(
            nuisance_mask,
            n_variables,
            'nuisance_mask',
        )
    routing_mask = _validate_role_mask(
        routing_mask,
        n_variables,
        'routing_mask',
    )
    nuisance_mask = nuisance_mask & ~routing_mask

    top_k_f1 = []
    top_k_degenerate = []
    nuisance_score_fraction = []
    top_k_nuisance_fraction = []
    routing_score_fraction = []
    top_k_routing_fraction = []

    for group_index, factor_index in zip(group_indexes, factor_indexes):
        group = groups[group_index]
        factor = factor_scores[factor_index]
        group_size = int(np.sum(group > 0))
        nonnegative_scores = np.maximum(factor, 0.0)
        total_score = float(nonnegative_scores.sum())
        if total_score <= DEGENERATE_MI_SUM:
            # empty predicted support: zeros for every top-k quantity
            top_k_degenerate.append(True)
            top_k_f1.append(0.0)
            top_k_nuisance_fraction.append(0.0)
            top_k_routing_fraction.append(0.0)
            nuisance_score_fraction.append(0.0)
            routing_score_fraction.append(0.0)
            continue
        top_k_degenerate.append(False)
        top_indexes = _descending_order(
            factor,
            _pair_tie_seed(tie_seed, group_index, factor_index),
        )[:group_size]
        recovered_count = np.sum(group[top_indexes] > 0)

        # Both the true and estimated supports contain k variables, so
        # precision, recall, and F1 all equal recovered_count / group_size.
        top_k_f1.append(float(recovered_count / group_size))
        top_k_nuisance_fraction.append(float(
            np.mean(nuisance_mask[top_indexes])
        ))
        top_k_routing_fraction.append(float(
            np.mean(routing_mask[top_indexes])
        ))

        if total_score == 0:
            nuisance_score_fraction.append(0.0)
            routing_score_fraction.append(0.0)
        else:
            nuisance_score_fraction.append(float(
                np.sum(nonnegative_scores[nuisance_mask]) / total_score
            ))
            routing_score_fraction.append(float(
                np.sum(nonnegative_scores[routing_mask]) / total_score
            ))

    return {
        'top_k_f1': np.asarray(top_k_f1),
        'top_k_degenerate': np.asarray(top_k_degenerate),
        'nuisance_score_fraction': np.asarray(nuisance_score_fraction),
        'top_k_nuisance_fraction': np.asarray(top_k_nuisance_fraction),
        'routing_score_fraction': np.asarray(routing_score_fraction),
        'top_k_routing_fraction': np.asarray(top_k_routing_fraction),
    }


def harmonic_mean(first, second):
    """Calculate a harmonic mean, returning zero when both inputs are zero."""
    if first + second == 0:
        return 0.0
    return 2 * first * second / (first + second)


def match_factors_to_groups(groups, factor_scores, nuisance_mask=None,
                            routing_mask=None, tie_seed=DEFAULT_TIE_SEED):
    """Match factors to groups once on maximum cosine similarity.

    The Hungarian assignment maximizes cosine similarity — the decided
    primary objective, harmonized with the nonlinear experiments; AUPRC
    remains computed and reported at the matched pairs but never drives
    the assignment. The assignment is reused for every other metric.
    Recovery divides the matched total by the number of true groups, which
    penalizes missing groups. Precision divides by the number of learned
    factors, which penalizes extra factors. Their harmonic mean gives a
    balanced summary. Degenerate factors (total nonnegative score at most
    DEGENERATE_MI_SUM) score empty top-k supports (zeros in the primary
    averages; conditional-on-valid averages reported alongside).
    """
    (average_precision,
     adjusted_average_precision,
     cosine_distance) = create_factor_score_matrices(groups, factor_scores)

    # cosine is the primary objective: minimize total cosine distance
    # (equivalently maximize total similarity) under a one-to-one pairing
    group_indexes, factor_indexes = linear_sum_assignment(cosine_distance)
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
        nuisance_mask=nuisance_mask,
        routing_mask=routing_mask,
        tie_seed=tie_seed,
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
            'top_k_degenerate': bool(top_k_degenerate),
            'nuisance_score_fraction': float(nuisance_fraction),
            'top_k_nuisance_fraction': float(top_k_nuisance),
            'routing_score_fraction': float(routing_fraction),
            'top_k_routing_fraction': float(top_k_routing),
        }
        for (group_index, factor_index, ap_score, adjusted_ap_score,
             cosine_score, top_k_f1, top_k_degenerate, nuisance_fraction,
             top_k_nuisance, routing_fraction, top_k_routing) in zip(
            group_indexes,
            factor_indexes,
            matched_average_precision,
            matched_adjusted_average_precision,
            matched_cosine_distance,
            support_diagnostics['top_k_f1'],
            support_diagnostics['top_k_degenerate'],
            support_diagnostics['nuisance_score_fraction'],
            support_diagnostics['top_k_nuisance_fraction'],
            support_diagnostics['routing_score_fraction'],
            support_diagnostics['top_k_routing_fraction'],
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
        'matched_top_k_f1_valid': (
            float(np.mean(support_diagnostics['top_k_f1'][
                ~support_diagnostics['top_k_degenerate']
            ]))
            if not support_diagnostics['top_k_degenerate'].all()
            else float('nan')),
        'num_top_k_valid': int(
            (~support_diagnostics['top_k_degenerate']).sum()),
        'matched_top_k_degenerate_fraction': float(np.mean(
            support_diagnostics['top_k_degenerate'])),
        'matched_nuisance_score_fraction': float(np.mean(
            support_diagnostics['nuisance_score_fraction']
        )),
        'matched_top_k_nuisance_fraction': float(np.mean(
            support_diagnostics['top_k_nuisance_fraction']
        )),
        'matched_routing_score_fraction': float(np.mean(
            support_diagnostics['routing_score_fraction']
        )),
        'matched_top_k_routing_fraction': float(np.mean(
            support_diagnostics['top_k_routing_fraction']
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
                             verbose=False, plot=False, nuisance_mask=None,
                             routing_mask=None):
    """Calculate matched primary metrics and directional diagnostics."""
    factor_scores = get_factor_variable_scores(corex_model, linear=linear)
    matched = match_factors_to_groups(
        groups,
        factor_scores,
        nuisance_mask=nuisance_mask,
        routing_mask=routing_mask,
    )
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
        recovery['matched_top_k_f1_valid'],
        recovery['num_top_k_valid'],
        recovery['matched_top_k_degenerate_fraction'],
        recovery['matched_nuisance_score_fraction'],
        recovery['matched_top_k_nuisance_fraction'],
        recovery['matched_routing_score_fraction'],
        recovery['matched_top_k_routing_fraction'],
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
    groups = np.zeros((5, 25))
    for group_index in range(5):
        start = 5 * group_index
        groups[group_index, start:start + 5] = 1.0

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

    # the assignment objective is cosine similarity, not AUPRC: in this
    # two-group example the objectives pick DIFFERENT pairings (margins
    # are strict on both sides) and the code must follow cosine
    conflict_groups = np.zeros((2, 5))
    conflict_groups[0, :3] = 1.0
    conflict_groups[1, 3:] = 1.0
    conflict_factor_a = np.array([1.06, 0.69, 4.92, 0.01, 1.83])
    conflict_factor_b = np.array([0.29, 3.2, 0.23, 0.34, 0.4])
    conflict = match_factors_to_groups(
        conflict_groups, np.vstack([conflict_factor_a, conflict_factor_b]))
    conflict_pairs = sorted(
        (item['true_group'], item['learned_factor'])
        for item in conflict['assignments']
    )
    assert conflict_pairs == [(0, 1), (1, 0)], conflict_pairs
    ap_optimum = linear_sum_assignment(-np.array([
        [0.8056, 0.7], [0.45, 0.5833]]))[1]
    assert not np.array_equal(ap_optimum, [1, 0])  # AP would pair (0,0),(1,1)

    # degenerate Gaussian factor: empty top-k support and the decided
    # per-quantity semantics (raw AUPRC = prevalence, cosine = 0)
    degenerate_scores = np.zeros((1, 25))
    degenerate = match_factors_to_groups(groups, degenerate_scores)
    degenerate_assignment = degenerate['assignments'][0]
    assert degenerate_assignment['top_k_degenerate'] is True
    assert degenerate_assignment['top_k_f1'] == 0.0
    assert np.isclose(degenerate['matched_cosine_similarity'], 0.0)
    assert np.isclose(degenerate_assignment['average_precision'], 5 / 25)
    assert np.isclose(
        degenerate_assignment['adjusted_average_precision'], 0.0)
    assert degenerate['matched_top_k_f1'] == 0.0
    assert np.isnan(degenerate['matched_top_k_f1_valid'])
    assert degenerate['num_top_k_valid'] == 0

    # neutral tie handling: with ALL scores tied, the legacy lower-index
    # rule hands every factor the leading group (matched top-k), while the
    # seeded neutral rule spreads the assignments; both orders reproduce
    tied_scores = np.ones((5, 25))  # all-tied: every ordering is a tie case
    legacy_tied = match_factors_to_groups(groups, tied_scores, tie_seed=None)
    assert legacy_tied['matched_top_k_f1'] == 0.2  # all matched to group 0
    neutral_tied = match_factors_to_groups(groups, tied_scores)
    assert all(assignment['top_k_f1'] < 1.0
               for assignment in neutral_tied['assignments'])
    repeat_tied = match_factors_to_groups(groups, tied_scores)
    assert repeat_tied['matched_top_k_f1'] == neutral_tied['matched_top_k_f1']

    # mixed valid + degenerate: the zero enters the primary average, the
    # conditional-on-valid average excludes it
    mixed_scores = groups.copy()
    mixed_scores[1] = 0.0
    mixed = match_factors_to_groups(groups, mixed_scores)
    assert np.isclose(mixed['matched_top_k_f1'], 4 / 5)
    assert np.isclose(mixed['matched_top_k_f1_valid'], 1.0)
    assert mixed['num_top_k_valid'] == 4
    assert np.isclose(mixed['matched_top_k_degenerate_fraction'], 0.2)

    extended_groups = np.pad(groups, ((0, 0), (0, 25)))
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
        'cosine_primary_assignment_checked': True,
        'degenerate_semantics_checked': True,
        'neutral_tie_handling_checked': True,
    }
