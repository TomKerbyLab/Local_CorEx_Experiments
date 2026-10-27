"""Shared bookkeeping helpers for synthetic experiment simulations."""

import numpy as np

from recovery_metrics import evaluate_factor_recovery, recovery_result_fields


def empty_partition_fields():
    """Return partition fields for a global model that uses no partition."""
    return {
        'partition_method': 'none',
        'partition_seed': np.nan,
        'partition_ari': np.nan,
        'partition_purity': np.nan,
        'partition_accuracy': np.nan,
        'region_purity': np.nan,
        'true_1_est_1': np.nan,
        'true_1_est_2': np.nan,
        'true_2_est_1': np.nan,
        'true_2_est_2': np.nan,
        'partition_time': 0.0,
    }


def local_partition_fields(partition, cluster, partition_seed):
    """Return diagnostics for one mapped local region."""
    confusion = partition['confusion_matrix']
    return {
        'partition_method': partition['method'],
        'partition_seed': partition_seed,
        'partition_ari': partition['ari'],
        'partition_purity': partition['purity'],
        'partition_accuracy': partition['mapped_accuracy'],
        'region_purity': partition['region_purity'][cluster],
        'true_1_est_1': int(confusion[0, 0]),
        'true_1_est_2': int(confusion[0, 1]),
        'true_2_est_1': int(confusion[1, 0]),
        'true_2_est_2': int(confusion[1, 1]),
        'partition_time': partition['partition_time'],
    }


def score_fitted_model(model, linear, groups, row, verbose=False, plot=False,
                       nuisance_mask=None, routing_mask=None):
    """Add factor-recovery results to one run-level result row."""
    recovery = evaluate_factor_recovery(
        groups,
        model,
        linear=linear,
        verbose=verbose,
        plot=plot,
        nuisance_mask=nuisance_mask,
        routing_mask=routing_mask,
    )
    row.update(recovery_result_fields(recovery))
    return row
