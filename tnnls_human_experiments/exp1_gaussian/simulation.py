"""Model fitting and simulation orchestration for Gaussian experiments."""

from time import time

import numpy as np
import pandas as pd
from local_corex import BioCorex, LinearCorex

from gaussian_data import N_VARIABLES, gen_clustered_sample, get_true_groups
from partitioning import PARTITION_SEED_CODES, partition_data
from recovery_metrics import evaluate_factor_recovery, recovery_result_fields


ALPHA_LIST = [0.0, 0.5, 1.0]
NUM_FACTOR_LIST = [x for x in range(3, 8)]


def covariance_setup_name(same_dist, disjoint):
    """Return the readable name of one covariance condition."""
    if same_dist and disjoint:
        raise ValueError("same_dist and disjoint cannot both be True.")
    if same_dist:
        return 'homogeneous'
    if disjoint:
        return 'disjoint'
    return 'nondisjoint'


def deterministic_seed(*values):
    """Create one reproducible integer seed from condition identifiers."""
    return int(np.random.SeedSequence(values).generate_state(1)[0])


def fit_corex_model(data, num_factors, estimator):
    """Fit one CorEx estimator and return the fitted model and elapsed time."""
    start_time = time()
    if estimator == 'linear':
        model = LinearCorex(
            n_hidden=num_factors,
            seed=42,
            gaussianize='outliers',
        )
        linear = True
        model_seed = 42
    elif estimator == 'bio':
        model = BioCorex(
            n_hidden=num_factors,
            dim_hidden=2,
            marginal_description='gaussian',
            smooth_marginals=False,
        )
        linear = False
        model_seed = np.nan
    else:
        raise ValueError("estimator must be 'linear' or 'bio'.")

    model.fit_transform(data)
    return model, linear, model_seed, time() - start_time


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


def score_fitted_model(model, linear, groups, row, verbose=False, plot=False):
    """Evaluate one fitted model against one region's true factor groups."""
    recovery = evaluate_factor_recovery(
        groups,
        model,
        linear=linear,
        verbose=verbose,
        plot=plot,
    )
    row.update(recovery_result_fields(recovery))
    return row


def run_simulation(rep,
                   cluster_size=100,
                   alphas=ALPHA_LIST,
                   num_factors=NUM_FACTOR_LIST,
                   estimators=('linear',),
                   partition_methods=('oracle', 'phate', 'random'),
                   disjoint=False,
                   same_dist=False,
                   nuisance_dimensions=0,
                   visualize_vars=False,
                   verbose=False,
                   progress=True):
    """Run all requested methods on shared datasets and partitions.

    One dataset is generated for each alpha. Every requested partition is
    computed once and reused across factor counts and estimators. Each global
    model is fit once and evaluated against both regional ground truths.
    """
    setup = covariance_setup_name(same_dist, disjoint)
    unknown_partitions = set(partition_methods) - set(PARTITION_SEED_CODES)
    if unknown_partitions:
        raise ValueError(f"Unknown partition methods: {sorted(unknown_partitions)}")
    unknown_estimators = set(estimators) - {'linear', 'bio'}
    if unknown_estimators:
        raise ValueError(f"Unknown estimators: {sorted(unknown_estimators)}")

    total_fits = (
        len(alphas)
        * len(num_factors)
        * len(estimators)
        * (1 + 2 * len(partition_methods))
    )
    completed_fits = 0
    rows = []

    for alpha in alphas:
        condition_values = (
            int(rep),
            int(cluster_size),
            int(round(alpha * 1000)),
            int(disjoint),
            int(same_dist),
        )
        data_seed = deterministic_seed(*condition_values, 0)

        generation_start = time()
        cluster_1, cluster_2 = gen_clustered_sample(
            same_dist=same_dist,
            alpha=alpha,
            cluster_size=cluster_size,
            disjoint=disjoint,
            nuisance_dimensions=nuisance_dimensions,
            random_state=data_seed,
        )
        full_data = np.concatenate([cluster_1, cluster_2], axis=0)
        true_labels = np.concatenate([
            np.repeat('clust_1', cluster_size),
            np.repeat('clust_2', cluster_size),
        ])
        generation_time = time() - generation_start

        partitions = {}
        partition_seeds = {}
        for partition_method in partition_methods:
            partition_seed = deterministic_seed(
                *condition_values,
                PARTITION_SEED_CODES[partition_method],
            )
            partition_seeds[partition_method] = partition_seed
            partitions[partition_method] = partition_data(
                full_data,
                true_labels,
                method=partition_method,
                random_state=partition_seed,
            )

        true_groups = {
            cluster: get_true_groups(
                cluster,
                disjoint=disjoint,
                same_dist=same_dist,
                nuisance_dimensions=nuisance_dimensions,
            )
            for cluster in ('clust_1', 'clust_2')
        }

        for factor_count in num_factors:
            for estimator in estimators:
                completed_fits += 1
                if progress:
                    print(
                        f"Fit {completed_fits}/{total_fits}: global {estimator}, "
                        f"alpha={alpha}, factors={factor_count}"
                    )
                model, linear, model_seed, corex_time = fit_corex_model(
                    full_data,
                    factor_count,
                    estimator,
                )
                fit_id = (
                    f"rep{rep}_n{cluster_size}_z{nuisance_dimensions}_"
                    f"a{alpha}_{setup}_"
                    f"m{factor_count}_{estimator}_global"
                )

                for cluster in ('clust_1', 'clust_2'):
                    row = {
                        'rep_num': rep,
                        'cluster_size': cluster_size,
                        'nuisance_dimensions': nuisance_dimensions,
                        'num_variables': N_VARIABLES + nuisance_dimensions,
                        'alpha': alpha,
                        'covariance_setup': setup,
                        'num_factors': factor_count,
                        'estimator': estimator,
                        'scope': 'global',
                        'groundtruth': cluster,
                        'fit_id': fit_id,
                        'data_seed': data_seed,
                        'model_seed': model_seed,
                        'model_sample_size': len(full_data),
                        'data_generation_time': generation_time,
                        'corex_time': corex_time,
                        'total_time': corex_time,
                    }
                    row.update(empty_partition_fields())
                    rows.append(score_fitted_model(
                        model,
                        linear,
                        true_groups[cluster],
                        row,
                        verbose=verbose,
                        plot=visualize_vars,
                    ))

                for partition_method in partition_methods:
                    partition = partitions[partition_method]
                    for cluster in ('clust_1', 'clust_2'):
                        completed_fits += 1
                        if progress:
                            print(
                                f"Fit {completed_fits}/{total_fits}: "
                                f"local {estimator}, "
                                f"partition={partition_method}, region={cluster}, "
                                f"alpha={alpha}, factors={factor_count}"
                            )
                        local_data = full_data[partition['indexes'][cluster]]
                        model, linear, model_seed, corex_time = fit_corex_model(
                            local_data,
                            factor_count,
                            estimator,
                        )
                        fit_id = (
                            f"rep{rep}_n{cluster_size}_z{nuisance_dimensions}_"
                            f"a{alpha}_{setup}_"
                            f"m{factor_count}_{estimator}_"
                            f"{partition_method}_{cluster}"
                        )
                        row = {
                            'rep_num': rep,
                            'cluster_size': cluster_size,
                            'nuisance_dimensions': nuisance_dimensions,
                            'num_variables': N_VARIABLES + nuisance_dimensions,
                            'alpha': alpha,
                            'covariance_setup': setup,
                            'num_factors': factor_count,
                            'estimator': estimator,
                            'scope': 'local',
                            'groundtruth': cluster,
                            'fit_id': fit_id,
                            'data_seed': data_seed,
                            'model_seed': model_seed,
                            'model_sample_size': len(local_data),
                            'data_generation_time': generation_time,
                            'corex_time': corex_time,
                            'total_time': (
                                corex_time + partition['partition_time']
                            ),
                        }
                        row.update(local_partition_fields(
                            partition,
                            cluster,
                            partition_seeds[partition_method],
                        ))
                        rows.append(score_fitted_model(
                            model,
                            linear,
                            true_groups[cluster],
                            row,
                            verbose=verbose,
                            plot=visualize_vars,
                        ))

    return pd.DataFrame(rows)
