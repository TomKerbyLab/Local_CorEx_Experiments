"""Model fitting and simulation orchestration for Gaussian experiments."""

from time import time

import numpy as np
import pandas as pd

from corex_fitting import fit_corex_model
from experiment_common import deterministic_seed
from gaussian_data import N_VARIABLES, gen_clustered_sample, get_true_groups
from partitioning import PARTITION_SEED_CODES, partition_data
from simulation_common import (
    empty_partition_fields,
    local_partition_fields,
    score_fitted_model,
)


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


def generate_gaussian_dataset(rep, cluster_size, alpha, disjoint=False,
                              same_dist=False, nuisance_dimensions=0):
    """One generated dataset with its ground truth (data separation helper).

    The data seed excludes the nuisance count, so the signal data is PAIRED
    across nuisance sweeps (nuisance is drawn from the continuing stream
    after the signal draws). Returns (full_data, true_labels, true_groups,
    data_seed, condition_values); condition_values feeds the partition
    seeds, which the caller derives from PARTITION_SEED_CODES.
    """
    condition_values = (
        int(rep),
        int(cluster_size),
        int(round(alpha * 1000)),
        int(disjoint),
        int(same_dist),
    )
    data_seed = deterministic_seed(*condition_values, 0)
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
    true_groups = {
        cluster: get_true_groups(
            cluster,
            disjoint=disjoint,
            same_dist=same_dist,
            nuisance_dimensions=nuisance_dimensions,
        )
        for cluster in ('clust_1', 'clust_2')
    }
    return full_data, true_labels, true_groups, data_seed, condition_values


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
        generation_start = time()
        (full_data, true_labels, true_groups, data_seed,
         condition_values) = generate_gaussian_dataset(
            rep=rep,
            cluster_size=cluster_size,
            alpha=alpha,
            disjoint=disjoint,
            same_dist=same_dist,
            nuisance_dimensions=nuisance_dimensions,
        )
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
