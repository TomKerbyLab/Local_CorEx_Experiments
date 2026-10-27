"""Global-capacity control for the Gaussian experiments (supplementary).

Apples-to-apples capacity comparison (methodology § capacity control):
every candidate is scored against the SAME concatenated set of cluster
supports — both clusters' ground truths stacked into one matrix — with the
same matching rule (Hungarian, cosine-primary inside the matcher):

- ``global_capacity``: one fit on all observations at each requested
  factor budget M in BUDGETS. The M = 5 global uses the primary grid's
  fit convention unchanged (fixed seed 42 with the nan-divergence retry),
  and the conditions mirror the primary grid's own (setup, n = 500,
  alpha = 0.5, no nuisance) cells, so the M = 5 global is the identical
  model the primary grid scored.
- ``local_pooled``: the two oracle-partition local fits at each budget,
  stacked into one (2M x p) score matrix — the defined local aggregate.

The existing per-cluster recovery in the primary grid remains the primary
workflow comparison; this grid separates CAPACITY (scored factor count)
from LOCALITY (which observations were fitted). The capacity score of a
single local fit against the other cluster's unsupported variables is
deliberately NOT part of this comparison (that is the per-cluster primary
grid's job); a stacked score keeps every candidate's support set identical.
"""

import argparse
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

from corex_fitting import fit_corex_model
from experiment_common import deterministic_seed
from gaussian_data import N_VARIABLES
from partitioning import PARTITION_SEED_CODES, partition_data
from recovery_metrics import (
    RECOVERY_COLUMNS,
    get_factor_variable_scores,
    match_factors_to_groups,
    recovery_result_fields,
)
from result_io import append_run_log, combine_csv_files, write_csv_atomically
from simulation import covariance_setup_name, generate_gaussian_dataset
from simulation_common import empty_partition_fields, local_partition_fields

from run_gaussian_experiments import SETUP_FLAGS  # noqa: E402

CAPACITY_SETUPS = ('nondisjoint', 'disjoint')
CAPACITY_SAMPLE_SIZE = 500
CAPACITY_ALPHAS = (0.5,)
CAPACITY_BUDGETS = (5, 10)
CAPACITY_ESTIMATORS = ('linear',)
CAPACITY_NUISANCE = (0,)
CAPACITY_REPS = 16

# capacity rows carry the primary grid's recovery schema plus these two
# bookkeeping columns (inserted first, so the combined CSVs keep the
# primary metric columns trailing)
CAPACITY_META_COLUMNS = ['capacity_kind', 'budget']

PARTITION_COLUMNS = [
    'partition_method', 'partition_seed', 'partition_ari',
    'partition_purity', 'partition_accuracy', 'region_purity',
    'true_1_est_1', 'true_1_est_2', 'true_2_est_1', 'true_2_est_2',
    'partition_time',
]

BASE_ROW_KEYS = [
    'capacity_kind', 'budget', 'rep_num', 'cluster_size',
    'nuisance_dimensions', 'num_variables', 'alpha', 'covariance_setup',
    'num_factors', 'estimator', 'scope', 'groundtruth', 'fit_id',
    'data_seed', 'model_seed', 'local_model_seeds', 'model_sample_size',
    'data_generation_time', 'corex_time', 'total_time',
]

OUTPUT_COLUMNS = (
    CAPACITY_META_COLUMNS + BASE_ROW_KEYS + PARTITION_COLUMNS
    + list(RECOVERY_COLUMNS)
)


def capacity_recovery(groups, factor_scores, nuisance_mask=None):
    """Matched recovery against one (pooled) truth, primary-grid schema.

    Mirrors evaluate_factor_recovery without the plotting/logging path: the
    directional best-per-group/factor fields need the matching matrices the
    matcher already returns.
    """
    matched = match_factors_to_groups(
        groups, factor_scores, nuisance_mask=nuisance_mask)
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
    return recovery_result_fields(matched), matched


def capacity_condition_rows(rep, setup, cluster_size, alphas=None, budgets=None,
                            estimators=None, nuisance_dimensions=0):
    """One capacity condition: shared data, all capacity fits."""
    setup_flags = SETUP_FLAGS[setup]
    alphas = alphas if alphas is not None else CAPACITY_ALPHAS
    budgets = budgets if budgets is not None else CAPACITY_BUDGETS
    estimators = estimators if estimators is not None else CAPACITY_ESTIMATORS
    rows = []

    for alpha in alphas:
        generation_start = time()
        (full_data, true_labels, true_groups, data_seed,
         condition_values) = generate_gaussian_dataset(
            rep=rep,
            cluster_size=cluster_size,
            alpha=alpha,
            disjoint=setup_flags['disjoint'],
            same_dist=setup_flags['same_dist'],
            nuisance_dimensions=nuisance_dimensions,
        )
        data_generation_time = time() - generation_start
        setup_name = covariance_setup_name(
            setup_flags['same_dist'], setup_flags['disjoint'])

        stacked_groups = np.vstack([
            true_groups[cluster] for cluster in ('clust_1', 'clust_2')])
        nuisance_mask = np.sum(stacked_groups, axis=0) == 0

        oracle_seed = deterministic_seed(
            *condition_values, PARTITION_SEED_CODES['oracle'])
        oracle = partition_data(
            full_data, np.asarray(true_labels), method='oracle',
            random_state=oracle_seed)

        for estimator in estimators:
            for budget in budgets:
                def base(kind):
                    return {
                        'capacity_kind': kind,
                        'budget': budget,
                        'rep_num': rep,
                        'cluster_size': cluster_size,
                        'nuisance_dimensions': nuisance_dimensions,
                        'num_variables': N_VARIABLES + nuisance_dimensions,
                        'alpha': alpha,
                        'covariance_setup': setup_name,
                        'num_factors': budget,
                        'estimator': estimator,
                        'groundtruth': 'pooled',
                        'data_seed': data_seed,
                        'data_generation_time': data_generation_time,
                    }

                # --- global fit at this budget ---
                model, linear, model_seed, corex_time = fit_corex_model(
                    full_data, budget, estimator)
                row = base('global_capacity')
                row.update({
                    'scope': 'global_capacity',
                    'fit_id': (
                        f"rep{rep}_n{cluster_size}_z{nuisance_dimensions}_"
                        f"a{alpha}_{setup_name}_m{budget}_{estimator}_"
                        f"global_capacity"),
                    'model_seed': model_seed,
                    'local_model_seeds': '',
                    'model_sample_size': len(full_data),
                    'corex_time': corex_time,
                    'total_time': corex_time,
                })
                row.update(empty_partition_fields())
                recovery, _matched = capacity_recovery(
                    stacked_groups,
                    get_factor_variable_scores(model, linear=linear),
                    nuisance_mask=nuisance_mask,
                )
                row.update(recovery)
                rows.append(row)

                # --- pooled oracle local fits at this budget ---
                pooled_scores = []
                local_seeds = []
                local_time = 0.0
                for cluster in ('clust_1', 'clust_2'):
                    local_data = full_data[oracle['indexes'][cluster]]
                    model, linear, model_seed, corex_time = fit_corex_model(
                        local_data, budget, estimator)
                    local_time += corex_time
                    local_seeds.append(
                        '' if model_seed != model_seed else str(model_seed))
                    pooled_scores.append(
                        get_factor_variable_scores(model, linear=linear))
                pooled = np.vstack(pooled_scores)
                row = base('local_pooled')
                row.update({
                    'scope': 'local_pooled_capacity',
                    'fit_id': (
                        f"rep{rep}_n{cluster_size}_z{nuisance_dimensions}_"
                        f"a{alpha}_{setup_name}_m{budget}_{estimator}_"
                        f"oracle_pooled"),
                    'model_seed': local_seeds[0],
                    'local_model_seeds': '|'.join(local_seeds),
                    'model_sample_size': sum(
                        len(oracle['indexes'][cluster])
                        for cluster in ('clust_1', 'clust_2')),
                    'corex_time': local_time,
                    'total_time': local_time + oracle['partition_time'],
                    'partition_time': oracle['partition_time'],
                })
                fields = local_partition_fields(oracle, 'clust_1', oracle_seed)
                fields['region_purity'] = float(
                    np.mean([
                        oracle['region_purity']['clust_1'],
                        oracle['region_purity']['clust_2'],
                    ]))
                row.update(fields)
                recovery, _matched = capacity_recovery(
                    stacked_groups, pooled, nuisance_mask=nuisance_mask)
                row.update(recovery)
                rows.append(row)

    return rows


def condition_output_path(output_dir, setup, sample_size,
                          nuisance_dimensions, rep):
    """Return the CSV path for one restartable capacity condition."""
    condition_dir = (
        Path(output_dir)
        / 'global_capacity'
        / setup
        / f'n_{sample_size}'
    )
    if nuisance_dimensions > 0:
        condition_dir = condition_dir / f'nuisance_{nuisance_dimensions}'
    return condition_dir / f'rep_{rep:02d}.csv'


def condition_rows_with_columns(rows):
    """Project rows onto the fixed combined-CSV column order."""
    return [{key: row.get(key, np.nan) for key in OUTPUT_COLUMNS}
            for row in rows]


def run_capacity_experiments(output_dir, repetitions=None, overwrite=False,
                             dry_run=False, alphas=None, budgets=None,
                             estimators=None):
    """Run every capacity condition (restartable, one CSV per condition)."""
    num_repetitions = repetitions or CAPACITY_REPS
    conditions = len(CAPACITY_SETUPS) * num_repetitions
    fits_per_condition = (
        len(CAPACITY_ALPHAS) * len(budgets or CAPACITY_BUDGETS)
        * len(estimators or CAPACITY_ESTIMATORS) * 3)
    print(
        f"Experiment: global_capacity ({conditions} conditions, "
        f"~{conditions * fits_per_condition:,} CorEx fits)"
    )
    if dry_run:
        return None

    output_dir = Path(output_dir)
    start_time = time()
    append_run_log(output_dir, {
        'event': 'run_start',
        'experiment': 'global_capacity',
        'repetitions': num_repetitions,
    })
    conditions_written = 0
    conditions_skipped = 0
    failed_conditions = []
    combined = None
    status = 'completed'
    try:
        for setup in CAPACITY_SETUPS:
            for rep in range(num_repetitions):
                output_path = condition_output_path(
                    output_dir, setup, CAPACITY_SAMPLE_SIZE,
                    CAPACITY_NUISANCE[0], rep)
                if output_path.exists() and not overwrite:
                    conditions_skipped += 1
                    continue
                try:
                    rows = capacity_condition_rows(
                        rep=rep,
                        setup=setup,
                        cluster_size=CAPACITY_SAMPLE_SIZE,
                        alphas=alphas,
                        budgets=budgets,
                        estimators=estimators,
                        nuisance_dimensions=CAPACITY_NUISANCE[0],
                    )
                except Exception as error:
                    print(f"FAILED condition {output_path}: "
                          f"{type(error).__name__}: {error}")
                    failed_conditions.append(output_path)
                    continue
                rows = condition_rows_with_columns(rows)
                write_csv_atomically(
                    pd.DataFrame(rows), output_path)
                conditions_written += 1
                print(f"Completed {output_path}")
        if not failed_conditions:
            combined = combine_results(output_dir)
        if failed_conditions:
            status = 'completed_with_failures'
    except BaseException:
        status = 'failed'
        raise
    finally:
        append_run_log(output_dir, {
            'event': 'run_end',
            'experiment': 'global_capacity',
            'status': status,
            'conditions_written': conditions_written,
            'conditions_skipped': conditions_skipped,
            'elapsed_s': round(time() - start_time, 3),
        })

    if failed_conditions:
        append_run_log(output_dir, {
            'event': 'condition_failures',
            'experiment': 'global_capacity',
            'n_failed': len(failed_conditions),
            'conditions': [str(path) for path in failed_conditions],
        })
    if combined is not None:
        print(f"Combined results: {combined}")
    return combined


def combine_results(output_dir):
    """Combine all completed capacity condition files."""
    experiment_dir = Path(output_dir) / 'global_capacity'
    result_files = sorted(experiment_dir.glob('**/rep_*.csv'))
    combined = combine_csv_files(result_files)
    if combined is None:
        return None
    combined_path = experiment_dir / 'global_capacity_combined.csv'
    write_csv_atomically(combined, combined_path)
    return combined_path


def main():
    parser = argparse.ArgumentParser(
        description='Run the Gaussian global-capacity control grid.',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('capacity_data'),
        help='Directory for capacity condition files and combined results.',
    )
    parser.add_argument(
        '--repetitions', type=int,
        help='Override the default repetition count (smoke tests).',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Rerun conditions whose CSV files already exist.',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print the condition and fit counts without running.',
    )
    args = parser.parse_args()
    run_capacity_experiments(
        output_dir=args.output_dir,
        repetitions=args.repetitions,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()