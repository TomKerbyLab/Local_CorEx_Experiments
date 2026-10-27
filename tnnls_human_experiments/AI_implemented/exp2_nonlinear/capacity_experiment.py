"""Global-capacity control for the nonlinear experiments.

Apples-to-apples capacity comparison per the methodology (§ capacity
control): all candidates are scored against the SAME concatenated set of
region-specific supports — ground truths of BOTH regions stacked — with
the same matching rule (Hungarian on the primary objective):

- ``global`` fits (all observations) at each requested factor budget M
  (2, 4, 8). The M = 4 global fit reuses the primary grid's global fit
  seed, so it is the identical model the primary grid scored.
- ``local_pooled``: the two ORACLE local fits (one per true region) at the
  standard factor count M = 4, stacked into one (2M x p) score matrix —
  the explicitly defined local aggregate for this comparison.

The existing per-region recovery in the primary grid remains the primary
workflow comparison; this grid only separates CAPACITY (scored factor
count) from LOCALITY (which observations were fitted).
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

import nonlinear_data
import recovery_metrics
from config import CAPACITY_CONFIG, CORE_LEVELS, DEFAULTS, ROOT_SEED_BASE
from corex_fitting import factor_variable_scores
from seeds import derived_seed

# result_io lives in ../../synthetic (append keeps sibling modules from
# shadowing this directory's modules)
sys.path.append(str(Path(__file__).resolve().parents[2] / 'synthetic'))
from result_io import append_run_log, combine_csv_files, write_csv_atomically  # noqa: E402

from run_nonlinear_experiments import (  # noqa: E402
    fit_matrices,
    fit_one_model,
    generate_condition_data,
    save_condition_metadata,
    variant_specs,
)

CAPACITY_COLUMNS = [
    'generator', 'n_per_region', 'alpha', 'noise', 'n_nuisance', 'root_seed',
    'n_factors', 'capacity_kind', 'budget', 'estimator', 'corex_input',
    'region', 'true_group', 'learned_factor', 'factor_tc', 'cosine', 'auprc',
    'topd_precision', 'topd_recall', 'topd_f1', 'topd_jaccard',
    'support_size', 'support_density',
    'support_precision', 'support_recall', 'support_f1', 'support_jaccard',
    'support_degenerate',
]

CAPACITY_FIT_COLUMNS = [
    'generator', 'n_per_region', 'alpha', 'noise', 'n_nuisance', 'root_seed',
    'capacity_kind', 'budget', 'n_factors', 'estimator', 'corex_input',
    'model_seed', 'fit_status', 'note', 'runtime_s', 'corex_converged',
    'n_factors_valid', 'total_tc',
]

BUDGETS = (2, 4, 8)


def capacity_conditions(repetitions=16):
    """One capacity condition per generator per replicate.

    Conditions mirror the primary grid's cell (n = 500, alpha = 0.5, the
    lowest nonzero noise level, core nuisance count) — present in BOTH core
    grids — so the pooled local fits and the M = 4 global fit are the exact
    models the primary grid scored (same data seeds and same fit seeds).
    """
    conditions = []
    for rep in range(repetitions):
        for generator in ('parity', 'sine_sum'):
            conditions.append({
                'generator': generator,
                'n_per_region': 500,
                'alpha': 0.5,
                'noise': 0.05,
                'n_nuisance': CORE_LEVELS['nuisance'],
                'root_seed': ROOT_SEED_BASE[generator] + rep,
            })
    return conditions


def combined_truth(dataset):
    """Ground truths of BOTH regions stacked, in a fixed order.

    Returns a list of (region, group, mask) triples — (0, 1) .. (0, N)
    then (1, 1) .. (1, N) — and the stacked truth matrix used in matching.
    """
    truth_rows = [
        (region, group, dataset['true_groups'][(region, group)])
        for region in (0, 1)
        for group in range(1, nonlinear_data.N_GROUPS + 1)
    ]
    truth = np.vstack([mask for _, _, mask in truth_rows])
    return truth_rows, truth


def score_fit(base_row, association, tcs, truth_rows, truth):
    """One matches-table row per matched (factor, combined group) pair.

    Same convention as score_region_fit: unmatched groups and factors get
    no row (the one-to-one constraint is what penalizes them downstream).
    """
    recovery = recovery_metrics.match_factors_to_groups(association, truth)
    rows = []
    for assignment in recovery['assignments']:
        position = int(assignment['true_group'])
        region, group, _mask = truth_rows[position]
        rows.append({
            **base_row,
            'region': region,
            'true_group': group,
            'learned_factor': int(assignment['learned_factor']),
            'factor_tc': float(tcs[assignment['learned_factor']]),
            'cosine': assignment['cosine_similarity'],
            'auprc': assignment['average_precision'],
            'topd_precision': assignment['top_d_precision'],
            'topd_recall': assignment['top_d_recall'],
            'topd_f1': assignment['top_d_f1'],
            'topd_jaccard': assignment['top_d_jaccard'],
            'support_size': assignment['support_size'],
            'support_density': assignment['support_density'],
            'support_precision': assignment['support_precision'],
            'support_recall': assignment['support_recall'],
            'support_f1': assignment['support_f1'],
            'support_jaccard': assignment['support_jaccard'],
            'support_degenerate': assignment['support_degenerate'],
        })
    return rows


def emit_fit(base, kind, budget, spec, model_seed, fit_fields):
    return {
        **base, 'capacity_kind': kind, 'budget': budget,
        'n_factors': budget, 'estimator': spec['estimator'],
        'corex_input': spec['name'], 'model_seed': model_seed,
        **fit_fields,
    }


def run_capacity_condition(condition, config, defaults):
    """Run one capacity condition: global budgets and the pooled local fits.

    Returns (fit_rows, match_rows): one per attempted fit (failures
    recorded, never raised) and one per matched (factor, combined group)
    pair respectively.
    """
    dataset = generate_condition_data(condition)
    specs = variant_specs(condition['generator'], config)
    matrices = fit_matrices(dataset, specs, defaults)
    truth_rows, truth = combined_truth(dataset)

    base = {key: condition[key] for key in (
        'generator', 'n_per_region', 'alpha', 'noise', 'n_nuisance',
        'root_seed')}
    local_budget = defaults['n_factors']

    fit_rows = []
    match_rows = []

    for spec in specs:
        matrix = matrices[spec['name']]['matrix']

        # --- global fits at every requested budget ---
        for budget in BUDGETS:
            if budget == local_budget:
                model_seed = derived_seed(
                    condition['root_seed'], 'corex', 0, 'global', 'global',
                    spec['estimator'], spec['name'])
            else:
                model_seed = derived_seed(
                    condition['root_seed'], 'capacity', budget,
                    spec['estimator'], spec['name'])
            model, linear, fit_fields = fit_one_model(
                matrix, spec['estimator'], budget, model_seed, defaults,
                marginal=spec['marginal'])
            fit_rows.append(emit_fit(
                base, 'global', budget, spec, model_seed, fit_fields))
            if model is None:
                continue
            association = np.maximum(
                factor_variable_scores(model, linear=linear), 0.0)
            tcs = np.asarray(model.tcs, dtype=float)
            scored_base = {
                **base, 'capacity_kind': 'global', 'budget': budget,
                'n_factors': budget, 'estimator': spec['estimator'],
                'corex_input': spec['name'], 'tcs': tcs,
            }
            match_rows.extend(score_fit(
                scored_base, association, tcs, truth_rows, truth))

        # --- pooled local aggregate: the two oracle local fits, stacked ---
        pooled_associations = []
        pooled_tcs = []
        for region in (0, 1):
            local_rows = np.flatnonzero(dataset['region'] == region)
            # seed args must match run_nonlinear_experiments.run_condition
            # so these are the primary grid's oracle models
            oracle_seed = derived_seed(
                condition['root_seed'], 'corex', 0, 'oracle',
                f'region{region}', spec['estimator'], spec['name'])
            model, linear, fit_fields = fit_one_model(
                matrix[local_rows, :], spec['estimator'], local_budget,
                oracle_seed, defaults, marginal=spec['marginal'])
            fit_rows.append(emit_fit(
                base, 'local_pooled', local_budget, spec, oracle_seed,
                fit_fields))
            if model is None:
                pooled_associations = None
                break
            pooled_associations.append(np.maximum(
                factor_variable_scores(model, linear=linear), 0.0))
            pooled_tcs.append(np.asarray(model.tcs, dtype=float))

        if pooled_associations is None:
            continue  # pooled scoring impossible this condition
        association = np.vstack(pooled_associations)
        tcs = np.concatenate(pooled_tcs)
        scored_base = {
            **base, 'capacity_kind': 'local_pooled', 'budget': local_budget,
            'n_factors': local_budget, 'estimator': spec['estimator'],
            'corex_input': spec['name'], 'tcs': tcs,
        }
        match_rows.extend(score_fit(
            scored_base, association, tcs, truth_rows, truth))

    return fit_rows, match_rows


def condition_output_paths(output_dir, condition):
    """CSV paths for one restartable capacity condition."""
    condition_dir = (
        Path(output_dir) / 'global_capacity' / condition['generator'])
    rep = condition['root_seed']
    return {
        'fits': condition_dir / f'rep_{rep:05d}_fits.csv',
        'matches': condition_dir / f'rep_{rep:05d}_matches.csv',
    }


def run_capacity_experiments(output_dir, repetitions=None, overwrite=False,
                             dry_run=False):
    """Run every capacity condition (restartable, one pair of files each)."""
    repetitions = repetitions or 16
    conditions = capacity_conditions(repetitions)
    fits_per_condition = (
        len(variant_specs('parity', CAPACITY_CONFIG))
        * (len(BUDGETS) + 2))
    print(f"Capacity grid: {len(conditions)} conditions, "
          f"~{fits_per_condition} fits each")
    if dry_run:
        return None

    output_dir = Path(output_dir)
    append_run_log(output_dir, {
        'event': 'run_start', 'experiment': 'global_capacity',
        'repetitions': repetitions,
    })
    conditions_written = 0
    conditions_skipped = 0
    conditions_failed = 0
    status = 'completed'
    started_at = time.time()
    try:
        for condition in conditions:
            paths = condition_output_paths(output_dir, condition)
            if (paths['fits'].exists() and paths['matches'].exists()
                    and not overwrite):
                conditions_skipped += 1
                continue
            conditions_written += 1
            try:
                fit_rows, match_rows = run_capacity_condition(
                    condition, CAPACITY_CONFIG, DEFAULTS)
                save_condition_metadata(condition, CAPACITY_CONFIG, DEFAULTS,
                                        {'runs': paths['fits']})
            except Exception as error:
                conditions_failed += 1
                print(f"FAILED condition {paths['fits']}: "
                      f"{type(error).__name__}: {error}")
                continue
            write_csv_atomically(
                pd.DataFrame(fit_rows, columns=CAPACITY_FIT_COLUMNS),
                paths['fits'])
            write_csv_atomically(
                pd.DataFrame(match_rows, columns=CAPACITY_COLUMNS),
                paths['matches'])
        combined = combine_results(output_dir)
    except BaseException:
        status = 'failed'
        raise
    finally:
        append_run_log(output_dir, {
            'event': 'run_end', 'experiment': 'global_capacity',
            'status': status,
            'conditions_written': conditions_written,
            'conditions_skipped': conditions_skipped,
            'conditions_failed': conditions_failed,
            'elapsed_s': round(time.time() - started_at, 3),
        })
    print(f"Capacity grid done: {conditions_written} written, "
          f"{conditions_skipped} skipped, {conditions_failed} failed; "
          f"combined: {combined}")
    return combined


def combine_results(output_dir):
    """Combine all capacity condition files into the shared-level CSVs."""
    experiment_dir = Path(output_dir) / 'global_capacity'
    combined_paths = {}
    for kind, pattern in (('fits', '**/rep_*_fits.csv'),
                          ('matches', '**/rep_*_matches.csv')):
        result_files = sorted(experiment_dir.glob(pattern))
        combined = combine_csv_files(result_files)
        if combined is None:
            continue
        combined_path = experiment_dir / f'global_capacity_{kind}_combined.csv'
        write_csv_atomically(combined, combined_path)
        combined_paths[kind] = combined_path
    return combined_paths


def main():
    parser = argparse.ArgumentParser(
        description='Run the nonlinear global-capacity control grid.',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('nonlinear_results'),
        help='Directory for condition files and combined results.',
    )
    parser.add_argument(
        '--repetitions', type=int,
        help='Override the default repetition count.',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Rerun conditions whose CSV files already exist.',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print counts without running.',
    )
    args = parser.parse_args()
    run_capacity_experiments(
        output_dir=args.output_dir, repetitions=args.repetitions,
        overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == '__main__':
    main()