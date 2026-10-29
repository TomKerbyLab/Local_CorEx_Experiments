"""Run the Experiment 3 redundancy-vs-synergy diagnostics (spec 03,
redesigned per paper/exp3.md).

Structure mirrors ../AI_implemented/exp2_nonlinear/run_nonlinear_experiments.py:
every condition (one replicate of one data-generating configuration) is
written to its own CSV files and completed conditions are skipped by
default, so interrupted runs are restartable, and 'k/N' shard invocations
reproduce the serial run exactly.

Per condition (paper exp3.md §E3.6-§E3.7):

- one dataset: dependent group (d columns) + 5d nuisance columns;
- ground-truth-side quantities (plug-in/MM TC/DTC/Omega on the true group,
  exact population TC/DTC/Omega/pairwise-MI) are computed BEFORE the fit and
  are stored even when the fit fails (``fit_status = failed_fit``: a failed
  fit costs none of the replicate's information content);
- one Bio CorEx fit at 3 requested factors; per-FACTOR rows preserve the
  full fit output (the reference implementation's top-factor-only records
  dropped per-factor recovery); primary factor selection is by explained TC
  only, with neutral tie handling (§E3.7);
- the two-stage demonstration: the top-TC factor's top-d discovered support
  is frozen, then Omega is estimated on exactly those columns of a FRESH
  characterization sample (independent derived seed, n_char observations),
  alongside Omega on the true group's columns of the same sample.

Two arms (§E3.5): 'core' configures dim_hidden = 2^(d-1) (at least
sufficient for parity, overspecified for redundancy — capacity cannot
confound the generator comparison); 'capacity_binary' configures
dim_hidden = 2 (the standard small cardinality: a practical-estimator
diagnostic, never pooled with the main arm). The flip rates come from the
frozen equal-TC calibration (target 0.5 bits, root-finding per §E3.4).
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# result_io lives in ../gaussian; APPEND (not insert) so the sibling
# modules there can never shadow this directory's same-named modules
sys.path.append(str(Path(__file__).resolve().parents[1] / 'gaussian'))
from result_io import (  # noqa: E402
    append_run_log,
    combine_csv_files,
    write_csv_atomically,
)

from redundancy_data import (  # noqa: E402
    GEN_REDUNDANT,
    GEN_SYNERGISTIC,
    GENERATORS,
    NUISANCE_MULTIPLIER,
    TC_TARGET,
    calibrate,
    generate_characterization_sample,
    generate_dataset,
    population_quantities,
)
from information_estimates import group_estimates  # noqa: E402
from recovery_metrics import (  # noqa: E402
    DEFAULT_TIE_SEED,
    SUPPORT_FRACTION,
    descending_order,
    auprc,
    cumulative_mi_support,
    nuisance_false_positive,
    support_vs_truth_metrics,
    top_d_metrics,
)
from seeds import derived_seed  # noqa: E402
from corex_fitting import factor_variable_scores, fit_bio_corex  # noqa: E402

ROOT_SEED_BASE = 4100

CONDITION_COLUMNS = (
    'generator', 'd', 'n_samples', 'dim_hidden', 'n_repeat', 'root_seed',
)

# fixed schemas: written per condition, combined by --combine-only
INFO_COLUMNS = (
    'generator', 'd', 'n_samples', 'dim_hidden', 'n_repeat', 'root_seed',
    'eta', 'n_nuisance', 'tc_target',
    'tc_plugin', 'dtc_plugin', 'omega_plugin', 'tc_mm', 'dtc_mm',
    'omega_mm', 'tc_population', 'dtc_population', 'omega_population',
    'mean_pmi_population', 'top_factor_index', 'top_factor_tc', 'total_tc',
    'n_factors_valid', 'n_negative_mi', 'n_factors_requested', 'converged',
    'fit_runtime_s', 'fit_status', 'note', 'support_discovered',
    'support_size', 'support_matches_group', 'omega_on_support_char',
    'omega_on_support_char_mm', 'omega_true_group_char', 'n_char',
)

FACTOR_COLUMNS = (
    'generator', 'd', 'n_samples', 'dim_hidden', 'n_repeat', 'root_seed',
    'eta', 'n_nuisance', 'factor_index', 'factor_tc', 'is_top_tc_factor',
    'auprc', 'topd_precision', 'topd_recall', 'topd_f1', 'topd_jaccard',
    'nfp_topd', 'support_size', 'support_density', 'support_f1',
    'nfp_support', 'degenerate_factor', 'fit_status',
)

# ---------------------------------------------------------------------------
# Frozen equal-TC calibration (§E3.4): deterministic, cached, frozen by
# construction before any replicate — it is a pure function of the target.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def frozen_calibration():
    """{(d, generator): eta} from the calibration at TC_TARGET."""
    return {
        (row['d'], row['generator']): row['eta']
        for row in calibrate(TC_TARGET)
    }


def eta_for(generator, d):
    return frozen_calibration()[d, generator]


# ---------------------------------------------------------------------------
# Fit-and-score defaults and experiment table (grids live in the builders)
# ---------------------------------------------------------------------------

DEFAULTS = {
    'n_factors': 3,
    'support_fraction': SUPPORT_FRACTION,
    'bio_max_iter': 200,
    'n_char': 100_000,          # fresh characterization sample size (§E3.7)
    'tie_seed': DEFAULT_TIE_SEED,
}


def dim_hidden_for(arm, d):
    """Configured latent states per factor (§E3.5)."""
    if arm == 'sufficient':
        return 2 ** (d - 1)     # 4 / 8 / 16 at d = 3 / 4 / 5
    if arm == 'binary':
        return 2                # exactly sufficient for the redundant cause
    raise ValueError(f"Unknown capacity arm: {arm!r}")


EXPERIMENTS = {
    'smoke': {
        'repetitions': 2,
        'd_levels': (3,),
        'n_levels': (500,),
        'arm': 'sufficient',
        'bio_n_repeat': 1,
        'builder': 'smoke_conditions',
    },
    'core': {
        'repetitions': 32,
        'd_levels': (3, 4, 5),
        'n_levels': (500, 5000),
        'arm': 'sufficient',
        'bio_n_repeat': 5,      # EM restarts, the program's standard
        'builder': 'core_conditions',
    },
    'capacity_binary': {
        'repetitions': 32,
        'd_levels': (3, 4, 5),
        'n_levels': (500, 5000),
        'arm': 'binary',
        'bio_n_repeat': 5,
        'builder': 'core_conditions',
    },
}


def core_conditions(config):
    conditions = []
    for d in config['d_levels']:
        for generator in GENERATORS:
            for n_samples in config['n_levels']:
                for rep in range(config['repetitions']):
                    conditions.append({
                        'generator': generator,
                        'd': d,
                        'n_samples': n_samples,
                        'dim_hidden': dim_hidden_for(config['arm'], d),
                        'n_repeat': config['bio_n_repeat'],
                        'root_seed': ROOT_SEED_BASE + rep,
                    })
    return conditions


def smoke_conditions(config):
    return core_conditions(config)


CONDITION_BUILDERS = {
    'smoke_conditions': smoke_conditions,
    'core_conditions': core_conditions,
}


def condition_label(condition):
    return (
        f"{condition['generator']} d={condition['d']} "
        f"n={condition['n_samples']} dim_hidden={condition['dim_hidden']} "
        f"root={condition['root_seed']}"
    )


# ---------------------------------------------------------------------------
# One condition
# ---------------------------------------------------------------------------

def run_condition(condition, defaults):
    """Run one condition: one dataset, one fit, all recorded quantities.

    Returns (info_rows, factor_rows). The info row stores the ground-truth
    side and fit diagnostics; factor rows preserve every fitted factor's
    recovery metrics. Ground truth is never used for factor selection.
    """
    generator = condition['generator']
    d = condition['d']
    root = condition['root_seed']
    eta = eta_for(generator, d)

    s_data = derived_seed(root, 'data', generator, d, condition['n_samples'])
    s_corex = derived_seed(root, 'corex', generator, d, condition['n_samples'])
    s_char = derived_seed(root, 'char', generator, d, condition['n_samples'])

    dataset = generate_dataset(
        generator, d, eta, condition['n_samples'], s_data)
    X = dataset['X']
    group_idx = dataset['group_idx']
    nuisance_idx = dataset['nuisance_idx']

    # --- ground-truth side (before and independent of any fit) -----------
    group = X[:, group_idx]
    estimates = group_estimates(group)
    population = population_quantities(generator, d, eta)

    base = {key: condition[key] for key in CONDITION_COLUMNS}
    base.update({'eta': eta, 'n_nuisance': len(nuisance_idx)})

    # --- fit (failures are data: failed_fit rows keep the replicate) -----
    try:
        fit = fit_bio_corex(
            X, defaults['n_factors'], s_corex,
            dim_hidden=condition['dim_hidden'],
            max_iter=defaults['bio_max_iter'],
            n_repeat=condition['n_repeat'],
        )
    except Exception as error:  # noqa: BLE001 — recorded, never raised away
        info_row = {
            **base, **estimates, **population,
            'tc_target': TC_TARGET,
            'top_factor_tc': np.nan,
            'total_tc': np.nan,
            'n_factors_valid': 0,
            'n_negative_mi': np.nan,
            'n_factors_requested': defaults['n_factors'],
            'converged': False,
            'fit_runtime_s': np.nan,
            'fit_status': 'failed_fit',
            'note': str(error),
            'support_discovered': json.dumps([]),
            'support_size': 0,
            'support_matches_group': False,
            'omega_on_support_char': np.nan,
            'omega_on_support_char_mm': np.nan,
            'omega_true_group_char': np.nan,
            'n_char': defaults['n_char'],
        }
        return [info_row], []

    tcs = fit['tcs']
    scores_all = np.clip(factor_variable_scores(fit['model']), 0.0, None)
    truth = np.zeros(X.shape[1], dtype=bool)
    truth[group_idx] = True
    nuisance_mask = np.zeros(X.shape[1], dtype=bool)
    nuisance_mask[nuisance_idx] = True

    # factor ranking by explained TC only (never by recovery ground truth);
    # ties broken by the neutral seeded order, not by factor index
    factor_order = descending_order(tcs, defaults['tie_seed'])

    # per-factor recovery rows, one row per requested factor
    factor_rows = []
    for factor_index in range(defaults['n_factors']):
        scores = scores_all[factor_index]
        top = top_d_metrics(scores, truth, d, defaults['tie_seed'])
        support = cumulative_mi_support(scores, defaults['support_fraction'],
                                        defaults['tie_seed'])
        support_metrics = (
            support_vs_truth_metrics(support.indexes, truth)
            if not support.degenerate
            else {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'jaccard': 0.0}
        )
        factor_rows.append({
            **base,
            'factor_index': factor_index,
            'factor_tc': float(tcs[factor_index]),
            'is_top_tc_factor': int(factor_index == factor_order[0]),
            'auprc': auprc(scores, truth.astype(float)),
            'topd_precision': top['precision'],
            'topd_recall': top['recall'],
            'topd_f1': top['f1'],
            'topd_jaccard': top['jaccard'],
            'nfp_topd': nuisance_false_positive(top['top_indexes'],
                                                nuisance_mask),
            'support_size': support.size,
            'support_density': support.density,
            'support_f1': support_metrics['f1'],
            'nfp_support': nuisance_false_positive(support.indexes,
                                                   nuisance_mask),
            'degenerate_factor': int(support.degenerate),
            'fit_status': 'ok',
        })

    # --- two-stage demonstration on a FRESH characterization sample (§E3.7)
    # freeze the top-TC factor's top-d discovered support, then estimate
    # Omega on exactly those columns of an independently generated sample
    top_factor_index = int(factor_order[0])
    support_discovered = tuple(
        top_d_metrics(scores_all[top_factor_index], truth, d,
                      defaults['tie_seed'])['top_indexes'])
    char_dataset = generate_characterization_sample(
        generator, d, eta, defaults['n_char'], s_char)
    X_char = char_dataset['X']
    discovered_estimates = (
        group_estimates(X_char[:, list(support_discovered)])
        if support_discovered else None)
    true_group_char = group_estimates(X_char[:, group_idx])
    support_matches_group = (
        tuple(sorted(support_discovered)) == tuple(group_idx))

    info_row = {
        **base, **estimates, **population,
        'tc_target': TC_TARGET,
        'top_factor_index': top_factor_index,
        'top_factor_tc': float(tcs[factor_order[0]]),
        'top_factor_index': int(factor_order[0]),
        'total_tc': float(np.sum(tcs)),
        'n_factors_valid': fit['n_factors_valid'],
        'n_negative_mi': fit['n_negative_mi'],
        'n_factors_requested': defaults['n_factors'],
        'converged': fit['converged'],
        'fit_runtime_s': fit['fit_runtime_s'],
        'fit_status': 'ok',
        'note': '',
        'support_discovered': json.dumps(
            [int(i) for i in support_discovered]),
        'support_size': len(support_discovered),
        'support_matches_group': bool(support_matches_group),
        'omega_on_support_char': (
            discovered_estimates['omega_plugin']
            if discovered_estimates else np.nan),
        'omega_on_support_char_mm': (
            discovered_estimates['omega_mm']
            if discovered_estimates else np.nan),
        'omega_true_group_char': true_group_char['omega_plugin'],
        'n_char': defaults['n_char'],
    }
    return [info_row], factor_rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def condition_output_paths(output_dir, experiment_name, condition):
    condition_dir = (
        Path(output_dir) / experiment_name / condition['generator']
        / f"d_{condition['d']}" / f"n_{condition['n_samples']}"
        / f"dim_hidden_{condition['dim_hidden']}"
    )
    rep = condition['root_seed']
    return {
        'info': condition_dir / f'root_{rep:05d}_info.csv',
        'factors': condition_dir / f'root_{rep:05d}_factors.csv',
    }


def calibration_output_path(output_dir, experiment_name):
    return (
        Path(output_dir) / experiment_name / 'redundancy_calibration.csv')


def write_calibration(output_dir, experiment_name):
    """Write the frozen calibration audit trail (deterministic; §E3.4)."""
    path = calibration_output_path(output_dir, experiment_name)
    write_csv_atomically(pd.DataFrame(calibrate(TC_TARGET)), path)
    return path


def _json_safe(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def save_condition_metadata(condition, config, defaults, paths):
    d = condition['d']
    calibration_rows = [
        row for row in calibrate(TC_TARGET)
        if row['d'] == d
        and row['generator'] == condition['generator']
    ]
    metadata = {
        'condition': condition,
        'config': {key: value for key, value in config.items()
                   if key != 'builder'},
        'defaults': defaults,
        'calibration': calibration_rows,
    }
    metadata_path = paths['info'].with_suffix('.meta.json')
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=_json_safe))


def parse_shard(shard):
    pieces = str(shard).split('/')
    if len(pieces) != 2:
        raise ValueError(f"Shard spec must be 'k/N', got {shard!r}.")
    k, n = int(pieces[0]), int(pieces[1])
    if n < 1:
        raise ValueError(f"Shard count must be >= 1, got {n}.")
    if not 0 <= k < n:
        raise ValueError(f"Shard index must satisfy 0 <= {k} < {n}.")
    return k, n


def run_experiment(experiment_name, output_dir, repetitions=None,
                   overwrite=False, dry_run=False, shard=None,
                   combine_only=False):
    """Run one named experiment; skips completed condition pairs.

    With ``shard`` as 'k/N', only every N-th condition is attempted, so
    identical shard invocations run in parallel workers without overlap: a
    sharded run skips the end-of-run combine; run ``--combine-only``
    afterwards.
    """
    config = dict(EXPERIMENTS[experiment_name])
    if repetitions:
        config['repetitions'] = repetitions
    all_conditions = CONDITION_BUILDERS[config['builder']](config)
    conditions = all_conditions
    if combine_only:
        combined = combine_results(output_dir, experiment_name)
        write_calibration(output_dir, experiment_name)
        print(f"Combined: {combined}")
        return combined
    if shard is not None:
        shard_k, shard_n = parse_shard(shard)
        conditions = all_conditions[shard_k::shard_n]
    print(
        f"Experiment: {experiment_name} ({len(conditions)} conditions, "
        f"~{len(conditions):,} CorEx fits"
        + (f", shard {shard}" if shard is not None else "") + ")"
    )
    if dry_run:
        for condition in conditions:
            print(f"  {condition_label(condition)}")
        return None

    append_run_log(output_dir, {
        'event': 'run_start',
        'experiment': experiment_name,
        'repetitions': config.get('repetitions'),
        **({'shard': shard} if shard is not None else {}),
    })
    conditions_written = 0
    status = 'completed'
    started = time.time()
    try:
        for index, condition in enumerate(conditions):
            paths = condition_output_paths(
                output_dir, experiment_name, condition)
            if (paths['info'].exists() and paths['factors'].exists()
                    and not overwrite):
                continue
            conditions_written += 1
            print(f"[{index + 1}/{len(conditions)}] "
                  f"{condition_label(condition)}")
            info_rows, factor_rows = run_condition(condition, DEFAULTS)
            write_csv_atomically(
                pd.DataFrame(info_rows, columns=INFO_COLUMNS),
                paths['info'])
            write_csv_atomically(
                pd.DataFrame(factor_rows, columns=FACTOR_COLUMNS),
                paths['factors'])
            save_condition_metadata(condition, config, DEFAULTS, paths)
        if shard is None:
            write_calibration(output_dir, experiment_name)
            combined = combine_results(output_dir, experiment_name)
        else:
            combined = None
    except BaseException:
        status = 'failed'
        raise
    finally:
        append_run_log(output_dir, {
            'event': 'run_end',
            'experiment': experiment_name,
            'status': status,
            'conditions_written': conditions_written,
            'conditions_skipped': len(conditions) - conditions_written,
            'elapsed_s': round(time.time() - started, 3),
        })
    print(
        f"Done: {len(conditions) - conditions_written}/{len(conditions)} "
        f"conditions were already complete; combined: {combined} "
        f"({time.time() - started:.0f}s)"
    )
    return combined


def combine_results(output_dir, experiment_name):
    experiment_dir = Path(output_dir) / experiment_name
    combined_paths = {}
    for kind in ('info', 'factors'):
        result_files = sorted(experiment_dir.glob(f'**/root_*_{kind}.csv'))
        combined = combine_csv_files(result_files)
        if combined is None:
            continue
        combined_path = (
            experiment_dir / f'{experiment_name}_{kind}_combined.csv')
        write_csv_atomically(combined, combined_path)
        combined_paths[kind] = str(combined_path)
    return combined_paths


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run Experiment 3 redundancy-vs-synergy diagnostics '
                    '(TC-matched, equal-capacity redesign).')
    parser.add_argument(
        'experiment',
        help='Experiment configuration to run',
        choices=sorted(EXPERIMENTS),
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('redundancy_results'),
        help='Directory for condition files, logs, and combined results.')
    parser.add_argument('--repetitions', type=int,
                        help='Override the configured repetition count.')
    parser.add_argument('--shard', type=str, default=None,
                        help="'k/N': attempt only every N-th condition.")
    parser.add_argument('--overwrite', action='store_true',
                        help='Rerun conditions whose files already exist.')
    parser.add_argument('--dry-run', action='store_true',
                        help='List the conditions and exit.')
    parser.add_argument('--combine-only', action='store_true',
                        help='Rebuild combined CSVs from condition files '
                             'without fitting.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    run_experiment(
        args.experiment, args.output_dir, repetitions=args.repetitions,
        overwrite=args.overwrite, dry_run=args.dry_run, shard=args.shard,
        combine_only=args.combine_only,
    )


if __name__ == '__main__':
    main()