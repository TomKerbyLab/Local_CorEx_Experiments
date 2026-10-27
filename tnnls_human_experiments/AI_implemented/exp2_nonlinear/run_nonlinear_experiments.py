"""Run the nonlinear higher-order Local CorEx experiments (spec 02).

Structure mirrors ../synthetic/run_gaussian_experiments.py: every condition
(one replicate of one data-generating configuration) is written to its own
pair of CSV files and completed conditions are skipped by default, so
interrupted runs are restartable. Both files of a condition are built from a
SINGLE generated dataset, so every partition method, estimator, and
corex-input variant within a condition is compared like-for-like.

Cell semantics per condition:

- global:     one fit on all observations, matched independently per region
- oracle:     one fit per true region
- phate:      PHATE(10d) + k-means (k=2, n_init=50); one fit per RAW
              cluster, assigned to true regions by one-to-one Hungarian
              selection maximizing total region-cluster overlap
- raw_kmeans: k-means directly on the standardized variables, same rule
- random:     10 balanced random partitions (draws are kept separate in all
              outputs; they are control replicates, not repeats)

The cell used for an estimated partition is the RAW estimated cluster (its
unmodified label id) — the aligned region id must never index raw labels,
since the two differ under a label swap. Each region is served by exactly
one distinct raw cluster under the one-to-one Hungarian rule (the decided
primary rule); the unconstrained per-region max-overlap selection is
retained only as a diagnostic (``max_overlap_same_cluster`` records the
old rule's two-regions-one-cluster event, ``max_overlap_matches_hungarian``
records whether the two rules coincide).

Scoring uses the nonnegative association scores a_{ji} = I(Z_j; X_i)
(Bio CorEx ``mis`` are permutation-bias-corrected and thresholded upstream
inside local_corex, then clipped at zero — the documented transform) and
Hungarian matching on cosine similarity (spec 00/02). For 'no_routing'
variants the association matrix is re-expanded to full width with zeros at
the dropped routing positions before scoring; routing false positives are
recorded as None there (routing is excluded from the input by construction).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# result_io lives in ../synthetic; APPEND (not insert) so the sibling
# modules there can never shadow this directory's same-named modules
sys.path.append(str(Path(__file__).resolve().parents[2] / 'synthetic'))
from result_io import (  # noqa: E402
    append_run_log,
    combine_csv_files,
    write_csv_atomically,
)

import nonlinear_data  # noqa: E402
import partitioning  # noqa: E402
import recovery_metrics  # noqa: E402
from seeds import derived_seed  # noqa: E402

from corex_fitting import (
    discretize_columns,  # noqa: E402
    factor_variable_scores,
    fit_corex_model,
    standardize_columns,
)
from config import (  # noqa: E402
    CORE_LEVELS,
    DEFAULTS,
    EXPERIMENTS,
    GENERATOR_NOISE_NAMES,
    ROOT_SEED_BASE,
    SUPPLEMENTARY_LEVELS,
)


N_RANDOM_DRAWS = DEFAULTS['n_random_draws']

RUN_COLUMNS = [
    'generator', 'n_per_region', 'alpha', 'noise', 'n_nuisance', 'root_seed',
    'n_factors', 'partition_method', 'partition_draw', 'region',
    'estimator', 'corex_input',
    'partition_ari', 'partition_purity', 'partition_accuracy',
    'region_purity', 'cluster_size', 'max_overlap_same_cluster',
    'max_overlap_matches_hungarian',
    'fit_status', 'runtime_s', 'corex_converged', 'n_factors_valid',
    'total_tc', 'note',
]

MATCH_COLUMNS = [
    'generator', 'n_per_region', 'alpha', 'noise', 'n_nuisance', 'root_seed',
    'n_factors', 'partition_method', 'region', 'partition_draw',
    'estimator', 'corex_input',
    'learned_factor', 'true_group', 'factor_tc', 'cosine', 'auprc',
    'topd_precision', 'topd_recall', 'topd_f1', 'topd_jaccard',
    'routing_fp', 'nuisance_fp', 'support_size', 'support_density',
    'support_precision', 'support_recall', 'support_f1', 'support_jaccard',
    'routing_fp_support', 'nuisance_fp_support',
    'support_degenerate',
]

# ---------------------------------------------------------------------------
# Condition grids (loop logic; the level tuples live in config)
# ---------------------------------------------------------------------------

def discrete_core_conditions(repetitions):
    """n x alpha x eta grid, paired replicates root 2000..2000+reps-1."""
    base = ROOT_SEED_BASE['parity']
    nuisance = CORE_LEVELS['nuisance']
    return [
        {'generator': 'parity', 'n_per_region': n, 'alpha': alpha,
         'noise': eta, 'n_nuisance': nuisance, 'root_seed': base + rep}
        for rep in range(repetitions)
        for n in CORE_LEVELS['n_per_region']
        for alpha in CORE_LEVELS['alpha']
        for eta in CORE_LEVELS['eta']
    ]


def continuous_core_conditions(repetitions):
    """n x alpha x sigma grid, paired replicates root 3000..3000+reps-1."""
    base = ROOT_SEED_BASE['sine_sum']
    nuisance = CORE_LEVELS['nuisance']
    return [
        {'generator': 'sine_sum', 'n_per_region': n, 'alpha': alpha,
         'noise': sigma, 'n_nuisance': nuisance, 'root_seed': base + rep}
        for rep in range(repetitions)
        for n in CORE_LEVELS['n_per_region']
        for alpha in CORE_LEVELS['alpha']
        for sigma in CORE_LEVELS['sigma']
    ]


def discrete_supplementary_conditions(repetitions):
    """One sensitivity axis at a time (config.SUPPLEMENTARY_LEVELS)."""
    base = ROOT_SEED_BASE['parity']
    nuisance = SUPPLEMENTARY_LEVELS
    conditions = []
    # small-sample regime (n = 50), full core grid
    conditions += [
        {'generator': 'parity', 'n_per_region': nuisance['small_n'],
         'alpha': alpha, 'noise': eta, 'n_nuisance': CORE_LEVELS['nuisance'],
         'root_seed': base + rep}
        for rep in range(repetitions)
        for alpha in CORE_LEVELS['alpha']
        for eta in CORE_LEVELS['eta']
    ]
    # extra noise levels at the core condition
    conditions += [
        {'generator': 'parity', 'n_per_region': 500, 'alpha': 0.5,
         'noise': eta, 'n_nuisance': CORE_LEVELS['nuisance'],
         'root_seed': base + rep}
        for rep in range(repetitions) for eta in nuisance['extra_eta']
    ]
    # nuisance sweep at the core condition
    conditions += [
        {'generator': 'parity', 'n_per_region': 500, 'alpha': 0.5,
         'noise': 0.05, 'n_nuisance': p_n, 'root_seed': base + rep}
        for rep in range(repetitions) for p_n in nuisance['nuisance_sweep']
    ]
    return conditions


def continuous_supplementary_conditions(repetitions):
    """Sensitivity axes for the continuous generator + product comparison."""
    base = ROOT_SEED_BASE['sine_sum']
    levels = SUPPLEMENTARY_LEVELS
    conditions = []
    conditions += [
        {'generator': 'sine_sum', 'n_per_region': levels['small_n'],
         'alpha': alpha, 'noise': sigma, 'n_nuisance': CORE_LEVELS['nuisance'],
         'root_seed': base + rep}
        for rep in range(repetitions)
        for alpha in CORE_LEVELS['alpha']
        for sigma in CORE_LEVELS['sigma']
    ]
    conditions += [
        {'generator': 'sine_sum', 'n_per_region': 500, 'alpha': 0.5,
         'noise': 0.20, 'n_nuisance': p_n, 'root_seed': base + rep}
        for rep in range(repetitions) for p_n in levels['nuisance_sweep']
    ]
    # product-group comparison at the core condition: nonlinear but NOT
    # pairwise-independent; results must be quoted with that qualification
    conditions += [
        {'generator': 'product', 'n_per_region': 500, 'alpha': 0.5,
         'noise': sigma, 'n_nuisance': CORE_LEVELS['nuisance'],
         'root_seed': base + rep}
        for rep in range(repetitions) for sigma in CORE_LEVELS['sigma']
    ]
    return conditions


def smoke_conditions(repetitions):
    return [
        {'generator': 'parity', 'n_per_region': 300, 'alpha': 0.5,
         'noise': 0.05, 'n_nuisance': 14, 'root_seed': 2000},
        {'generator': 'sine_sum', 'n_per_region': 300, 'alpha': 0.5,
         'noise': 0.20, 'n_nuisance': 14, 'root_seed': 3000},
    ]


# the config keeps the editing surface; the runner owns the loop logic
CONDITION_BUILDERS = {
    'smoke': smoke_conditions,
    'discrete_core': discrete_core_conditions,
    'continuous_core': continuous_core_conditions,
    'discrete_supplementary': discrete_supplementary_conditions,
    'continuous_supplementary': continuous_supplementary_conditions,
}


def variant_specs(generator, config):
    """Fit-variant spec table for one condition's generator.

    One entry per (estimator, corex-input) fit variant, stating WHAT the
    fit does — downstream code reads these fields and never parses display
    names. Variants for one condition (options come from config):

    - every (estimator, corex_input) pair; for continuous data the
      'with_routing' input is prefixed 'binned_' (standardize, then
      discretize all columns);
    - native gaussian Bio CorEx on the standardized space ('gaussian_'
      prefix) as a secondary variant when config enables it;
    - a 'no_routing' input drops the routing columns from the fit matrix.
    """
    continuous = generator != 'parity'
    specs = []
    for estimator in config['estimators']:
        for corex_input in config['corex_inputs']:
            specs.append({
                'name': (f'binned_{corex_input}'
                         if continuous and corex_input == 'with_routing'
                         else corex_input),
                'estimator': estimator,
                'marginal': 'discrete',
                'standardize': continuous,
                # parity bins only the routing columns; continuous data is
                # standardized then discretized (bin and drop commute, so a
                # no_routing variant sees exactly the old reduced matrix)
                'discretize': 'routing' if not continuous else 'binned',
                'drop_routing': corex_input.endswith('no_routing'),
            })
    if config.get('native_gaussian_bio', DEFAULTS['native_gaussian_bio']):
        specs += [{
            'name': f'gaussian_{corex_input}',
            'estimator': 'bio',
            'marginal': 'gaussian',
            'standardize': True,
            'discretize': None,
            'drop_routing': corex_input.endswith('no_routing'),
        } for corex_input in config['corex_inputs']]
    return specs


# ---------------------------------------------------------------------------
# Fit cells: one dict per fit, holding rows to fit and truth regions served
# ---------------------------------------------------------------------------

def global_cells(dataset):
    """Fixed cells: the global fit on the full dataset (serves both regions)."""
    return [{
        'method': 'global', 'draw': 0, 'name': 'global',
        'region_id': -1, 'rows': np.arange(len(dataset['region'])),
        'regions': (0, 1), 'fields': None,
    }]


def oracle_cells(dataset):
    """Fixed cells: one local fit per true region (cell id == region id)."""
    return [
        {
            'method': 'oracle', 'draw': 0, 'name': f'region{region}',
            'region_id': region,
            'rows': np.flatnonzero(dataset['region'] == region),
            'regions': (region,), 'fields': None,
        }
        for region in (0, 1)
    ]


def estimated_cells(partition_result, true_labels, method, draw=0):
    """Data-driven cells for one estimated partition (Hungarian primary).

    Each true region is served by exactly one RAW estimated cluster,
    chosen by the one-to-one Hungarian assignment that maximizes the total
    region-cluster overlap — the decided primary search rule. Labels are
    never re-aligned; raw-label slicing must stay valid (the
    raw-vs-aligned regression). The previous unconstrained per-region
    max-overlap rule is retained only as a diagnostic, recorded in the
    cell fields:
    ``max_overlap_same_cluster`` — the old rule would have picked ONE
    cluster for both regions (possible only there; the one-to-one rule
    never shares a cluster);
    ``max_overlap_matches_hungarian`` — the per-region argmax choices
    coincide with the Hungarian assignments.
    """
    true_labels = np.asarray(true_labels)
    predicted = np.asarray(partition_result['predicted_labels'])
    cluster_ids = np.unique(predicted)
    regions = np.unique(true_labels)
    contingency = np.array([
        [int(np.sum((predicted == cluster) & (true_labels == region)))
         for cluster in cluster_ids]
        for region in regions
    ])
    region_indexes, cluster_indexes = linear_sum_assignment(-contingency)
    selected = {
        int(regions[region_index]): (
            int(cluster_ids[cluster_index])
            if contingency[region_index, cluster_index] > 0 else None)
        for region_index, cluster_index in zip(
            region_indexes, cluster_indexes)
    }
    max_overlap = {
        int(region): (
            int(cluster_ids[int(np.argmax(contingency[region_index]))])
            if contingency[region_index].sum() > 0 else None)
        for region_index, region in enumerate(regions)
    }
    served = {}
    for region in sorted(selected):
        served.setdefault(selected[region], []).append(region)
    fields = {
        'partition_ari': partition_result['ari'],
        'partition_purity': partition_result['purity'],
        'partition_accuracy': partition_result['mapped_accuracy'],
        'max_overlap_same_cluster': (
            max_overlap.get(0) is not None
            and max_overlap.get(0) == max_overlap.get(1)),
        'max_overlap_matches_hungarian': selected == max_overlap,
    }
    return [
        {
            'method': method, 'draw': draw, 'name': f'cluster{cluster}',
            'region_id': int(cluster),
            'rows': np.flatnonzero(predicted == cluster),
            'regions': tuple(regions), 'fields': fields,
        }
        for cluster, regions in served.items() if cluster is not None
    ]


def fit_matrices(dataset, specs, defaults):
    """Fit matrix for every variant spec, computed once per condition.

    Returns {spec name: {'matrix', 'dropped', 'edges', 'binning'}}. The
    matrix depends only on the input recipe, so estimators sharing an input
    share the entry. Routing columns are binned with the fixed rule for
    discrete data; continuous data is standardized then discretized, except
    native gaussian Bio CorEx which fits the standardized space.
    'dropped' lists routing columns removed for no_routing specs; scoring
    re-inserts zeros so association columns stay aligned with the
    full-width ground truth.
    """
    signal = np.asarray(dataset['X'], dtype=float)
    routing_idx = np.asarray(dataset['routing_idx'], dtype=int)
    built = {}
    for spec in specs:
        name = spec['name']
        if name in built:
            continue
        matrix = standardize_columns(signal) if spec['standardize'] else np.copy(signal)
        edges = None
        binning = 'standardized' if spec['standardize'] else 'signal'
        if spec['discretize'] == 'binned':
            matrix, edges = discretize_columns(matrix, defaults['n_bins'])
            binning = 'standardized_then_binned'
        elif spec['discretize'] == 'routing':
            (matrix[:, routing_idx],
             edges) = discretize_columns(signal[:, routing_idx],
                                         defaults['n_bins'])
            binning = 'routing_binned'
        if spec['drop_routing']:
            keep = np.ones(matrix.shape[1], dtype=bool)
            keep[routing_idx] = False
            matrix = matrix[:, keep]
        built[name] = {
            'matrix': matrix, 'dropped': (routing_idx if spec['drop_routing']
                                          else None),
            'edges': edges, 'binning': binning,
        }
    missing = [spec['name'] for spec in specs if spec['name'] not in built]
    if missing:
        raise ValueError(f"Unbuilt corex-input variants: {missing}")
    return built


# ---------------------------------------------------------------------------
# Condition data and fit matrices
# ---------------------------------------------------------------------------

def generate_condition_data(condition):
    """One generated dataset with its ground truth (data separation)."""
    data_seed = derived_seed(condition['root_seed'], 'data')
    if condition['generator'] == 'parity':
        return nonlinear_data.generate_discrete_dataset(
            n_per_region=condition['n_per_region'],
            alpha=condition['alpha'],
            eta=condition['noise'],
            n_nuisance=condition['n_nuisance'],
            seed=data_seed,
        )
    return nonlinear_data.generate_continuous_dataset(
        n_per_region=condition['n_per_region'],
        alpha=condition['alpha'],
        sigma=condition['noise'],
        n_nuisance=condition['n_nuisance'],
        seed=data_seed,
        product_groups=(condition['generator'] == 'product'),
    )


def estimate_partitions(dataset, defaults):
    """Every estimated partition for one dataset.

    Returns {'phate': ..., 'raw_kmeans': ..., ('random', draw): ...} from
    partitioning.partition_data, plus the standardized matrix they were
    computed on. Raises ValueError on collapsed partitions (callers record
    them as failed partitions).
    """
    standardized = standardize_columns(dataset['X'])
    true_labels = dataset['region']
    partition_seed = derived_seed(dataset['seed'], 'partition')
    return {
        'standardized': standardized,
        'phate': partitioning.partition_data(
            standardized, true_labels, method='phate',
            random_state=partition_seed,
            phate_dimensions=defaults['phate_dimensions'],
            kmeans_n_init=defaults['kmeans_n_init'],
        ),
        'raw_kmeans': partitioning.partition_data(
            standardized, true_labels, method='raw',
            random_state=partition_seed + 1,
            kmeans_n_init=defaults['kmeans_n_init'],
        ),
        **{
            ('random', draw): partitioning.partition_data(
                standardized, true_labels, method='random',
                random_state=partition_seed + 100 + draw,
            )
            for draw in range(defaults['n_random_draws'])
        },
    }




# ---------------------------------------------------------------------------
# Fitting and scoring
# ---------------------------------------------------------------------------

def fit_one_model(matrix, estimator, n_factors, model_seed, defaults,
                  marginal='discrete'):
    """Fit one CorEx model; returns (model, linear, fit_fields).

    fit_fields always carries fit_status/note plus runtime, convergence,
    factor-count, and total-TC fields (TCs in model units, local_corex
    nats). Failed fits are returned with a failed_fit status, never raised.
    """
    start = time.time()
    try:
        bio_options = {
            'max_iter': defaults['bio_max_iter'],
            'dim_hidden': defaults['bio_dim_hidden'],
            'n_repeat': defaults['bio_n_repeat'],
        }
        if marginal == 'gaussian':
            bio_options['marginal_description'] = 'gaussian'
        model, linear, used_seed, _ = fit_corex_model(
            matrix, n_factors,
            'linear' if estimator == 'linear' else 'bio',
            model_seed=int(model_seed),
            bio_options=bio_options,
            linear_options={'max_iter': defaults['linear_max_iter']},
        )
    except Exception as error:  # failed fits are recorded, not raised
        return None, None, {
            'fit_status': 'failed_fit',
            'note': f'{type(error).__name__}: {error}',
            'runtime_s': time.time() - start,
            'corex_converged': False,
            'n_factors_valid': 0,
            'total_tc': np.nan,
        }
    tcs = np.asarray(model.tcs, dtype=float)
    if linear:
        history = getattr(model, 'history', None) or {}
        converged = len(history.get('TC', [])) < defaults['linear_max_iter']
    else:
        history = getattr(model, 'tc_history', None) or []
        converged = (
            len(history) < defaults['bio_max_iter'] if history else True)
    return model, linear, {
        'fit_status': 'ok',
        'note': '',
        'runtime_s': time.time() - start,
        'corex_converged': bool(converged),
        'n_factors_valid': int(np.sum(tcs > 0)),
        'total_tc': float(np.sum(tcs)),
    }


def expand_to_full_width(association, dropped, n_variables):
    """Re-insert zero columns at dropped positions (full-width scoring).

    Truth vectors are zero at the dropped routing positions, so cosine
    matching is unchanged, while all false-positive and support indexes
    remain in the original column space.
    """
    if dropped is None:
        return association
    keep = np.setdiff1d(np.arange(n_variables), np.asarray(dropped, dtype=int))
    expanded = np.zeros((association.shape[0], n_variables), dtype=float)
    expanded[:, keep] = association
    return expanded


def fit_variant(dataset, cell_matrix, spec, n_factors, model_seed, defaults):
    """Fit one variant (spec table entry) on one cell; returns (fit, fields).

    ``fit`` carries the model, linearity, the zero-clipped association
    matrix re-expanded to full width, the raw TCs, and which columns were
    dropped (None when the fit keeps the routing columns).
    """
    model, linear, fit_fields = fit_one_model(
        cell_matrix, spec['estimator'], n_factors, model_seed, defaults,
        marginal=spec['marginal'])
    if model is None:
        return None, fit_fields
    association = np.maximum(
        factor_variable_scores(model, linear=linear), 0.0)
    dropped = (dataset['routing_idx'] if spec['drop_routing'] else None)
    association = expand_to_full_width(
        association, dropped, dataset['X'].shape[1])
    fit = {
        'model': model,
        'linear': linear,
        'association': association,
        'tcs': np.asarray(model.tcs, dtype=float),
        'dropped': None if dropped is None else np.asarray(dropped, dtype=int),
        'true_groups': dataset['true_groups'],
    }
    return fit, fit_fields


def score_region_fit(base_row, fit, region):
    """Hungarian-match one fit's factors to one region's true groups.

    Returns one matches-table row per matched (factor, group) pair. Routing
    false positives are None for no_routing variants (routing is excluded
    from the input by construction) and computed on the zero-expanded
    association otherwise.
    """
    truth = np.vstack([
        fit['true_groups'][(region, group)]
        for group in range(1, nonlinear_data.N_GROUPS + 1)
    ])
    recovery = recovery_metrics.match_factors_to_groups(
        fit['association'], truth)
    rows = []
    for assignment in recovery['assignments']:
        row = dict(base_row)
        learned = assignment['learned_factor']
        row.update({
            'region': region,
            'learned_factor': learned,
            'true_group': int(assignment['true_group'] + 1),
            'factor_tc': float(fit['tcs'][learned]),
            'cosine': assignment['cosine_similarity'],
            'auprc': assignment['average_precision'],
            'topd_precision': assignment['top_d_precision'],
            'topd_recall': assignment['top_d_recall'],
            'topd_f1': assignment['top_d_f1'],
            'topd_jaccard': assignment['top_d_jaccard'],
            'routing_fp': (
                assignment['top_d_routing_fraction']
                if fit['dropped'] is None else None),
            'nuisance_fp': assignment['top_d_nuisance_fraction'],
            'support_size': assignment['support_size'],
            'support_density': assignment['support_density'],
            'support_precision': assignment['support_precision'],
            'support_recall': assignment['support_recall'],
            'support_f1': assignment['support_f1'],
            'support_jaccard': assignment['support_jaccard'],
            'routing_fp_support': (
                assignment['support_routing_fraction']
                if fit['dropped'] is None else None),
            'nuisance_fp_support': assignment['support_nuisance_fraction'],
            'support_degenerate': assignment['support_degenerate'],
        })
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Per-condition execution
# ---------------------------------------------------------------------------

def run_condition(condition, config, defaults):
    """Run one condition: one dataset, all partitions, estimators, variants.

    Returns (runs_rows, match_rows). Both tables record every attempted fit
    (also failed ones, with fit_status / note) so failures stay visible.
    Partition-only diagnostics travel on the fit rows (one row per fit).
    """
    dataset = generate_condition_data(condition)
    generator = condition['generator']
    n_factors = defaults['n_factors']
    specs = variant_specs(generator, config)
    matrices = fit_matrices(dataset, specs, defaults)

    base_row = {
        key: condition[key]
        for key in ('generator', 'n_per_region', 'alpha', 'noise',
                    'n_nuisance', 'root_seed')
    }
    base_row.update({
        'n_factors': n_factors,
        'partition_draw': 0,
        'partition_ari': np.nan,
        'partition_purity': np.nan,
        'partition_accuracy': np.nan,
        'max_overlap_same_cluster': np.nan,
        'max_overlap_matches_hungarian': np.nan,
        'region_purity': np.nan,
        'cluster_size': np.nan,
    })

    runs_rows = []
    match_rows = []

    # fixed cells first, then the estimated partitions in fixed order;
    # every cell is one uniform dict, so the fitting loop below is uniform
    cells = global_cells(dataset) + oracle_cells(dataset)
    try:
        partitions = estimate_partitions(dataset, defaults)
        partition_error = None
    except ValueError as error:
        partitions = {}
        partition_error = str(error)
    if partition_error is None:
        cells += estimated_cells(partitions['phate'], dataset['region'],
                                 method='phate')
        cells += estimated_cells(
            partitions['raw_kmeans'], dataset['region'],
                                 method='raw_kmeans')
        # balanced random partitions: draws kept separate (control replicates)
        for draw in range(defaults['n_random_draws']):
            cells += estimated_cells(
                partitions[('random', draw)], dataset['region'],
                method='random', draw=draw)
    else:
        # a collapsed partition fails both phate and raw and is recorded
        # with a failed_partition status instead of fits
        for method in ('phate', 'raw_kmeans'):
            runs_rows.append({
                **base_row,
                'partition_method': method,
                'partition_draw': 0,
                'region': -1,
                'estimator': 'none',
                'corex_input': 'none',
                'fit_status': 'failed_partition',
                'note': partition_error,
            })

    for cell in cells:
        for spec in specs:
            cell_matrix = matrices[spec['name']]['matrix'][cell['rows'], :]
            fit, fit_fields = fit_variant(
                dataset, cell_matrix, spec, n_factors,
                derived_seed(
                    condition['root_seed'], 'corex', cell['draw'],
                    cell['method'], cell['name'], spec['estimator'],
                    spec['name']),
                defaults,
            )
            run_row = {
                **base_row,
                'partition_method': cell['method'],
                'partition_draw': cell['draw'],
                'region': cell['region_id'],
                'estimator': spec['estimator'],
                'corex_input': spec['name'],
                'cluster_size': int(len(cell['rows'])),
                **(cell['fields'] or {}),
                **fit_fields,
            }
            runs_rows.append(run_row)
            if fit is None:
                continue
            for served_region in cell['regions']:
                match_rows.extend(score_region_fit(
                    {
                        **base_row,
                        'partition_method': cell['method'],
                        'partition_draw': cell['draw'],
                        'estimator': spec['estimator'],
                        'corex_input': spec['name'],
                    },
                    fit,
                    served_region,
                ))

    return runs_rows, match_rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def condition_output_paths(output_dir, experiment_name, condition):
    """CSV paths for one restartable condition (runs and matches)."""
    condition_dir = (
        Path(output_dir)
        / experiment_name
        / condition['generator']
        / f"n_{condition['n_per_region']}"
        / f"alpha_{condition['alpha']}"
        / f"{GENERATOR_NOISE_NAMES[condition['generator']]}"
        f"_{condition['noise']}"
        / f"nuisance_{condition['n_nuisance']}"
    )
    rep = condition['root_seed']
    return {
        'runs': condition_dir / f'root_{rep:05d}_runs.csv',
        'matches': condition_dir / f'root_{rep:05d}_matches.csv',
    }


def _json_safe(value):
    """Convert numpy scalars and unknown objects for JSON metadata."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)  # functions and other objects: their repr is enough


def save_condition_metadata(condition, config, defaults, paths):
    """Save the resolved configuration for one condition (spec 00 record)."""
    # the experiment config carries the builder function; record only the
    # resolved values that affect the science
    metadata = {
        'condition': condition,
        'config': {key: value for key, value in config.items()
                   if key != 'builder'},
        'defaults': defaults,
    }
    metadata_path = paths['runs'].with_suffix('.meta.json')
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=_json_safe))


def run_experiment(experiment_name, output_dir, repetitions=None,
                   overwrite=False, include_no_routing=False, dry_run=False):
    """Run one named experiment; skips completed condition pairs."""
    config = dict(EXPERIMENTS[experiment_name])
    config['builder'] = CONDITION_BUILDERS[experiment_name]
    if repetitions:
        config['repetitions'] = repetitions
    if include_no_routing:
        config['corex_inputs'] = tuple(config['corex_inputs']) + (
            'no_routing',)
    conditions = config['builder'](config.get('repetitions'))
    print(
        f"Experiment: {experiment_name} ({len(conditions)} conditions, "
        f"~{expected_fit_count(config, conditions):,} CorEx fits)"
    )
    if dry_run:
        return None

    append_run_log(output_dir, {
        'event': 'run_start',
        'experiment': experiment_name,
        'repetitions': config.get('repetitions'),
    })
    conditions_written = 0
    status = 'completed'
    started = time.time()
    try:
        for index, condition in enumerate(conditions):
            paths = condition_output_paths(
                output_dir, experiment_name, condition)
            if (paths['runs'].exists() and paths['matches'].exists()
                    and not overwrite):
                continue
            conditions_written += 1
            noise_name = GENERATOR_NOISE_NAMES[condition['generator']]
            print(
                f"[{index + 1}/{len(conditions)}] "
                f"{condition['generator']} n={condition['n_per_region']} "
                f"alpha={condition['alpha']} {noise_name}={condition['noise']} "
                f"nuisance={condition['n_nuisance']} "
                f"root={condition['root_seed']}"
            )
            runs_rows, match_rows = run_condition(
                condition, config, DEFAULTS)
            write_csv_atomically(
                pd.DataFrame(runs_rows, columns=RUN_COLUMNS), paths['runs'])
            write_csv_atomically(
                pd.DataFrame(match_rows, columns=MATCH_COLUMNS),
                paths['matches'])
            save_condition_metadata(condition, config, DEFAULTS, paths)
        combined = combine_results(output_dir, experiment_name)
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


def expected_fit_count(config, conditions):
    """Worst-case number of CorEx fits for one experiment configuration."""
    variants_per_generator = {}
    for condition in conditions:
        variants_per_generator.setdefault(
            condition['generator'],
            len(variant_specs(condition['generator'], config)))
    total = 0
    for condition in conditions:
        n_variants = variants_per_generator[condition['generator']]
        # global (1) + oracle (2) fixed cells; up to 2 cells for each of
        # phate/raw_kmeans; up to 2 cells for each of the 10 random draws
        n_cells = 3 + 2 + 2 + 2 * config.get(
            'n_random_draws', DEFAULTS['n_random_draws'])
        total += n_variants * n_cells
    return total


def combine_results(output_dir, experiment_name):
    """Combine all completed condition files for one experiment."""
    experiment_dir = Path(output_dir) / experiment_name
    combined_paths = {}
    for kind in ('runs', 'matches'):
        result_files = sorted(experiment_dir.glob(f'**/root_*_{kind}.csv'))
        combined = combine_csv_files(result_files)
        if combined is None:
            continue
        combined_path = (
            experiment_dir / f'{experiment_name}_{kind}_combined.csv')
        write_csv_atomically(combined, combined_path)
        combined_paths[kind] = combined_path
    return combined_paths.get('matches')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run nonlinear higher-order Local CorEx experiments.',
    )
    parser.add_argument(
        'experiment',
        choices=tuple(EXPERIMENTS),
        help='Experiment configuration to run.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('nonlinear_results'),
        help='Directory for condition files and combined results.',
    )
    parser.add_argument(
        '--repetitions',
        type=int,
        help='Override the configured repetition count (smoke tests).',
    )
    parser.add_argument(
        '--include-no-routing',
        action='store_true',
        help='Also run the secondary corex-input variant without the two '
             'routing columns.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Rerun conditions whose CSV files already exist.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the expected condition and fit counts without running.',
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    run_experiment(
        experiment_name=args.experiment,
        output_dir=args.output_dir,
        repetitions=args.repetitions,
        overwrite=args.overwrite,
        include_no_routing=args.include_no_routing,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()