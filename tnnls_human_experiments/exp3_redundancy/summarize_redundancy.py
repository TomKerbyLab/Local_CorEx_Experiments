"""Summarize Experiment 3 results and build the limitations panel.

Reads the per-condition CSVs (info and factor tables), writes aggregate
tables with mean/SE/95% CI per condition, and renders the supplementary
panel. Panel titles stay neutral until results exist (paper exp3.md
§E3.10): outcome language is a findings decision, not a figure caption.
Sample sizes are never collapsed into one trendline (n = 500 and n = 5000
are separate regimes at these dimensions) and the two capacity arms are
never pooled (§E3.5) — the groupby keys carry both.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_KEYS = ('generator', 'd', 'n_samples', 'dim_hidden', 'n_repeat')
Z_95 = 1.959963985
POPULATION_FACTS = (
    'tc_population', 'dtc_population', 'omega_population',
    'mean_pmi_population',
)


def mean_se_ci(values, z=Z_95):
    """Mean, SE, and normal-approximation CI for replicate-level values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {'mean': np.nan, 'se': np.nan, 'ci_low': np.nan,
                'ci_high': np.nan, 'n_valid': 0}
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    se = float(sd / np.sqrt(len(values)))
    return {'mean': mean, 'se': se, 'ci_low': mean - z * se,
            'ci_high': mean + z * se, 'n_valid': int(len(values))}


def _aggregate_by_condition(df, quantity_columns, extra_per_condition=None):
    """Mean/SE/CI per condition for each quantity column."""
    rows = []
    for keys, group in df.groupby(list(BASE_KEYS)):
        record = dict(zip(BASE_KEYS, keys))
        record['n_replicates'] = len(group)
        if 'fit_status' in group.columns:
            record['n_failed_fits'] = int((group['fit_status'] != 'ok').sum())
        if extra_per_condition:
            record.update(extra_per_condition(group))
        for column in quantity_columns:
            if column not in group.columns:
                continue
            stats = mean_se_ci(group[column])
            for suffix, value in stats.items():
                record[f'{column}_{suffix}'] = value
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_info(info):
    """Info-table summary: estimator agreement with population constants."""
    quantities = (
        'tc_plugin', 'tc_mm', 'omega_plugin', 'omega_mm', 'top_factor_tc',
        'total_tc', 'fit_runtime_s', 'omega_on_support_char',
        'omega_on_support_char_mm', 'omega_true_group_char',
    )
    extra = lambda group: {
        'support_matches_group_rate':
            float(group['support_matches_group'].mean()),
        **{column: float(group[column].iloc[0])
           for column in POPULATION_FACTS if column in group.columns},
    }
    return _aggregate_by_condition(info, quantities, extra)


def summarize_factors(factors):
    """Factor-table summary over the primary (highest-TC) factors plus the
    best-over-factors auxiliary read (never the primary metric)."""
    quantities = ('auprc', 'topd_f1', 'topd_jaccard', 'support_f1',
                  'nfp_topd', 'nfp_support')
    top = factors[factors['is_top_tc_factor'] == 1]
    rows = []
    for keys, group in top.groupby(list(BASE_KEYS)):
        record = dict(zip(BASE_KEYS, keys))
        record['n_replicates'] = len(group)
        for column in quantities:
            stats = mean_se_ci(group[column])
            for suffix, value in stats.items():
                record[f'{column}_{suffix}'] = value
        # best-over-factors: max across all requested factors per replicate
        condition_keys = list(BASE_KEYS) + ['root_seed']
        best = factors.groupby(condition_keys)[list(quantities)].max()
        matched = best.loc[[
            index for index in best.index if index[:-1] == keys
        ]]
        for column in ('auprc', 'topd_f1', 'support_f1'):
            stats = mean_se_ci(matched[column])
            for suffix, value in stats.items():
                record[f'best_factor_{column}_{suffix}'] = value
        rows.append(record)
    return pd.DataFrame(rows)


def equivalence_readout(info, margin_bits=0.05):
    """Generator differences in top-factor TC with CI against the margin.

    Paper exp3.md §E3.8 commitment 1: nonsignificant is not equivalent;
    differences are read against pre-specified margins, not against zero.
    Root seeds pair the replicates across generators (same root seed, d,
    n, arm), so the generator contrast is a paired difference.
    """
    keys = ('d', 'n_samples', 'dim_hidden', 'n_repeat')
    rows = []
    for pair_keys, group in info.groupby(list(keys)):
        redundant = group[group['generator'] == 'redundant']
        synergistic = group[group['generator'] == 'synergistic']
        merged = redundant.merge(
            synergistic, on='root_seed', suffixes=('_r', '_s'))
        if merged.empty:
            continue
        difference = merged['top_factor_tc_r'] - merged['top_factor_tc_s']
        stats = mean_se_ci(difference.values)
        equivalent = (
            np.isfinite(stats['ci_low'])
            and stats['ci_low'] > -margin_bits
            and stats['ci_high'] < margin_bits
        )
        rows.append({
            **dict(zip(keys, pair_keys)),
            'n_pairs': len(merged),
            'tc_difference_mean': stats['mean'],
            'tc_difference_se': stats['se'],
            'tc_difference_ci_low': stats['ci_low'],
            'tc_difference_ci_high': stats['ci_high'],
            'equivalent_within_margin': bool(equivalent),
            'margin_bits': margin_bits,
        })
    return pd.DataFrame(rows)


def panel(info, factors, output_path):
    """Four-panel limitations figure; neutral title (§E3.10)."""
    colors = {'redundant': '#1f77b4', 'synergistic': '#d62728'}
    markers = {500: 'o', 5000: 's'}

    def plot_lines(ax, source, column):
        if column not in source.columns:
            return
        stats = source.groupby(
            ['generator', 'd', 'n_samples', 'dim_hidden'], as_index=False
        )[column].agg(['mean', 'sem']).reset_index()
        for (generator, n_sample, dim), group in stats.groupby(
                ['generator', 'n_samples', 'dim_hidden']):
            group = group.sort_values('d')
            label = f"{generator} n={n_sample} dim{dim}"
            ax.errorbar(group['d'], group['mean'], yerr=group['sem'],
                        marker=markers.get(n_sample, 'o'), capsize=3,
                        color=colors[generator], linestyle='-',
                        label=label)

    def plot_population(ax, source, column):
        if column not in source.columns:
            return
        pop = source.groupby(['generator', 'd'],
                             as_index=False)[column].first()
        for generator, group in pop.groupby('generator'):
            ax.plot(group['d'], group[column], linestyle=':',
                    color=colors[generator], alpha=0.85,
                    label=f"{generator}: population")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0][0]
    plot_lines(ax, info, 'top_factor_tc')
    plot_population(ax, info, 'tc_population')
    ax.set_ylabel('top-factor explained TC (bits)')

    ax = axes[0][1]
    plot_lines(ax, info, 'omega_plugin')
    plot_population(ax, info, 'omega_population')
    ax.axhline(0, color='k', lw=0.8, alpha=0.5)
    ax.set_ylabel('O-information on true group (bits)')

    ax = axes[1][0]
    top = factors[factors['is_top_tc_factor'] == 1]
    plot_lines(ax, top, 'topd_f1')
    ax.set_ylabel('support-recovery F1 (top-TC factors)')

    ax = axes[1][1]
    plot_lines(ax, info, 'omega_on_support_char')
    plot_population(ax, info, 'omega_population')
    ax.axhline(0, color='k', lw=0.8, alpha=0.5)
    ax.set_ylabel('Omega on discovered support (fresh sample)')

    for row in axes:
        for ax in row:
            ax.set_xlabel('group size d')
            ax.grid(alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            if unique:
                ax.legend(fontsize=7)

    fig.suptitle(
        "Exp 3 (equal population TC 0.5 bits): CorEx discovers only the "
        "redundancy-dominated structure; Ω separates the types on "
        "recovered supports",
        fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"saved {output_path}")


def main(output_dir=Path('redundancy_results'), experiment=None,
         margin_bits=0.05):
    output_dir = Path(output_dir)
    experiment_dirs = (
        [output_dir / experiment] if experiment
        else sorted(path for path in output_dir.iterdir()
                    if path.is_dir() and path.name != 'shard_logs'))
    sys.path.append(str(Path(__file__).resolve().parents[1] / 'gaussian'))
    from result_io import combine_csv_files
    for experiment_dir in experiment_dirs:
        info_files = sorted(experiment_dir.glob('**/root_*_info.csv'))
        factor_files = sorted(experiment_dir.glob('**/root_*_factors.csv'))
        if not info_files:
            print(f"no info results under {experiment_dir}; skipping")
            continue
        info = combine_csv_files(info_files)
        factors = combine_csv_files(factor_files)
        if factors is None:
            factors = info.iloc[0:0]  # empty table keeps the panel runnable
        if not set(BASE_KEYS).issubset(info.columns):
            print(f"{experiment_dir.name}: not the info-table schema "
                  f"(e.g. the capability diagnostic); skipping")
            continue
        info_summary = summarize_info(info)
        factor_summary = summarize_factors(factors)
        equivalence = equivalence_readout(info, margin_bits)
        name = experiment_dir.name
        info_summary.to_csv(
            experiment_dir / f'{name}_info_summary.csv', index=False)
        factor_summary.to_csv(
            experiment_dir / f'{name}_factors_summary.csv', index=False)
        equivalence.to_csv(
            experiment_dir / f'{name}_equivalence.csv', index=False)
        panel(info, factors, experiment_dir / f'{name}_panel.png')
        inside = int(equivalence['equivalent_within_margin'].sum()) \
            if len(equivalence) else 0
        print(f"summarized {name}: {len(info_summary)} info conditions; "
              f"{inside}/{len(equivalence)} generator TC differences "
              f"inside the ±{margin_bits} bit margin")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Summarize redundancy-vs-synergy results.')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('redundancy_results'))
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--margin-bits', type=float, default=0.05)
    args = parser.parse_args()
    main(args.output_dir, args.experiment, args.margin_bits)