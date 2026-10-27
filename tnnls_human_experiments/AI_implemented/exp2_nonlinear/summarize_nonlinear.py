"""Summarize the nonlinear experiment outputs and draw the paper figure.

Inputs (produced by run_nonlinear_experiments.py and structure
diagnostics): condition-level ``root_*_runs.csv`` / ``root_*_matches.csv``
files (or their combined CSVs) and the structure/capability diagnostics
CSVs.

The summary aggregates WITHIN each replicate fit — one row per (condition,
partition method, region, partition draw, estimator, corex input) holding
the mean over that region's matched factors — so random-partition draws are
kept separate (they are control replicates, not repeats of a condition);
collapsing over draws is deliberately left to downstream analysis.

The main-paper figure is 2 rows (discrete parity / continuous sine-sum) x
3 panels:
  1. matched AUPRC vs routing separation alpha, by partition method
     (bio estimator, with-routing input, mid noise level)
  2. matched AUPRC vs noise level, bio vs linear (oracle local fits)
  3. structure diagnostics: group TC (bits) vs permuted TC per noise level
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / 'synthetic'))
from result_io import combine_csv_files  # noqa: E402
from config import GENERATOR_NOISE_NAMES as GENERATOR_NOISE_NAME  # noqa: E402

# categorical palette (validated default instance): slots 1..5 in fixed
# order; markers provide a second identity channel for grayscale/CVD print
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
SERIES_MARKERS = ['o', 's', '^', 'D', 'v']
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID_COLOR = '#e5e3df'
SURFACE = '#fcfcfb'

PARTITION_ORDER = ['global', 'oracle', 'phate', 'raw_kmeans', 'random']
PARTITION_LABELS = {
    'global': 'global',
    'oracle': 'oracle',
    'phate': 'PHATE',
    'raw_kmeans': 'raw k-means',
    'random': 'random',
}
ESTIMATOR_LABELS = {'bio': 'Bio CorEx', 'linear': 'Linear CorEx'}

CONDITION_KEYS = [
    'generator', 'n_per_region', 'alpha', 'noise', 'n_nuisance',
    'root_seed', 'n_factors',
]
WITHIN_FIT_KEYS = [
    *CONDITION_KEYS, 'partition_method', 'partition_draw', 'region',
    'estimator', 'corex_input',
]

MATCH_MEAN_COLUMNS = [
    'auprc', 'cosine', 'topd_precision', 'topd_recall', 'topd_f1',
    'topd_jaccard', 'nuisance_fp', 'routing_fp', 'support_size',
    'support_density', 'support_f1', 'routing_fp_support',
    'nuisance_fp_support',
]


def load_matches(results_dir, experiment):
    """Load one experiment's match-level rows (combined if present)."""
    experiment_dir = Path(results_dir) / experiment
    combined = experiment_dir / f'{experiment}_matches_combined.csv'
    if combined.exists():
        return pd.read_csv(combined)
    return combine_csv_files(experiment_dir.glob('**/root_*_matches.csv'))


def summarize_matches(matches):
    """One summary row per replicate fit (mean over matched factors).

    Random draws stay separate: partition_draw is part of the group keys.
    Aggregating over regions and draws is downstream analysis's job.
    """
    grouped = matches.groupby(WITHIN_FIT_KEYS, dropna=False)
    means = grouped[list(MATCH_MEAN_COLUMNS)].mean()
    counts = grouped.size().rename('n_matched_factors')
    standard_errors = (
        grouped[list(MATCH_MEAN_COLUMNS)].std()
        .div(np.sqrt(counts), axis=0)
        .rename(columns={column: f'{column}_se'
                         for column in MATCH_MEAN_COLUMNS})
    )
    return (
        means.join(standard_errors).join(counts).reset_index()
    )


def mid_noise_level(generator, matches):
    """Middle noise level with rows for a generator (eta/sigma grid)."""
    noise_name = GENERATOR_NOISE_NAME.get(generator, 'noise')
    noise_values = sorted(matches['noise'].unique())
    return noise_name, noise_values[len(noise_values) // 2] if noise_values else None


def panel_axis(ax, title, ylabel=None, xlabel=None):
    """Apply the shared quiet axis styling."""
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID_COLOR)
    ax.grid(True, axis='y', color=GRID_COLOR, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_PRIMARY, fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_PRIMARY, fontsize=8)


def _noise_label(noise_name, value):
    return f'{noise_name} = {value:g}'


def query(summary, generator, alpha=None, noise=None, estimator=None,
          corex_input=None, partition_method=None, n_per_region=None):
    """Filter the summary to one condition slice (None matches anything)."""
    frame = summary[summary['generator'] == generator]
    if alpha is not None:
        frame = frame[np.isclose(frame['alpha'], alpha)]
    if noise is not None:
        frame = frame[np.isclose(frame['noise'], noise)]
    if estimator is not None:
        frame = frame[frame['estimator'] == estimator]
    if corex_input is not None:
        frame = frame[frame['corex_input'] == corex_input]
    if partition_method is not None:
        if isinstance(partition_method, str):
            partition_method = (partition_method,)
        frame = frame[frame['partition_method'].isin(partition_method)]
    if n_per_region is not None:
        frame = frame[frame['n_per_region'] == n_per_region]
    return frame


def prepare_panel_frame(summary, generator, alpha, noise, estimator,
                        corex_input, n_per_region=None):
    """Filter to the plotted condition slice; average over draws ONLY."""
    frame = query(
        summary, generator, alpha=alpha, noise=noise, estimator=estimator,
        corex_input=corex_input, partition_method=PARTITION_ORDER,
        n_per_region=n_per_region,
    )
    if frame.empty:
        return frame
    # draw-level rows first average over regions, then we average draws:
    # draws are control replicates, so their spread stays available in data
    return frame.groupby(
        [key for key in WITHIN_FIT_KEYS if key != 'region'], dropna=False,
        as_index=False)['auprc'].mean()


def draw_method_lines(ax, frame, x_key, value_column):
    """One 2px line + marker per partition method in fixed color order."""
    for position, method in enumerate(PARTITION_ORDER):
        method_frame = frame[frame['partition_method'] == method]
        if method_frame.empty:
            continue
        ordered = method_frame.sort_values(x_key)
        ax.plot(
            ordered[x_key], ordered[value_column],
            color=SERIES_COLORS[position], linewidth=2.0,
            marker=SERIES_MARKERS[position], markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5,
            label=PARTITION_LABELS[method],
        )


def make_main_figure(summary, diagnostics=None, output_path=None,
                     alpha=0.5, estimation_estimator='bio',
                     corex_input='with_routing',
                     figure_title='Nonlinear higher-order recovery'):
    """2 generators x 3 panels: AUPRC vs alpha, AUPRC vs noise, diagnostics.

    Diagnostics need structure_diagnostics.csv columns
    (generator, group_size, noise, group_tc_bits,
    permuted_one_tc_mean_bits); the panel is skipped when the file is
    absent.
    """
    present = set(summary['generator'])
    generators = [g for g in ('parity', 'sine_sum') if g in present]
    rows = len(generators)
    fig, axes = plt.subplots(
        rows, 3, figsize=(9.0, 2.6 * rows), dpi=200, facecolor=SURFACE,
        squeeze=False,
    )
    fig.suptitle(figure_title, color=TEXT_PRIMARY, fontsize=11)

    for row, generator in enumerate(generators):
        ax_alpha, ax_noise, ax_diag = axes[row]

        # panel 1: AUPRC vs alpha, by partition method (mid noise level)
        noise_name, noise = mid_noise_level(generator, summary)
        frame = prepare_panel_frame(
            summary, generator, alpha, noise, estimation_estimator,
            corex_input,
        )
        draw_method_lines(ax_alpha, frame, 'alpha', 'auprc')
        panel_axis(
            ax_alpha,
            f"{generator}: AUPRC vs alpha ({_noise_label(noise_name, noise)})",
            ylabel=('matched AUPRC' if row == 0 else None),
            xlabel='alpha',
        )
        if row == 0:
            ax_alpha.legend(frameon=False, fontsize=7,
                            labelcolor=TEXT_SECONDARY)

        # panel 2: AUPRC vs noise, bio vs linear (oracle local fits)
        noises = sorted(summary[
            (summary['generator'] == generator)
            & np.isclose(summary['alpha'], alpha)
        ]['noise'].unique())
        for position, estimator in enumerate(('bio', 'linear')):
            oracle_rows = []
            for noise_value in noises:
                slice_frame = query(
                    summary, generator, alpha=alpha, noise=noise_value,
                    estimator=estimator, partition_method='oracle',
                )
                slice_frame = slice_frame[
                    slice_frame['corex_input'].str.contains('with_routing')
                ]
                if not slice_frame.empty:
                    oracle_rows.append(
                        (noise_value, slice_frame['auprc'].mean()))
            if oracle_rows:
                xs, ys = zip(*oracle_rows)
                ax_noise.plot(
                    xs, ys, color=SERIES_COLORS[position], linewidth=2.0,
                    marker=SERIES_MARKERS[position], markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=ESTIMATOR_LABELS[estimator],
                )
        panel_axis(
            ax_noise,
            f"{generator}: AUPRC vs noise (oracle)",
            xlabel=GENERATOR_NOISE_NAME.get(generator, 'noise'),
        )
        ax_noise.legend(frameon=False, fontsize=7,
                        labelcolor=TEXT_SECONDARY)

        # panel 3: structure diagnostics, group TC vs permuted TC (bits)
        if diagnostics is not None:
            diag = diagnostics[
                (diagnostics['generator'] == generator)
                & (diagnostics['group_size'] == 4)
            ].sort_values('noise')
            width = 0.35
            positions = np.arange(len(diag))
            if len(diag):
                ax_diag.bar(
                    positions - width / 2, diag['group_tc_bits'], width,
                    color=SERIES_COLORS[0], label='group TC',
                )
                ax_diag.bar(
                    positions + width / 2,
                    diag['permuted_one_tc_mean_bits'],
                    width, color=SERIES_COLORS[2], label='permuted TC (mean)',
                )
                ax_diag.set_xticks(positions)
                ax_diag.set_xticklabels(
                    [f'{value:g}' for value in diag['noise']],
                    color=TEXT_SECONDARY, fontsize=8)
        panel_axis(
            ax_diag,
            f"{generator}: group TC (d = 4, bits)",
            xlabel=GENERATOR_NOISE_NAME.get(generator, 'noise'),
            ylabel=('TC (bits)' if row == 0 else None),
        )
        if row == 0 and diagnostics is not None and len(diag):
            ax_diag.legend(frameon=False, fontsize=7,
                           labelcolor=TEXT_SECONDARY)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if output_path is not None:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"Figure saved: {output_path}")
    return fig


def load_diagnostics(results_dir):
    """Load structure diagnostics CSV if present (else None)."""
    path = Path(results_dir) / 'structure_diagnostics.csv'
    if path.exists():
        return pd.read_csv(path)
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Summarize nonlinear experiment CSVs and draw figures.',
    )
    parser.add_argument(
        '--results-dir', type=Path, default=Path('nonlinear_results'),
        help='Directory containing the experiment condition CSVs.',
    )
    parser.add_argument(
        '--experiments', nargs='+', default=['discrete_core',
                                             'continuous_core'],
        help='Experiments to include in the paper figure.',
    )
    parser.add_argument(
        '--alpha', type=float, default=0.5,
        help='Routing separation alpha for the figure panels.',
    )
    parser.add_argument(
        '--estimator', default='bio', choices=('bio', 'linear'),
        help='Estimator for the by-partition-method panel.',
    )
    parser.add_argument(
        '--output', type=Path, default=None,
        help='Output path for the paper figure (default: results_dir/figures).',
    )
    args = parser.parse_args()

    summaries = []
    for experiment in args.experiments:
        matches = load_matches(args.results_dir, experiment)
        if matches is None:
            print(f"No match rows found for '{experiment}'; skipping.")
            continue
        summary = summarize_matches(matches)
        summary['experiment'] = experiment
        summaries.append(summary)
    if not summaries:
        raise SystemExit('No experiment summaries could be built.')
    summary = pd.concat(summaries, ignore_index=True)
    summary_path = args.results_dir / 'nonlinear_summary.csv'
    summary.to_csv(summary_path, index=False)
    print(f"Summary written: {summary_path}")

    diagnostics = load_diagnostics(args.results_dir)
    output_path = args.output or (
        args.results_dir / 'figures' / 'nonlinear_main_figure.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    make_main_figure(
        summary,
        diagnostics=diagnostics,
        output_path=output_path,
        alpha=args.alpha,
        estimation_estimator=args.estimator,
    )


if __name__ == '__main__':
    main()