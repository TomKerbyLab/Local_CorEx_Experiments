"""Draw the Gaussian experiment figures from the summary tables.

Reads ``ablation_data/summaries/gaussian_summary.csv`` (produced by
``summarize_gaussian.py``) and writes one PNG per experiment grid under
``ablation_data/figures/``. Visual conventions follow the nonlinear
pipeline's main figure: fixed method color order (identity never rides the
plotted value), markers as a second identity channel, quiet axes.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summarize_gaussian import METHOD_ORDER, load_combined

# categorical palette (validated default instance): slots 1..4 in fixed
# order; markers provide a second identity channel for grayscale/CVD print
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
SERIES_MARKERS = ['o', 's', '^', 'D']
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID_COLOR = '#e5e3df'
SURFACE = '#fcfcfb'

METHOD_LABELS = {'global': 'global', 'oracle': 'oracle',
                 'phate': 'PHATE', 'random': 'random'}
SETUP_LABELS = {'homogeneous': 'homogeneous', 'nondisjoint': 'nondisjoint',
                'disjoint': 'disjoint'}
SETUP_ORDER = ('homogeneous', 'nondisjoint', 'disjoint')

PRIMARY_METRIC = 'matched_average_precision'
METRIC_LABEL = {'matched_average_precision': 'matched AUPRC',
                'matched_cosine_similarity': 'matched cosine similarity',
                'balanced_average_precision': 'balanced AUPRC'}


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


def method_frame(summary, setup, metric=PRIMARY_METRIC, **filters):
    """Filter the summary to one setup slice with extra equality filters."""
    frame = summary[summary['covariance_setup'] == setup]
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (int, float)):
            frame = frame[np.isclose(frame[key].astype(float), value)]
        else:
            frame = frame[frame[key].isin(value if isinstance(value, tuple)
                                         else (value,))]
    return frame


def draw_method_lines(ax, frame, x_key, metric=PRIMARY_METRIC):
    """One 2px line + marker per method in fixed color order (mean ± SE)."""
    for position, method in enumerate(METHOD_ORDER):
        method_frame = frame[frame['method'] == method].sort_values(x_key)
        if method_frame.empty:
            continue
        x = method_frame[x_key].to_numpy(float)
        mean = method_frame[metric].to_numpy(float)
        se = method_frame[f'{metric}_se'].to_numpy(float)
        ax.plot(x, mean,
                color=SERIES_COLORS[position], linewidth=2.0,
                marker=SERIES_MARKERS[position], markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=METHOD_LABELS[method])
        ax.fill_between(x, mean - se, mean + se,
                        color=SERIES_COLORS[position], alpha=0.12,
                        linewidth=0)


def finish_lines_figure(fig, output_path):
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(output_path, dpi=200)
    print(f'figure written: {output_path}')
    plt.close(fig)


def make_primary_figure(summary, output_path):
    """2 rows x 3 setups: matched AUPRC vs n (top) and vs alpha (bottom).

    Top row holds sample size fixed at alpha = 0.5; bottom row holds
    n = 500 per region. The SE band is the across-replicate spread.
    """
    setups = [s for s in SETUP_ORDER if s in set(summary['covariance_setup'])]
    fig, axes = plt.subplots(
        2, len(setups), figsize=(2.9 * len(setups), 4.8), dpi=200,
        facecolor=SURFACE, squeeze=False, sharey='row',
    )
    for column, setup in enumerate(setups):
        top = method_frame(summary, setup, alpha=0.5)
        draw_method_lines(axes[0][column], top, 'cluster_size')
        panel_axis(axes[0][column], SETUP_LABELS[setup],
                   ylabel='matched AUPRC' if column == 0 else None,
                   xlabel='n per region')
        axes[0][column].set_xscale('log')

        bottom = method_frame(summary, setup, cluster_size=500)
        draw_method_lines(axes[1][column], bottom, 'alpha')
        panel_axis(axes[1][column], None,
                   ylabel='matched AUPRC' if column == 0 else None,
                   xlabel='alpha')
        if column == 0:
            axes[0][column].legend(
                frameon=False, fontsize=7, labelcolor=TEXT_SECONDARY,
                loc='lower left')
    fig.suptitle('Gaussian grouped-covariance recovery (primary)',
                 color=TEXT_PRIMARY, fontsize=11)
    finish_lines_figure(fig, output_path)


def make_bio_check_figure(summary, output_path):
    """2 setups x 2 estimators grid: matched AUPRC vs partition method.

    Bio CorEx and Linear CorEx appear as two lines per panel over the
    methods, at n = 500, alpha = 0.5, M = 5. Estimator identity is fixed
    order: linear first, bio second.
    """
    estimator_order = ('linear', 'bio')
    estimator_colors = (SERIES_COLORS[0], SERIES_COLORS[1])
    estimator_markers = (SERIES_MARKERS[0], SERIES_MARKERS[1])
    estimator_labels = ('Linear CorEx', 'Bio CorEx')

    setups = [s for s in ('nondisjoint', 'disjoint')
              if s in set(summary['covariance_setup'])]
    method_x = {method: position
                for position, method in enumerate(METHOD_ORDER)}
    fig, axes = plt.subplots(
        1, len(setups), figsize=(2.9 * len(setups), 2.7), dpi=200,
        facecolor=SURFACE, squeeze=False, sharey=True,
    )
    labeled_estimators = set()
    for column, setup in enumerate(setups):
        frame = method_frame(summary, setup)
        ax = axes[0][column]
        for x_position, method in enumerate(METHOD_ORDER):
            estimator_frame = frame[frame['method'] == method]
            if estimator_frame[estimator_frame['estimator'] == 'linear'].empty:
                continue
            for position, estimator in enumerate(estimator_order):
                row = estimator_frame[estimator_frame['estimator'] == estimator]
                if row.empty:
                    continue
                mean = float(row[PRIMARY_METRIC].iloc[0])
                se = float(row[f'{PRIMARY_METRIC}_se'].iloc[0])
                # dodge the two estimators around each method slot so an
                # overlap at an identical value still shows both identities
                x = x_position + (position - 0.5) * 0.14
                label = (estimator_labels[position]
                         if estimator not in labeled_estimators
                         and column == 0 else None)
                ax.errorbar(
                    x, mean, yerr=se,
                    color=estimator_colors[position], linewidth=2.0,
                    marker=estimator_markers[position], markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5,
                    capsize=3, label=label,
                )
                if column == 0:
                    labeled_estimators.add(estimator)
        ax.set_xticks(list(method_x.values()),
                      [METHOD_LABELS[m] for m in METHOD_ORDER])
        panel_axis(axes[0][column], SETUP_LABELS[setup],
                   ylabel='matched AUPRC' if column == 0 else None,
                   xlabel='partition method')
        if column == 0:
            ax.legend(frameon=False, fontsize=7,
                      labelcolor=TEXT_SECONDARY, loc='lower left')
    fig.suptitle('Bio CorEx check vs Linear CorEx',
                 color=TEXT_PRIMARY, fontsize=11)
    finish_lines_figure(fig, output_path)


def make_nuisance_figure(summary, output_path):
    """2 setups: matched AUPRC vs nuisance dimensions, by method."""
    setups = [s for s in ('nondisjoint', 'disjoint')
              if s in set(summary['covariance_setup'])]
    fig, axes = plt.subplots(
        1, len(setups), figsize=(2.9 * len(setups), 2.7), dpi=200,
        facecolor=SURFACE, squeeze=False,
    )
    for column, setup in enumerate(setups):
        frame = method_frame(summary, setup, cluster_size=500)
        draw_method_lines(axes[0][column], frame, 'nuisance_dimensions')
        panel_axis(axes[0][column], SETUP_LABELS[setup],
                   ylabel='matched AUPRC' if column == 0 else None,
                   xlabel='nuisance dimensions')
        if column == 0:
            axes[0][column].legend(frameon=False, fontsize=7,
                                   labelcolor=TEXT_SECONDARY, loc='lower left')
    fig.suptitle('Nuisance-dimension stress (n = 500, alpha = 0.5)',
                 color=TEXT_PRIMARY, fontsize=11)
    finish_lines_figure(fig, output_path)


def make_factor_sensitivity_figure(summary, output_path):
    """2 setups: matched AUPRC vs requested factors M, by method."""
    setups = [s for s in ('nondisjoint', 'disjoint')
              if s in set(summary['covariance_setup'])]
    fig, axes = plt.subplots(
        1, len(setups), figsize=(2.9 * len(setups), 2.7), dpi=200,
        facecolor=SURFACE, squeeze=False,
    )
    for column, setup in enumerate(setups):
        frame = method_frame(summary, setup, cluster_size=500)
        draw_method_lines(axes[0][column], frame, 'num_factors')
        panel_axis(axes[0][column], SETUP_LABELS[setup],
                   ylabel='matched AUPRC' if column == 0 else None,
                   xlabel='requested factors M')
        axes[0][column].xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True))
        if column == 0:
            axes[0][column].legend(frameon=False, fontsize=7,
                                   labelcolor=TEXT_SECONDARY, loc='lower left')
    fig.suptitle('Factor-count sensitivity (n = 500, alpha = 0.5)',
                 color=TEXT_PRIMARY, fontsize=11)
    finish_lines_figure(fig, output_path)


def main(data_dir='ablation_data'):
    """Summarize (if needed) and draw the four experiment figures."""
    summaries_dir = Path(data_dir) / 'summaries'
    summary_path = summaries_dir / 'gaussian_summary.csv'
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        from summarize_gaussian import main as summarize_main
        summarize_main(data_dir=data_dir)
        summary = pd.read_csv(summary_path)

    figures_dir = Path(data_dir) / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    for experiment, name in (
        ('primary', 'gaussian_primary_figure.png'),
        ('bio_check', 'gaussian_bio_check_figure.png'),
        ('nuisance_stress', 'gaussian_nuisance_figure.png'),
        ('factor_sensitivity', 'gaussian_factor_sensitivity_figure.png'),
    ):
        experiment_summary = summary[summary['experiment'] == experiment]
        if experiment_summary.empty:
            print(f'No summary rows for {experiment}; skipping.')
            continue
        output_path = figures_dir / name
        if experiment == 'primary':
            make_primary_figure(experiment_summary, output_path)
        elif experiment == 'bio_check':
            make_bio_check_figure(experiment_summary, output_path)
        elif experiment == 'nuisance_stress':
            make_nuisance_figure(experiment_summary, output_path)
        else:
            make_factor_sensitivity_figure(experiment_summary, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Draw the Gaussian experiment figures.',
    )
    parser.add_argument('--data-dir', default='ablation_data',
                        help='Directory holding per-experiment subfolders.')
    args = parser.parse_args()
    main(data_dir=args.data_dir)