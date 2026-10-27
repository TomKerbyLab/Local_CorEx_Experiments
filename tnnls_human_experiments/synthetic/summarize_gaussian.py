"""Summarize the Gaussian experiment outputs; never reruns models.

Reads each experiment's combined CSV under ``ablation_data/<experiment>/``
and produces three tables under ``ablation_data/summaries/``:

- ``gaussian_replicate_aggregates.csv`` — one row per (replicate, condition,
  method), averaging the two regional ground-truth rows of each fit within
  the replicate;
- ``gaussian_summary.csv`` — condition-level mean, SE, t-based 95% CI, and
  replicate count per metric;
- ``gaussian_contrasts.csv`` — paired differences of each local method vs
  global and vs oracle, replicate-aligned, mirroring the reference
  pipeline's contrasts.

Random-partition draws are kept separate through aggregation: the Gaussian
pipeline draws one partition per replicate, so the replicate already is the
draw, and collapsing over it happens downstream.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENTS = ('primary', 'factor_sensitivity', 'bio_check', 'nuisance_stress')

# Methods in fixed order; colors/markers slot into this order downstream.
METHOD_ORDER = ('global', 'oracle', 'phate', 'random')

METRICS = (
    'matched_average_precision',
    'matched_adjusted_average_precision',
    'matched_cosine_similarity',
    'matched_top_k_f1',
    'balanced_average_precision',
    'balanced_cosine_similarity',
)

REPLICATE_KEYS = (
    'experiment', 'rep_num', 'covariance_setup', 'cluster_size', 'alpha',
    'nuisance_dimensions', 'num_factors', 'estimator', 'method',
)
CONDITION_KEYS = tuple(k for k in REPLICATE_KEYS if k != 'rep_num')


def load_combined(data_dir, experiment):
    """Load one experiment's combined results CSV."""
    combined = Path(data_dir) / experiment / f'{experiment}_combined.csv'
    if not combined.exists():
        print(f'No combined results for {experiment}; skipping.')
        return None
    return pd.read_csv(combined)


def method_of(frame):
    """Derive the method column: fit scope or partition method."""
    return np.where(frame['scope'] == 'global', 'global',
                    frame['partition_method'])


def aggregate_replicates(frame):
    """Average the two regional rows of each fit within one replicate.

    The global fit contributes a row per regional ground truth and each
    local fit is scored against its own region's ground truth, so averaging
    over regions puts every method on the same within-replicate footing.
    """
    frame = frame.copy()
    frame['method'] = method_of(frame)
    grouped = frame.groupby(list(REPLICATE_KEYS), dropna=False)
    return grouped[list(METRICS)].mean().reset_index()


def summarize_conditions(aggregates):
    """Condition-level mean, SE, and t-based 95% CI per metric."""
    grouped = aggregates.groupby(list(CONDITION_KEYS), dropna=False)
    means = grouped[list(METRICS)].mean()
    sizes = grouped.size().rename('n_reps')
    ses = grouped[list(METRICS)].std().div(
        np.sqrt(sizes.to_numpy()), axis=0).rename(
        columns=lambda c: f'{c}_se')
    summary = means.join(ses).join(sizes).reset_index()
    crit_values = summary['n_reps'].map(lambda n: t_critical(int(n)))
    for metric in METRICS:
        se = summary[f'{metric}_se']
        summary[f'{metric}_ci95_low'] = summary[metric] - crit_values * se
        summary[f'{metric}_ci95_high'] = summary[metric] + crit_values * se
    return summary


def t_critical(n, confidence=0.95):
    """Two-sided t critical value with n - 1 degrees of freedom."""
    from scipy import stats

    return float(stats.t.ppf(1 - (1 - confidence) / 2, n - 1))


def paired_difference_summary(x, y, confidence=0.95):
    """Paired differences x - y with mean, se, and CI over paired reps."""
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        return {'mean_diff': np.nan, 'se_diff': np.nan, 'ci_low': np.nan,
                'ci_high': np.nan, 'n_pairs': 0}
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    if n > 1:
        crit = float(stats.t.ppf(1 - (1 - confidence) / 2, n - 1))
        lo, hi = mean - crit * se, mean + crit * se
    else:
        lo = hi = mean
    return {'mean_diff': mean, 'se_diff': se, 'ci_low': float(lo),
            'ci_high': float(hi), 'n_pairs': n}


def paired_contrasts(aggregates, contrast_metrics=None):
    """Paired differences of each local method vs global and vs oracle."""
    contrast_metrics = contrast_metrics or (
        'matched_average_precision', 'matched_cosine_similarity',
        'balanced_average_precision', 'matched_top_k_f1',
    )
    # method itself must not be a grouping key: the contrast finds both of
    # its arms within one condition group
    contrast_keys = [k for k in CONDITION_KEYS if k != 'method']
    rows = []
    for condition, grp in aggregates.groupby(contrast_keys, dropna=False):
        for method in (m for m in METHOD_ORDER
                       if m not in ('global', 'oracle')):
            sub = grp[grp['method'] == method]
            if sub.empty:
                continue
            for vs in ('global', 'oracle'):
                base = grp[grp['method'] == vs]
                if base.empty:
                    continue
                merged = sub.merge(
                    base, on='rep_num', suffixes=('', '_base'))
                for metric in contrast_metrics:
                    if metric not in sub.columns:
                        continue
                    result = paired_difference_summary(
                        merged[metric].to_numpy(float),
                        merged[f'{metric}_base'].to_numpy(float))
                    row = dict(zip(contrast_keys, condition))
                    rows.append({
                        **row,
                        'method': method, 'vs': vs, 'metric': metric,
                        **result,
                    })
    return pd.DataFrame(rows)


def main(data_dir='ablation_data'):
    """Summarize all four experiments; write the three tables."""
    aggregates_frames = []
    summaries = []
    contrasts = []
    for experiment in EXPERIMENTS:
        frame = load_combined(data_dir, experiment)
        if frame is None:
            continue
        aggregates = aggregate_replicates(frame)
        aggregates_frames.append(aggregates)
        summaries.append(summarize_conditions(aggregates))
        contrasts.append(paired_contrasts(aggregates))

    if not aggregates_frames:
        raise SystemExit('No combined results found; run experiments first.')

    all_aggregates = pd.concat(aggregates_frames, ignore_index=True)
    all_summary = pd.concat(summaries, ignore_index=True)
    all_contrasts = pd.concat(contrasts, ignore_index=True)

    out_dir = Path(data_dir) / 'summaries'
    out_dir.mkdir(parents=True, exist_ok=True)
    all_aggregates.to_csv(out_dir / 'gaussian_replicate_aggregates.csv',
                          index=False)
    all_summary.to_csv(out_dir / 'gaussian_summary.csv', index=False)
    all_contrasts.to_csv(out_dir / 'gaussian_contrasts.csv', index=False)
    print(f'aggregate rows: {len(all_aggregates)}; '
          f'summary rows: {len(all_summary)}; '
          f'contrast rows: {len(all_contrasts)}')
    print(f'summaries written to {out_dir}/')
    return all_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Summarize Gaussian experiment results.',
    )
    parser.add_argument('--data-dir', default='ablation_data',
                        help='Directory holding per-experiment subfolders.')
    args = parser.parse_args()
    main(data_dir=args.data_dir)