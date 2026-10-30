"""The §E3.5 contingency: parity capability diagnostic (Experiment 2 pattern).

The full grid (paper exp3.md §E3.11) shows ordinary EM collapsing the
parity fits into trivial basins at sufficient capacity — top-factor TC
near zero at n = 5000 in BOTH capacity arms, with the fits converging
early. This script answers the pre-registered capability question: is
that an expressiveness failure of the model family, or an optimization
failure?

Design, exactly the Experiment 2 pattern (its
structure_diagnostics.capability_diagnostic): on the parity group block
ALONE (no nuisance columns), one latent factor, both initializations at
the SAME 2^d-state cardinality (sufficient and conservative; noisy
parity is representable with 2^(d-1) states already):

- standard init: default Bio CorEx initialization, n_repeat = 5 EM
  restarts, best solution kept (the program's standard);
- atom-seeded init: the latent representation starts one-hot on the
  JOINT ATOM — each block row encoded by binary powers (2^d atoms cover
  every noisy-parity joint pattern; bit sums would collide under the
  parity constraint), n_repeat = 1. Flagged
  uses_ground_truth_init = True and permanently excluded from recovery
  summaries; it supports a capability statement only.

The reading: if std-init TC ~ 0 while seeded-init TC tracks the analytic
population TC at the same cardinality, the failure is optimization (EM
basins), not expressiveness. Nothing here is a recovery claim.

Results land in redundancy_results/parity_capability/ as restartable
per-condition CSVs (union of a resume run equals the serial run), with
the combined table and a per-(d, n) summary written at the end.

Usage (from this directory):

    python parity_capability.py [--repetitions 8]
        [--output-dir redundancy_results] [--overwrite] [--combine-only]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / 'gaussian'))
from result_io import append_run_log, combine_csv_files, write_csv_atomically  # noqa: E402

import corex_fitting  # noqa: E402
import redundancy_data as rdata  # noqa: E402
from seeds import derived_seed  # noqa: E402

REPS = 8
D_LEVELS = (3, 4, 5)
N_LEVELS = (500, 5000)
BIO_MAX_ITER = 200
INFO_COLUMNS = (
    'd', 'n_samples', 'root_seed', 'eta', 'tc_target', 'dim_hidden_states',
    'analytic_tc_bits', 'std_init_tc_bits', 'std_init_converged',
    'std_init_runtime_s', 'seeded_init_tc_bits', 'seeded_init_converged',
    'seeded_init_runtime_s', 'atom_count', 'uses_ground_truth_init',
    'fit_status', 'note',
)


def frozen_calibration():
    """{(d, generator): eta} — the target-calibrated flip rates."""
    return {
        (row['d'], row['generator']): row['eta']
        for row in rdata.calibrate(rdata.TC_TARGET)
    }


def run_condition(frozen_eta, d, n_samples, root):
    """One diagnostic condition: std-init vs atom-seeded-init Bio CorEx."""
    eta = frozen_eta[d, rdata.GEN_SYNERGISTIC]
    population = rdata.population_quantities(rdata.GEN_SYNERGISTIC, d, eta)
    card_states = 2 ** d  # exp2's convention: sufficient and conservative
    record = {
        'd': d,
        'n_samples': n_samples,
        'root_seed': root,
        'eta': float(eta),
        'tc_target': float(rdata.TC_TARGET),
        'dim_hidden_states': card_states,
        'analytic_tc_bits': float(population['tc_population']),
        'uses_ground_truth_init': True,
        'fit_status': 'ok',
        'note': ('standard EM vs atom-seeded one-hot init at the same '
                 '2^d-state cardinality on the parity group block; the '
                 'seeded fit is a capability statement only (flagged) and '
                 'never enters recovery summaries.'),
    }
    rng = np.random.default_rng(
        derived_seed(root, 'parity_block', d, n_samples))
    block = rdata.synergistic_group(rng, d, n_samples, eta)
    # 2^d atoms: binary powers encode every observed joint pattern exactly
    powers = 2 ** np.arange(d, dtype=np.int64)
    atom_keys = block.astype(np.int64) @ powers
    record['atom_count'] = int(len(np.unique(atom_keys)))

    for label, seed_purpose, n_repeat, atom in (
            ('std_init', 'std_init', 5, None),
            ('seeded_init', 'seeded_init', 1, atom_keys)):
        try:
            fit = corex_fitting.fit_bio_corex(
                block, 1,
                model_seed=derived_seed(root, seed_purpose, d, n_samples),
                dim_hidden=card_states, n_repeat=n_repeat,
                max_iter=BIO_MAX_ITER, atom_keys=atom)
        except Exception as error:  # noqa: BLE001 - failures-as-data
            record['fit_status'] = f'failed_{label}'
            record['note'] += f' [{label}: {type(error).__name__}]'
            print(f"  condition (d={d}, n={n_samples}, root={root}) "
                  f"{label} fit FAILED: {error}", flush=True)
            break
        record[f'{label}_tc_bits'] = float(fit['tcs'][0])
        record[f'{label}_converged'] = bool(fit['converged'])
        record[f'{label}_runtime_s'] = float(fit['fit_runtime_s'])
    return record


def combine(output_dir):
    """Write the combined condition table and per-(d, n) summary."""
    exp_dir = output_dir / 'parity_capability'
    files = sorted(exp_dir.glob('**/root_*_info.csv'))
    if not files:
        print(f"no results under {exp_dir}; nothing to combine")
        return
    combined = combine_csv_files(files)
    write_csv_atomically(
        combined, exp_dir / 'parity_capability_info_combined.csv')
    summary = combined.groupby(['d', 'n_samples'], as_index=False).agg(
        n=('root_seed', 'count'),
        analytic_tc_bits=('analytic_tc_bits', 'first'),
        std_init_tc_bits=('std_init_tc_bits', 'mean'),
        std_init_se=('std_init_tc_bits', lambda s: float(np.std(s, ddof=1))),
        seeded_init_tc_bits=('seeded_init_tc_bits', 'mean'),
        seeded_init_se=('seeded_init_tc_bits',
                        lambda s: float(np.std(s, ddof=1))),
        std_init_conv=('std_init_converged', 'mean'),
        seeded_init_conv=('seeded_init_converged', 'mean'),
    )
    write_csv_atomically(summary, exp_dir / 'parity_capability_summary.csv')
    combined_path = exp_dir / 'parity_capability_info_combined.csv'
    print(f"combined: {combined_path} ({len(combined)} conditions); "
          f"summary at {exp_dir / 'parity_capability_summary.csv'}")


def main():
    parser = argparse.ArgumentParser(
        description='Parity capability diagnostic (§E3.5 contingency).')
    parser.add_argument('--repetitions', type=int, default=REPS)
    parser.add_argument('--output-dir', type=Path,
                        default=Path('redundancy_results'))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--combine-only', action='store_true')
    args = parser.parse_args()

    exp_dir = args.output_dir / 'parity_capability'
    if args.combine_only:
        combine(args.output_dir)
        return
    frozen_eta = frozen_calibration()
    conditions = [
        (d, n_samples, 4100 + rep)
        for n_samples in N_LEVELS for d in D_LEVELS
        for rep in range(args.repetitions)
    ]
    started = time.time()
    append_run_log(args.output_dir, {
        'utc': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'event': 'run_start', 'experiment': 'parity_capability',
        'repetitions': args.repetitions,
    })
    written = skipped = 0
    for index, (d, n_samples, root) in enumerate(conditions, 1):
        cond_dir = exp_dir / f'd_{d}' / f'n_{n_samples}'
        cond_dir.mkdir(parents=True, exist_ok=True)
        info_path = cond_dir / f'root_{root:05d}_info.csv'
        if info_path.exists() and not args.overwrite:
            skipped += 1
            continue
        condition_started = time.time()
        record = run_condition(frozen_eta, d, n_samples, root)
        for column in INFO_COLUMNS:  # deterministic column order
            record.setdefault(column, np.nan)
        write_csv_atomically(
            pd.DataFrame([record])[list(INFO_COLUMNS)], info_path)
        metadata_path = info_path.with_suffix('.meta.json')
        metadata_path.write_text(
            json.dumps(
                record, indent=2, default=str))
        written += 1
        print(f"[{index}/{len(conditions)}] d={d} n={n_samples} root={root} "
              f"std={record.get('std_init_tc_bits')} "
              f"seeded={record.get('seeded_init_tc_bits')} "
              f"({time.time() - condition_started:.0f}s)", flush=True)
    append_run_log(args.output_dir, {
        'utc': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'event': 'run_end', 'experiment': 'parity_capability',
        'status': 'completed', 'conditions_written': written,
        'conditions_skipped': skipped,
        'elapsed_s': time.time() - started,
    })
    combine(args.output_dir)


if __name__ == '__main__':
    main()