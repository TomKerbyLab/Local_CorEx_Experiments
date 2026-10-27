"""Favorable-conditions pilot for the nonlinear experiments (Appendix E, 7).

The decided pilot before any full nonlinear batch: large n, low noise,
oracle partition (plus the global fit), no_routing input, NO nuisance —
the settings where localization should demonstrably help if it helps at
all, read against the pure-block capability diagnostic (standard vs
atom-seeded init) at the same sample size. Two readouts per generator:

1. Pipeline level (standard init only): global and oracle-local Bio/Linear
   CorEx fits, scored the primary way; the seeded-init arm stays a
   capability statement (ground-truth-informed init) and is never mixed
   into recovery summaries.
2. Pure-block level: structure_diagnostics.capability_diagnostic at the
   same n, giving standard-init vs seeded-init explained TC.

Determines whether sine-sum stays the intended primary demonstration; the
results CSVs land in <output-dir>/ for inspection.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2] / 'gaussian'))
from result_io import write_csv_atomically  # noqa: E402

import structure_diagnostics  # noqa: E402
from config import DEFAULTS, ROOT_SEED_BASE  # noqa: E402
from run_nonlinear_experiments import (  # noqa: E402
    derived_seed,
    fit_matrices,
    fit_variant,
    generate_condition_data,
    global_cells,
    oracle_cells,
    score_region_fit,
    variant_specs,
)

PILOT_N = 5000
PILOT_ALPHA = 0.5
NOISE_LEVELS = {'parity': (0.05,), 'sine_sum': (0.05,)}  # lowest nonzero
PILOT_REPS = 2  # enough to see the pattern; full grids follow if favorable
PILOT_CONFIG = {
    'estimators': ('bio', 'linear'),
    'corex_inputs': ('no_routing',),
    'native_gaussian_bio': False,
}


def pilot_conditions():
    conditions = []
    for generator, noises in NOISE_LEVELS.items():
        for rep in range(PILOT_REPS):
            for noise in noises:
                conditions.append({
                    'generator': generator, 'n_per_region': PILOT_N,
                    'alpha': PILOT_ALPHA, 'noise': noise,
                    'n_nuisance': 0,
                    'root_seed': ROOT_SEED_BASE[generator] + rep,
                })
    return conditions


def run_pilot_condition(condition):
    """Global + oracle cells, standard init, primary scoring."""
    dataset = generate_condition_data(condition)
    specs = variant_specs(condition['generator'], PILOT_CONFIG)
    matrices = fit_matrices(dataset, specs, DEFAULTS)
    base_row = {key: condition[key] for key in (
        'generator', 'n_per_region', 'alpha', 'noise', 'n_nuisance',
        'root_seed')}
    rows = []
    for cell in global_cells(dataset) + oracle_cells(dataset):
        for spec in specs:
            cell_matrix = matrices[spec['name']]['matrix'][cell['rows'], :]
            fit, fit_fields = fit_variant(
                dataset, cell_matrix, spec, DEFAULTS['n_factors'],
                derived_seed(
                    condition['root_seed'], 'pilot', cell['draw'],
                    cell['method'], cell['name'], spec['estimator'],
                    spec['name']),
                DEFAULTS,
            )
            for served_region in cell['regions']:
                row = {
                    **base_row,
                    'partition_method': cell['method'],
                    'estimator': spec['estimator'],
                    'corex_input': spec['name'],
                }
                if fit is None:
                    rows.append({**row, 'fit_status': 'failed_fit',
                                 'note': fit_fields['note']})
                    continue
                rows.extend(score_region_fit(row, fit, served_region))
    return rows


def capability_conditions():
    """Pure-block capability rows at the pilot's sample size (parity only).

    The atom-seeded init is defined on the discrete parity generator; the
    continuous generator has no ground-truth joint atom to one-hot on.
    """
    rows = []
    for eta in (0.0, 0.05):
        rows.append(
            structure_diagnostics.capability_diagnostic(
                group_size=4, eta=eta, n=PILOT_N,
                seed=derived_seed(ROOT_SEED_BASE['parity'], 'capability'))
            | {'generator': 'parity', 'root_seed': ROOT_SEED_BASE['parity']})
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Run the favorable nonlinear pilot.',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('pilot_data'),
        help='Directory for pilot CSVs.',
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for condition in pilot_conditions():
        print(
            f"Pilot: {condition['generator']} n={condition['n_per_region']} "
            f"noise={condition['noise']} nuisance=0 root="
            f"{condition['root_seed']}",
            flush=True,
        )
        all_rows.extend(run_pilot_condition(condition))
    matches = pd.DataFrame(all_rows)
    write_csv_atomically(matches, output_dir / 'pilot_matches.csv')
    print(f"Pilot matches written: {output_dir / 'pilot_matches.csv'}")

    capability = pd.DataFrame(capability_conditions())
    write_csv_atomically(capability, output_dir / 'pilot_capability.csv')
    print(f"Pilot capability written: {output_dir / 'pilot_capability.csv'}")


if __name__ == '__main__':
    main()