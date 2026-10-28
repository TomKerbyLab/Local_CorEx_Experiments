"""Run the staged Gaussian Local CorEx experiments.

Each repetition and data-generating condition is saved separately. Completed
files are skipped by default, so interrupted experiments can be restarted
without repeating finished work. A combined CSV is rebuilt after the run.

Parallel execution: `parse_shard` and `enumerate_conditions` mirror the
sharding scheme of ../exp2_nonlinear/run_nonlinear_experiments.py — a
'k/N' shard attempts only every N-th condition of the deterministic
condition enumeration, so N parallel workers (parallel_runner.py in the
same directory here) reproduce the serial run exactly. A sharded run
skips the end-of-run combine (each shard's snapshot would be partial);
combine afterwards with --combine-only.
"""

import argparse
from pathlib import Path
from time import time

from result_io import append_run_log, combine_csv_files, write_csv_atomically
import simulation as lc_simulation


SETUP_FLAGS = {
    'homogeneous': {'same_dist': True, 'disjoint': False},
    'nondisjoint': {'same_dist': False, 'disjoint': False},
    'disjoint': {'same_dist': False, 'disjoint': True},
}


EXPERIMENTS = {
    'primary': {
        'sample_sizes': (50, 500, 5000),
        'alphas': (0.0, 0.5, 1.0),
        'setups': ('homogeneous', 'nondisjoint', 'disjoint'),
        'num_factors': (5,),
        'estimators': ('linear',),
        'partition_methods': ('oracle', 'phate', 'random'),
        'nuisance_dimensions': (0,),
        'repetitions': 16,
    },
    'factor_sensitivity': {
        'sample_sizes': (500,),
        'alphas': (0.5,),
        'setups': ('nondisjoint', 'disjoint'),
        'num_factors': (3, 4, 5, 6, 7),
        'estimators': ('linear',),
        'partition_methods': ('oracle', 'phate', 'random'),
        'nuisance_dimensions': (0,),
        'repetitions': 16,
    },
    'bio_check': {
        'sample_sizes': (500,),
        'alphas': (0.5,),
        'setups': ('nondisjoint', 'disjoint'),
        'num_factors': (5,),
        'estimators': ('linear', 'bio'),
        'partition_methods': ('oracle', 'phate', 'random'),
        'nuisance_dimensions': (0,),
        'repetitions': 16,
    },
    'nuisance_stress': {
        'sample_sizes': (500,),
        'alphas': (0.5,),
        'setups': ('nondisjoint', 'disjoint'),
        'num_factors': (5,),
        'estimators': ('linear',),
        'partition_methods': ('oracle', 'phate', 'random'),
        'nuisance_dimensions': (0, 25, 100, 475),
        'repetitions': 16,
    },
}


def condition_output_path(output_dir, experiment, setup, sample_size,
                          nuisance_dimensions, rep):
    """Return the CSV path for one restartable experimental condition."""
    condition_dir = (
        output_dir
        / experiment
        / setup
        / f'n_{sample_size}'
    )
    if nuisance_dimensions > 0:
        condition_dir = condition_dir / f'nuisance_{nuisance_dimensions}'
    return condition_dir / f'rep_{rep:02d}.csv'


def combine_results(output_dir, experiment):
    """Combine all completed condition files for one experiment."""
    experiment_dir = output_dir / experiment
    result_files = sorted(experiment_dir.glob('**/rep_*.csv'))
    combined = combine_csv_files(result_files)
    if combined is None:
        return None
    combined_path = experiment_dir / f'{experiment}_combined.csv'
    write_csv_atomically(combined, combined_path)
    return combined_path


def expected_fit_count(config, repetitions=None):
    """Count CorEx fits before launching a potentially long experiment."""
    num_repetitions = repetitions or config['repetitions']
    fits_per_alpha = (
        len(config['num_factors'])
        * len(config['estimators'])
        * (1 + 2 * len(config['partition_methods']))
    )
    return (
        len(config['sample_sizes'])
        * len(config['setups'])
        * len(config['nuisance_dimensions'])
        * num_repetitions
        * len(config['alphas'])
        * fits_per_alpha
    )


def parse_shard(shard):
    """Parse a 'k/N' shard spec into (k, n); 0 <= k < N required."""
    pieces = str(shard).split('/')
    if len(pieces) != 2:
        raise ValueError(f"Shard spec must be 'k/N', got {shard!r}.")
    k, n = int(pieces[0]), int(pieces[1])
    if n < 1:
        raise ValueError(f"Shard count must be >= 1, got {n}.")
    if not 0 <= k < n:
        raise ValueError(f"Shard index must satisfy 0 <= {k} < {n}.")
    return k, n


def enumerate_conditions(config, num_repetitions):
    """Enumerate every condition in the runner's deterministic loop order.

    The explicit list is what sharding slices: the nested loops below used
    to be the iteration itself; now they only build the list, and the run
    loop iterates over the (shard-filtered) list. Same order, same skip
    logic — this is purely the enumerable form of the old nest.
    """
    conditions = []
    for setup in config['setups']:
        for sample_size in config['sample_sizes']:
            for nuisance_dimensions in config['nuisance_dimensions']:
                for rep in range(num_repetitions):
                    conditions.append({
                        'setup': setup,
                        'sample_size': sample_size,
                        'nuisance_dimensions': nuisance_dimensions,
                        'rep': rep,
                    })
    return conditions


def condition_output_path_from(condition, output_dir, experiment):
    return condition_output_path(
        output_dir, experiment, condition['setup'],
        condition['sample_size'], condition['nuisance_dimensions'],
        condition['rep'],
    )


def run_experiment(experiment, output_dir, repetitions=None,
                   overwrite=False, dry_run=False, shard=None,
                   combine_only=False):
    """Run one named experiment configuration.

    With ``shard`` as 'k/N', only every N-th condition (indices k, k+N,
    ...) of the enumerated list is attempted, so identical shard
    invocations can run as parallel workers without overlap.
    """
    config = EXPERIMENTS[experiment]
    num_repetitions = repetitions or config['repetitions']
    all_conditions = enumerate_conditions(config, num_repetitions)
    conditions = all_conditions
    if combine_only:
        combined_path = combine_results(output_dir, experiment)
        print(f"Combined results: {combined_path}" if combined_path
              else "No condition files to combine.")
        return combined_path
    if shard is not None:
        shard_k, shard_n = parse_shard(shard)
        conditions = all_conditions[shard_k::shard_n]
    print(
        f"Experiment: {experiment}"
        + (f", shard {shard}" if shard is not None else "")
        + f" ({len(conditions)} conditions of {len(all_conditions)})"
    )
    print(f"Expected CorEx fits across the full grid: "
          f"{expected_fit_count(config, num_repetitions):,}")
    if dry_run:
        return None

    start_time = time()
    append_run_log(output_dir, {
        'event': 'run_start',
        'experiment': experiment,
        'repetitions': num_repetitions,
        **({'shard': shard} if shard is not None else {}),
    })
    conditions_written = 0
    conditions_skipped = 0
    failed_conditions = []
    status = 'completed'
    try:
        for condition in conditions:
            setup = condition['setup']
            setup_flags = SETUP_FLAGS[setup]
            output_path = condition_output_path_from(
                condition, output_dir, experiment)
            if output_path.exists() and not overwrite:
                conditions_skipped += 1
                print(f"Skipping completed condition: {output_path}")
                continue

            print(
                f"Running {experiment}: setup={setup}, "
                f"n={condition['sample_size']}, "
                f"nuisance={condition['nuisance_dimensions']}, "
                f"rep={condition['rep']}"
            )
            # One failing condition must not abort the whole
            # experiment: the rest of the grid still fits, and the
            # failed condition is retried on the next invocation.
            try:
                results = lc_simulation.run_simulation(
                    rep=condition['rep'],
                    cluster_size=condition['sample_size'],
                    alphas=config['alphas'],
                    num_factors=config['num_factors'],
                    estimators=config['estimators'],
                    partition_methods=config['partition_methods'],
                    same_dist=setup_flags['same_dist'],
                    disjoint=setup_flags['disjoint'],
                    nuisance_dimensions=condition['nuisance_dimensions'],
                    progress=True,
                )
            except Exception as exc:
                print(f"FAILED condition {output_path}: "
                      f"{type(exc).__name__}: {exc}")
                failed_conditions.append(output_path)
                continue
            results.insert(0, 'experiment', experiment)
            write_csv_atomically(results, output_path)
            conditions_written += 1

        # run_end (logged in finally) should distinguish a fully clean run
        # from one that finished with failed conditions.
        if failed_conditions:
            status = 'completed_with_failures'

        # a sharded run leaves the combine to a final --combine-only pass:
        # each shard's own end-of-run snapshot would otherwise be partial
        # and the last one to finish would win
        combined_path = None
        if shard is None:
            combined_path = combine_results(output_dir, experiment)
    except BaseException:
        status = 'failed'
        raise
    finally:
        append_run_log(output_dir, {
            'event': 'run_end',
            'experiment': experiment,
            'status': status,
            'conditions_written': conditions_written,
            'conditions_skipped': conditions_skipped,
            'elapsed_s': round(time() - start_time, 3),
        })

    if failed_conditions:
        status = 'completed_with_failures'
        print(f"{len(failed_conditions)} condition(s) failed:")
        for path in failed_conditions:
            print(f"  {path}")
        append_run_log(output_dir, {
            'event': 'condition_failures',
            'experiment': experiment,
            'n_failed': len(failed_conditions),
            'conditions': [str(path) for path in failed_conditions],
        })

    if combined_path is not None:
        print(f"Combined results: {combined_path}")
    return combined_path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run Gaussian Local CorEx experiments.',
    )
    parser.add_argument(
        'experiment',
        choices=tuple(EXPERIMENTS),
        help='Experiment configuration to run.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('ablation_data'),
        help='Directory in which condition files and combined results are saved.',
    )
    parser.add_argument(
        '--repetitions',
        type=int,
        help='Override the configured repetition count, useful for smoke tests.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Rerun conditions whose CSV files already exist.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the expected fit count without running models.',
    )
    parser.add_argument(
        '--shard',
        type=str,
        help="Run only every N-th condition: 'k/N' with 0 <= k < N. "
             "Shards are disjoint and cover all conditions exactly, so "
             "N parallel workers reproduce the serial run's outputs.",
    )
    parser.add_argument(
        '--combine-only',
        action='store_true',
        help='Skip fitting; recombine the per-condition CSVs that exist '
             'under the output directory (used after a sharded run).',
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    run_experiment(
        experiment=args.experiment,
        output_dir=args.output_dir,
        repetitions=args.repetitions,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        shard=args.shard,
        combine_only=args.combine_only,
    )


if __name__ == '__main__':
    main()
