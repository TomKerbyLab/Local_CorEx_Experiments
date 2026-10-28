"""Run one experiment as a pool of parallel shard workers.

Every condition is deterministic and self-contained (its data generation,
its seed derivations, and its fits touch nothing outside itself), so the
condition list can be split by stride selection and the shards executed
in independent processes: ``runner.run_experiment(shard='k/N')`` attempts
only conditions whose list index satisfies ``index % N == k``, and the
union of shards equals the serial run exactly (test:
``test_shard_partition_is_exact_disjoint_cover``).

This script spawns one subprocess per shard (each a normal runner
invocation with ``--shard k/N``), caps each worker's BLAS thread pools at
1 (the fits are small-matrix, single-thread-bound EM loops; one thread
per worker avoids oversubscription), captures per-shard logs under the
output directory, and recombines per-condition CSVs when all shards
finish. Interrupted batches restart cleanly: completed conditions are
skipped by default, so relaunching with the same command resumes.

Sharded runs leave the results combine to the final ``--combine-only``
pass here; each shard's own end-of-run snapshot would be partial.

Usage (from this directory, or `uv run python ...` from the repo root):

    python parallel_runner.py discrete_core --workers 16 \
        --output-dir nonlinear_results
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# the fits spend their time in Python-level EM loops over small matrices;
# force one BLAS thread per worker so N workers use N cores, not N * ncpu
WORKER_ENV = {
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run one experiment as parallel shard workers.',
    )
    parser.add_argument(
        'experiment',
        help='The runner experiment configuration to run.',
    )
    parser.add_argument(
        '--workers',
        type=int,
        required=True,
        help='Number of shard processes (each worker gets every N-th '
             'condition; N = workers).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('nonlinear_results'),
        help='Directory for condition files, shard logs, and combined '
             'results.',
    )
    parser.add_argument(
        '--repetitions',
        type=int,
        help='Override the configured repetition count.',
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
        '--shard-command',
        type=str,
        default=str(sys.executable),
        help='Python executable used for shard subprocesses (default: this '
             "process's interpreter).",
    )
    return parser.parse_args()


def shard_command(args, shard):
    command = [
        args.shard_command, 'run_nonlinear_experiments.py',
        args.experiment,
        '--output-dir', str(args.output_dir),
        '--shard', f'{shard}/{args.workers}',
    ]
    if args.repetitions:
        command += ['--repetitions', str(args.repetitions)]
    if args.include_no_routing:
        command += ['--include-no-routing']
    if args.overwrite:
        command += ['--overwrite']
    return command


def worker_env():
    """Parent environment with single-thread BLAS caps for one worker."""
    return {**os.environ, **WORKER_ENV}


def main():
    args = parse_arguments()
    log_dir = args.output_dir / 'shard_logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    processes = []
    print(
        f"Launching {args.workers} shard workers for {args.experiment}; "
        f"logs under {log_dir}",
        flush=True,
    )
    for shard in range(args.workers):
        log_path = log_dir / f'shard_{shard}.log'
        log_file = log_path.open('w')
        process = subprocess.Popen(
            shard_command(args, shard),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=worker_env(),
        )
        processes.append((shard, log_path, process))
        print(f"  shard {shard}/{args.workers} -> pid {process.pid}",
              flush=True)

    failures = []
    for shard, log_path, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((shard, log_path, return_code))
            print(f"  shard {shard}/{args.workers} FAILED "
                  f"(exit {return_code}); see {log_path}", flush=True)
        else:
            print(f"  shard {shard}/{args.workers} done", flush=True)
    elapsed = time.time() - started

    if failures:
        print(
            f"{len(failures)}/{args.workers} shards failed after "
            f"{elapsed:.0f}s; no combine attempted. Failed logs: "
            + ', '.join(str(path) for _, path, _ in failures)
        )
        sys.exit(1)

    # one serial pass to combine per-condition files (no extra interpreter
    # environment concerns: this is the same runner, --combine-only skips
    # every fit)
    command = [
        args.shard_command, 'run_nonlinear_experiments.py',
        args.experiment,
        '--output-dir', str(args.output_dir),
        '--combine-only',
    ]
    print(f"All shards done in {elapsed / 3600:.1f}h; combining...",
          flush=True)
    combine = subprocess.run(command, env=worker_env())
    print(f"Combine exit code: {combine.returncode}")


if __name__ == '__main__':
    main()