"""Run one Gaussian experiment as a pool of parallel shard workers.

Mirrors ../exp2_nonlinear/parallel_runner.py for the Gaussian runner:
every condition writes its own CSV, the runner enumerates its conditions
deterministically, and `--shard k/N` attempts only every N-th condition —
so the union of N shard workers equals the serial run exactly. Interrupted
batches restart by relaunching this script with the same command.

Usage (from this directory, or `uv run python ...` from the repo root):

    python parallel_runner.py primary --workers 16 --output-dir ablation_data
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# one BLAS thread per worker: the CorEx fits here are small-matrix loop
# work (n <= 5000), so N workers should use N cores, not N * ncpu
WORKER_ENV = {
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run Gaussian Local CorEx experiments in parallel.',
    )
    parser.add_argument(
        'experiment',
        help='The runner experiment configuration to run '
             '(primary, factor_sensitivity, bio_check, nuisance_stress).',
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
        default=Path('ablation_data'),
        help='Directory for condition files, shard logs, and combined '
             'results.',
    )
    parser.add_argument(
        '--repetitions',
        type=int,
        help='Override the configured repetition count.',
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
        args.shard_command, 'run_gaussian_experiments.py',
        args.experiment,
        '--output-dir', str(args.output_dir),
        '--shard', f'{shard}/{args.workers}',
    ]
    if args.repetitions:
        command += ['--repetitions', str(args.repetitions)]
    if args.overwrite:
        command += ['--overwrite']
    return command


def worker_env():
    """Parent environment with single-thread BLAS caps for one worker."""
    return {**os.environ, **WORKER_ENV}


def main():
    args = parse_arguments()
    # namespace by experiment: two launchers sharing one output directory
    # must not overwrite each other's per-shard logs
    log_dir = args.output_dir / 'shard_logs' / args.experiment
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

    command = [
        args.shard_command, 'run_gaussian_experiments.py',
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