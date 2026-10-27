"""Restartable result-file utilities shared by experiment runners."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RUN_LOG_NAME = 'run_log.jsonl'


def utc_timestamp():
    """Return the current UTC time as a sortable ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def append_run_log(output_dir, record):
    """Append one JSON line to the run log inside an output directory.

    Each runner invocation records when it started and finished, so
    completed result files can be attributed to a batch without relying
    on file modification times. Timing metadata only: result-file schemas
    never change, so this log is excluded from golden-smoke regenerations.
    """
    log_path = Path(output_dir) / RUN_LOG_NAME
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {'utc': utc_timestamp(), **record}
    with log_path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(stamped, default=str) + '\n')
    return log_path


def write_csv_atomically(data, output_path):
    """Write a CSV through a temporary file to avoid partial results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix('.tmp')
    data.to_csv(temporary_path, index=False)
    temporary_path.replace(output_path)


def combine_csv_files(result_files):
    """Read completed CSV files and concatenate them in sorted order."""
    result_files = sorted(Path(path) for path in result_files)
    if not result_files:
        return None
    return pd.concat(
        [pd.read_csv(path) for path in result_files],
        ignore_index=True,
    )
