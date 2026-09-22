"""Restartable result-file utilities shared by experiment runners."""

from pathlib import Path

import pandas as pd


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
