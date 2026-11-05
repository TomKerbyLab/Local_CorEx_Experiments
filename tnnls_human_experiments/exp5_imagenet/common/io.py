"""Tidy result persistence: incremental CSV/Parquet writers, failure records."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import LOGS_DIR, RAW_DIR

_LOCK = threading.Lock()
_FLUSH_EVERY = 200  # rows before an incremental flush


class TidyWriter:
    """Incrementally appends tidy rows to CSV (compact) or Parquet (large).

    Usage:
        w = TidyWriter(RAW_DIR / "gaussian_runs.csv")
        w.writerow({...})
        w.close()
    """

    def __init__(self, path: str | Path, flush_every: int = _FLUSH_EVERY):
        self.path = Path(path)
        self.flush_every = flush_every
        self._buf: list[dict] = []
        self._header_written = self.path.exists() and self.path.stat().st_size > 0
        self._n = 0
        self._lock = threading.Lock()

    def writerows(self, rows: Iterable[dict]) -> None:
        for r in rows:
            self.writerow(r)

    def writerow(self, row: dict) -> None:
        with self._lock:
            self._buf.append(row)
            self._n += 1
            if len(self._buf) >= self.flush_every:
                self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        df = pd.DataFrame(self._buf)
        self._buf.clear()
        header = not self._header_written
        df.to_csv(self.path, mode="a", header=header, index=False)
        self._header_written = True

    def close(self) -> None:
        with self._lock:
            self._flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FailureLog:
    """Appends failure records to outputs/logs/<experiment>_failures.jsonl."""

    def __init__(self, experiment: str):
        self.path = LOGS_DIR / f"{experiment}_failures.jsonl"
        self._lock = threading.Lock()

    def record(
        self,
        condition: dict[str, Any],
        seed: int | None,
        stage: str,
        exc: BaseException | None = None,
        message: str | None = None,
        retried: bool = False,
        retry_config: dict | None = None,
    ) -> None:
        row = {
            "condition": condition,
            "seed": seed,
            "stage": stage,
            "exception_type": type(exc).__name__ if exc else None,
            "message": message or (str(exc) if exc else None),
            "retried": retried,
            "retry_config": retry_config,
        }
        line = json.dumps(row, default=str)
        with self._lock:
            with open(self.path, "a") as f:
                f.write(line + "\n")

    def read(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        with open(self.path) as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        if not rows:
            return pd.DataFrame()
        return pd.json_normalize(rows)


def write_run_report(
    experiment: str,
    run_id: str,
    success_conditions: int,
    failed_conditions: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Short human-readable run report next to the logs."""
    path = LOGS_DIR / f"{experiment}_run_report.txt"
    lines = [
        f"Run: {run_id}",
        f"Successful conditions: {success_conditions}",
        f"Failed conditions: {failed_conditions}",
    ]
    if extra:
        lines += [f"{k}: {v}" for k, v in extra.items()]
    with open(path, "a") as f:
        f.write("\n".join(lines) + "\n")
    return path


def save_npz(path: str | Path, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def dataframe_checksum(df: pd.DataFrame) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()[:16]