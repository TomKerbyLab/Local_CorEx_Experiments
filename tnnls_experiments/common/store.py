"""Resumable per-condition result storage.

Each condition writes its own small Parquet shard, named by a hash of the
condition keys (never overwritten; a completed shard means the condition is
done). Summarizers glob and concatenate shards. A crash mid-run loses only
the incomplete shard, and rerunning skips completed conditions — "never
overwrite a completed run" from the README.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .config import RAW_DIR


def condition_hash(condition: dict[str, Any]) -> str:
    blob = json.dumps(condition, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


class ResultStore:
    """Shard-based result store for one experiment."""

    def __init__(self, experiment: str):
        self.dir = RAW_DIR / experiment
        self.dir.mkdir(parents=True, exist_ok=True)

    def shard_path(self, condition: dict, kind: str) -> Path:
        return self.dir / f"{condition_hash(condition)}.{kind}.parquet"

    def is_done(self, condition: dict, kinds: tuple[str, ...]) -> bool:
        return all(self.shard_path(condition, k).exists() for k in kinds)

    def write(self, condition: dict, kind: str, rows: list[dict]) -> Path:
        """Atomically write one shard (temp file + rename).

        Raises FileExistsError if the shard already exists: completed
        conditions are never overwritten (README convention). Callers check
        is_done() first; the double guard protects against races.
        """
        path = self.shard_path(condition, kind)
        if path.exists():
            raise FileExistsError(f"shard already exists: {path}")
        tmp = path.with_suffix(".parquet.tmp")
        df = pd.DataFrame(rows)
        df.to_parquet(tmp, index=False)
        try:
            # link+unlink instead of rename so a concurrent duplicate write
            # fails loudly rather than silently replacing
            os.link(tmp, path)
        except FileExistsError:
            tmp.unlink(missing_ok=True)
            raise FileExistsError(f"shard already exists: {path}")
        tmp.unlink(missing_ok=True)
        return path

    def read_all(self, kind: str) -> pd.DataFrame:
        """Concatenate all shards of one kind (summarizer entry point)."""
        frames = []
        for p in sorted(self.dir.glob(f"*.{kind}.parquet")):
            frames.append(pd.read_parquet(p))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def completed_conditions(self) -> list[str]:
        return [p.name.split(".")[0] for p in sorted(self.dir.glob("*.parquet"))]

    def count(self, kind: str) -> int:
        return len(list(self.dir.glob(f"*.{kind}.parquet")))