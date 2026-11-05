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

    def write_multiple(
        self, condition: dict, kinds_rows: dict[str, list[dict]]
    ) -> list[Path]:
        """Write several shards of one condition as a unit (e.g. runs+matches).

        A crash between two single-kind writes used to wedge the condition:
        the runs shard existed, so the next rerun raised FileExistsError on
        every attempt. A condition is only complete when *every* kind exists
        (is_done semantics), so a partial shard set is stale, not completed,
        and is safely replaced here. Callers must still check is_done()
        first; a fully-present shard set raises FileExistsError rather than
        being overwritten.
        """
        paths = {kind: self.shard_path(condition, kind) for kind in kinds_rows}
        if all(p.exists() for p in paths.values()):
            raise FileExistsError(
                "completed condition (all shards present): "
                + ", ".join(sorted(p.name for p in paths.values()))
            )
        for path in paths.values():
            path.unlink(missing_ok=True)
        return [self.write(condition, kind, rows) for kind, rows in kinds_rows.items()]

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