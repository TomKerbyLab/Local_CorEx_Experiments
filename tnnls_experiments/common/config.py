"""Run configuration, resolved-config persistence, and run identities.

Every run gets:
  - a run id  "<experiment>.<short-hash>.<seed>"  (never collides across configs);
  - a resolved config dict saved to outputs/configs/<run_id>.json;
  - a metadata record with git hash, package versions, hardware, timing.

Completed runs are never overwritten: a run with existing outputs is skipped
unless force=True.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = Path(os.environ.get("TNNLS_OUTPUTS_ROOT", REPO_ROOT / "outputs"))

CONFIGS_DIR = OUTPUTS_ROOT / "configs"
RAW_DIR = OUTPUTS_ROOT / "raw"
SUMMARIES_DIR = OUTPUTS_ROOT / "summaries"
FIGURES_DIR = OUTPUTS_ROOT / "figures"
LOGS_DIR = OUTPUTS_ROOT / "logs"
CHECKPOINTS_DIR = OUTPUTS_ROOT / "checkpoints"

for _d in (CONFIGS_DIR, RAW_DIR, SUMMARIES_DIR, FIGURES_DIR, LOGS_DIR, CHECKPOINTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load a YAML config; shallow-merge overrides (values may be dotted paths)."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        for dotted, value in overrides.items():
            set_dotted(cfg, dotted, value)
    return cfg


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot set {dotted!r}: {k!r} is not a section")
    node[keys[-1]] = value


def config_hash(cfg: dict) -> str:
    """Stable short hash of the resolved config (json-sorted)."""
    blob = json.dumps(_jsonable(cfg), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:10]


def make_run_id(experiment: str, cfg: dict, root_seed: int) -> str:
    return f"{experiment}.{config_hash(cfg)}.{int(root_seed)}"


def save_resolved_config(run_id: str, cfg: dict, extra: dict | None = None) -> Path:
    path = CONFIGS_DIR / f"{run_id}.json"
    payload = {"run_id": run_id, "config": _jsonable(cfg)}
    if extra:
        payload.update(_jsonable(extra))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    return path


def run_marker(run_id: str) -> Path:
    return LOGS_DIR / f"{run_id}.done.json"


def is_complete(run_id: str) -> bool:
    m = run_marker(run_id)
    return m.exists()


def mark_complete(run_id: str, record: dict) -> None:
    m = run_marker(run_id)
    if m.exists():  # never overwrite a completed run
        raise FileExistsError(f"Run {run_id} is already marked complete")
    with open(m, "w") as f:
        json.dump(_jsonable(record), f, indent=2, default=str)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def package_versions(packages: list[str]) -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for pkg in packages:
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "NOT_INSTALLED"
    return out


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    return info


def run_metadata(cfg: dict, extra: dict | None = None) -> dict:
    """Standard metadata block recorded with every run."""
    now = datetime.datetime.now(datetime.timezone.utc)
    meta = {
        "started_utc": now.isoformat(),
        "git_commit": git_commit(),
        "packages": package_versions(
            ["local-corex", "numpy", "scipy", "scikit-learn", "phate", "torch", "pyarrow", "pandas", "matplotlib"]
        ),
        "hardware": hardware_info(),
    }
    if extra:
        meta.update(extra)
    return meta


class RunTimer:
    """Simple wall-clock timer for run records."""

    def __init__(self):
        self.start = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start


def _jsonable(obj: Any) -> Any:
    """Recursively convert to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(dataclasses.asdict(obj))
    return str(obj)