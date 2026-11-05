"""Parallel execution helpers for grid runs.

Conditions are pure functions of (condition dict, seed); they are executed
over a process pool with derived seeds. Failures are caught and recorded —
a failed condition is data, not a crash of the run.
"""

from __future__ import annotations

import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable

from .io import FailureLog


def setup_logging(experiment: str, run_id: str, level: int = logging.INFO):
    """Log to outputs/logs/<experiment>_<run_id>.log and console."""
    from .config import LOGS_DIR

    path = LOGS_DIR / f"{experiment}_{run_id}.log"
    handlers = [logging.FileHandler(path), logging.StreamHandler()]
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(f"tnnls.{experiment}")


def run_parallel(
    fn: Callable[[dict], Any],
    conditions: Iterable[dict],
    max_workers: int | None,
    failure_log: FailureLog,
    experiment: str,
    stage_name: str = "condition",
    progress_every: int = 20,
) -> list[Any]:
    """Execute fn(condition) over a process pool; collect results and failures.

    fn must be picklable (e.g. a module-level function or functools.partial
    thereof). Each condition dict is passed as a single argument; failures are
    recorded with all condition keys.
    """
    conditions = list(conditions)
    results: list[Any] = []
    logger = logging.getLogger(f"tnnls.{experiment}")
    if max_workers in (None, 0, 1):
        for i, cond in enumerate(conditions):
            try:
                results.append(fn(cond))
            except KeyboardInterrupt:
                raise
            except Exception as e:
                failure_log.record(
                    condition=cond,
                    seed=cond.get("seed") or cond.get("root_seed"),
                    stage=stage_name,
                    exc=e,
                    message=traceback.format_exc(limit=3),
                )
            if (i + 1) % progress_every == 0:
                logger.info("%s: %d/%d done", stage_name, i + 1, len(conditions))
        return results

    workers = max_workers or min(32, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, cond): cond for cond in conditions}
        for i, fut in enumerate(as_completed(futures)):
            cond = futures[fut]
            try:
                results.append(fut.result())
            except KeyboardInterrupt:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as e:
                failure_log.record(
                    condition=cond,
                    seed=cond.get("seed") or cond.get("root_seed"),
                    stage=stage_name,
                    exc=e,
                    message=traceback.format_exc(limit=3),
                )
            if (i + 1) % progress_every == 0:
                logger.info("%s: %d/%d done", stage_name, i + 1, len(conditions))
    return results


def deterministic_retry(fn, attempts: int = 1, rule: str | None = None):
    """One deterministic numerical-stability retry (rule must be pre-specified).

    Raises the last exception if all attempts fail.
    """
    last = None
    for _ in range(max(1, attempts)):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
    raise last