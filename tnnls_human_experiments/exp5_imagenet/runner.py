"""Experiment 5 runner (human pipeline port): stage orchestration.

Stages (each resumable independently; order matters only as data
dependencies):
  cache       fresh ResNet-18 forward over all 50k validation images -> npz
  verify      F1 demonstrated cache verification (>=1k stratified, atol 1e-4)
  splits      per-class 25/25 split table + dataset fingerprint + integrity
  clusters    PHATE-10d + k-means k=100 on discovery representations;
              evaluation routed by nearest discovery neighbor (std 512-d)
  fits        Linear CorEx [H|L] per discovery cluster, M=10, top-3 by TC
  ablations   q=0.80 supports; LC + 5 matched randoms + univariate; exact
              sparse-update ablation on the per-cluster cohort
  case_study  African-grey cluster + its 2 factors at q in {0.6,0.7,0.9}
  summarize   tidy CSVs in outputs/summaries (reads shards only)

Run report: written per stage; the F2 per-cluster size distribution is
recorded by `clusters`/`ablations` (into the report + a JSON artifact)
BEFORE any systematic table is computed.

All outputs live in this worktree's outputs/ tree (TNNLS_OUTPUTS_ROOT),
i.e. outputs/raw/imagenet, outputs/summaries, outputs/logs.
"""

from __future__ import annotations

import os
from pathlib import Path

# resolve outputs inside THIS worktree before common.config is imported
_HERE = Path(__file__).resolve().parent
os.environ.setdefault(
    "TNNLS_OUTPUTS_ROOT", str(_HERE.parent.parent / "outputs"))

import json
import time  # noqa: E402
import logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import config  # noqa: E402
from common.io import FailureLog, write_run_report  # noqa: E402
from common.runtime import setup_logging  # noqa: E402
from common.store import ResultStore  # noqa: E402

EXPERIMENT = "imagenet"
CACHE_PATH = config.RAW_DIR / "imagenet" / "resnet18_imagenet_val_cache.npz"
N_RANDOM_CONTROLS = 5
N_CLUSTERS = 100
N_FACTORS = 10
PRIMARY_Q = 0.80

DEFAULTS = {"n_clusters": N_CLUSTERS, "n_factors": N_FACTORS,
            "primary_q": PRIMARY_Q, "n_random_controls": N_RANDOM_CONTROLS,
            "verify_n_per_class": 2, "cache_batch_size": 256,
            "fits_max_workers": 16, "ablations_max_workers": 16}

STAGES = ("cache", "verify", "splits", "clusters", "fits", "ablations",
          "case_study", "summarize")


def stage_cache(store: ResultStore, failure_log: FailureLog, cfg: dict) -> dict:
    from data import build_cache

    if CACHE_PATH.exists():
        meta = json.loads(
            CACHE_PATH.with_suffix(".cache_meta.json").read_text())
        return {"cached": True, "meta": meta}
    t0 = time.time()
    meta = build_cache(CACHE_PATH, device="cuda:0",
                       batch_size=int(cfg["cache_batch_size"]))
    return {"cached": False, "runtime_s": time.time() - t0, "meta": meta}


def stage_verify(store: ResultStore, failure_log: FailureLog, cfg: dict) -> dict:
    from data import verify_cache

    n_per_class = int(cfg["verify_n_per_class"])
    cond = {"stage": "verify", "n_per_class": n_per_class}
    if store.is_done(cond, ("cache_verification",)):
        row = pd.read_parquet(
            store.shard_path(cond, "cache_verification")).iloc[0].to_dict()
        if not bool(row["verified"]):
            raise RuntimeError(f"cached verification record says FAILED: {row}")
        return {"cached": True, "report": row}
    report = verify_cache(CACHE_PATH, n_per_class=n_per_class, device="cuda:0")
    if not report["verified"]:
        raise RuntimeError(f"F1 demonstrated cache verification FAILED: {report}")
    # agreement statistics are written to the run log AND to artifacts (F1)
    logger = logging.getLogger(f"tnnls.{EXPERIMENT}")
    logger.info("F1 cache verification: %s", json.dumps(report))
    out_path = config.RAW_DIR / "imagenet" / "verify_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    store.write(cond, "cache_verification", [report])
    return {"cached": False, "report": report}


def stage_splits(store: ResultStore, failure_log: FailureLog) -> dict:
    from data import dataset_fingerprint, load_cache, per_class_integrity

    cond = {"stage": "splits", "fingerprint": _labels_sha()}
    if store.is_done(cond, ("split", "split_integrity")):
        return {"cached": True}
    cache = load_cache(CACHE_PATH)
    labels = cache["labels"]
    is_disc, is_eval = _split_arrays(labels)
    integrity = per_class_integrity(labels)
    rows = [{"image_index": int(i), "true_class": int(labels[i]),
             "split": "discovery" if is_disc[i] else "evaluation",
             "integrity_status": "ok", "hf_row_order": int(i)}
            for i in range(len(labels))]
    store.write(cond, "split", rows)
    store.write(cond, "split_integrity",
                [{**dataset_fingerprint(labels), **integrity,
                  "n_discovery": int(is_disc.sum()),
                  "n_evaluation": int(is_eval.sum())}])
    return {"n_rows": len(rows)}


def _labels_sha() -> str:
    import hashlib

    with np.load(CACHE_PATH) as z:
        labels = z["labels"]
    return hashlib.sha256(labels.astype(np.int64).tobytes()).hexdigest()[:16]


def _split_arrays(labels: np.ndarray):
    from data import imagenet_split

    return imagenet_split(labels)


def stage_clusters(store: ResultStore, failure_log: FailureLog, cfg: dict) -> dict:
    import clustering

    result = clustering.run_clustering(store, cfg, CACHE_PATH)
    dist = result.cluster_size_report
    write_size_distribution_artifacts(dist)
    return result.info


def write_size_distribution_artifacts(dist: dict) -> Path:
    """F2: per-cluster size distribution in the run report + as an artifact
    BEFORE any systematic table is computed."""
    path = config.RAW_DIR / "imagenet" / "cluster_size_distribution.json"
    path.write_text(json.dumps(dist, indent=2))
    write_run_report(EXPERIMENT, _run_id(), 0, 0, {
        "stage": "cluster_size_distribution (F2, written before any "
                 "systematic table)",
        "distribution": json.dumps(dist)})
    return path


def size_report_from_store(store: ResultStore) -> dict:
    import clustering

    rows = pd.read_parquet(store.shard_path(
        {"stage": "clusters", "k": N_CLUSTERS, "phate_seed": 7000,
         "kmeans_seed": 7001}, "cluster_sizes")).to_dict("records")
    return clustering.size_distribution_report(rows)


def stage_fits(store: ResultStore, failure_log: FailureLog, cfg: dict) -> dict:
    import fits

    return fits.run_fits(store, cfg, CACHE_PATH, failure_log,
                         max_workers=int(cfg["fits_max_workers"]))


def stage_ablations(store: ResultStore, failure_log: FailureLog, cfg: dict) -> dict:
    import ablations
    import clustering

    cohorts, low_sample, meta = clustering.build_eval_cohorts(store, cfg, CACHE_PATH)
    dist = size_report_from_store(store)
    write_size_distribution_artifacts(dist)
    write_run_report(EXPERIMENT, _run_id(), 0, 0, {
        "stage": "low_sample_clusters (<50 evaluation images)",
        "n_low_sample": len(low_sample), "clusters": sorted(low_sample)})
    ablations.build_final_layer_cache()
    out = ablations.run_ablations(store, cfg, CACHE_PATH, failure_log,
                                  max_workers=int(cfg["ablations_max_workers"]))
    out["n_low_sample_clusters"] = len(low_sample)
    out["low_sample_clusters"] = sorted(low_sample)
    return out


def stage_case_study(store: ResultStore, failure_log: FailureLog, cfg: dict) -> dict:
    import case_study

    out = case_study.run_case_study(store, cfg, CACHE_PATH, failure_log,
                                    max_workers=int(cfg["fits_max_workers"]))
    rows = case_study.secondary_cluster_report(store, CACHE_PATH)
    store.write({"stage": "case_study", "kind": "secondary_clusters"},
                "case_study_secondary_clusters", rows)
    return {"selected_cluster": out["selected"],
            "n_secondary_rows": len(rows)}


def stage_summarize(store: ResultStore, cfg: dict) -> dict:
    import summarize

    return summarize.main()


def _run_id() -> str:
    import json as _json

    path = LOG_ID_PATH
    if path.exists():
        return _json.loads(path.read_text())["run_id"]
    rid = config.make_run_id(EXPERIMENT, DEFAULTS, 0)
    path.write_text(_json.dumps({"run_id": rid}))
    config.save_resolved_config(rid, DEFAULTS, extra={"experiment": EXPERIMENT,
                                                      "stages": list(STAGES)})
    return rid


LOG_ID_PATH = config.LOGS_DIR / "imagenet_run_id.json"


def main(stages, force: bool = False) -> dict:
    run_id = _run_id()
    setup_logging(EXPERIMENT, run_id)
    store = ResultStore(EXPERIMENT)
    failure_log = FailureLog(EXPERIMENT)
    out: dict = {"run_id": run_id}
    if "cache" in stages:
        out["cache"] = stage_cache(store, failure_log, DEFAULTS)
    if "verify" in stages:
        out["verify"] = stage_verify(store, failure_log, DEFAULTS)
    if "splits" in stages:
        out["splits"] = stage_splits(store, failure_log)
    if "clusters" in stages:
        out["clusters"] = stage_clusters(store, failure_log, DEFAULTS)
    if "fits" in stages:
        out["fits"] = stage_fits(store, failure_log, DEFAULTS)
    if "ablations" in stages:
        out["ablations"] = stage_ablations(store, failure_log, DEFAULTS)
    if "case_study" in stages:
        out["case_study"] = stage_case_study(store, failure_log, DEFAULTS)
    if "summarize" in stages:
        out["summarize"] = stage_summarize(store, DEFAULTS)
    failures = failure_log.read()
    n_failed = int(len(failures))
    n_ok = int(len(store.read_all("ablation_summary"))) + 1
    write_run_report(EXPERIMENT, run_id, n_ok, n_failed,
                     {"stages": ",".join(stages),
                      "stage_results": json.dumps(
                          {k: _jsonable(v) for k, v in out.items()},
                          default=str)})
    return out


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_jsonable(x) for x in v[:5]] + ["..."] if len(v) > 5 else v
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", type=str,
                    default="cache,verify,splits,clusters,fits,ablations,"
                            "case_study,summarize")
    args = ap.parse_args()
    stages = tuple(s for s in args.stages.split(",") if s)
    result = main(stages)
    print(json.dumps(result, indent=2, default=str))
