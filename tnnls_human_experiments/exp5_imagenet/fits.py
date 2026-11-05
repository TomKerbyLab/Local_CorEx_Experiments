"""Experiment 5 stage 4: Linear CorEx fits per discovery cluster on
[H_k | L_k] (1512-d), M=10 factors, top-3 retained by explained TC.

Standardization uses only the discovery observations in the cluster (per
spec step 1). Factor->unit and factor->logit MIs are stored separately; MI
units are recorded in nats (Linear CorEx's native scale) and in bits
(= nats / ln 2).

Supports: q = 0.80 cumulative-MI on the 512 pooled units (primary
threshold). Supports for the case-study thresholds {0.60,0.70,0.90} are
computed later from the stored MI vectors (case_study.py), never fitted
from evaluation data.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from common import config, corex, metrics, seeds
from common.io import FailureLog
from common.runtime import run_parallel
from common.store import ResultStore

N_FACTORS = 10
N_RETAINED = 3
PRIMARY_Q = 0.80
NATS_TO_BITS = 1.0 / np.log(2.0)
MIN_DISCOVERY_OBS = 10

# tiny-cluster rule: clusters with too few discovery observations are
# recorded as failure rows and reported in coverage, never silently dropped


def _read_cache(cache_path: Path) -> dict:
    with np.load(cache_path) as z:
        return {"features": z["features"], "logits": z["logits"],
                "labels": z["labels"]}


def fit_one_cluster(k: int, cfg: dict, cache_path: Path) -> dict:
    """Fit CorEx for one discovery cluster; write its shards. Runs in a worker
    process or inline; failures propagate to the FailureLog."""
    from common.store import ResultStore as _Store

    from clustering import load_cluster_assignments
    from data import imagenet_split

    store = _Store("imagenet")
    cond = {"stage": "fit", "cluster": k}
    if store.is_done(cond, ("factors", "supports")):
        return {"cluster": k, "cached": True}
    cache = _read_cache(cache_path)
    features, logits, labels = cache["features"], cache["logits"], cache["labels"]
    is_disc, _ = imagenet_split(labels)
    cluster_of_image = load_cluster_assignments(store, cfg)
    disc_idx = np.flatnonzero(is_disc)
    members = disc_idx[cluster_of_image[disc_idx] == k]
    if len(members) < MIN_DISCOVERY_OBS:
        store.write(cond, "factors", [{"cluster": k, "fit_failed": True,
                                       "failure_reason": "n_discovery_obs_below_10"}])
        store.write(cond, "supports", [])
        return {"cluster": k, "status": "skipped_tiny_cluster",
                "n_disc_obs": int(len(members))}
    H = features[members].astype(np.float64)
    L = logits[members].astype(np.float64)
    H_std, valid_h, excl_h = corex.standardize_columns(H)
    L_std, valid_l, excl_l = corex.standardize_columns(L)
    s_fit = seeds.derive_seed(k, "corex")
    V = np.hstack([H_std, L_std])
    fit = corex.fit_linear_corex(V, cfg["n_factors"], seed=s_fit)
    mis_raw = fit.mis  # nats
    mis = fit.association_scores()  # nonneg clip (documented transform)
    n_h = H.shape[1]
    order = np.argsort(-fit.tcs)[: cfg.get("n_retained", N_RETAINED)]

    factor_rows, support_rows = [], []
    for rank, j in enumerate(order):
        lmis = mis[j, n_h:]
        dominant = int(np.argmax(lmis))
        factor_rows.append({
            "cluster": k, "factor_index": int(j), "factor_rank": rank,
            "factor_tc_nats": float(fit.tcs[j]),
            "factor_tc_bits": float(fit.tcs[j] * NATS_TO_BITS),
            "unit_mi_total_nats": float(mis[j, :n_h].sum()),
            "unit_mi_total_bits": float(mis[j, :n_h].sum() * NATS_TO_BITS),
            "logit_mi_total_nats": float(lmis.sum()),
            "logit_mi_total_bits": float(lmis.sum() * NATS_TO_BITS),
            "dominant_logit": dominant,
            "dominant_logit_mi_nats": float(lmis[dominant]),
            "dominant_logit_mi_bits": float(lmis[dominant] * NATS_TO_BITS),
            "logit_mi_gap": float(np.sort(lmis)[-1] - np.sort(lmis)[-2]),
            "n_valid_units": int(valid_h.sum()),
            "n_invalid_unit_cols": int(excl_h.sum()),
            "n_invalid_logit_cols": int(excl_l.sum()),
            "n_negative_raw_mi": fit.n_negative_mi,
            "min_raw_mi_nats": fit.min_raw_mi,
            "corex_converged": fit.converged,
            "corex_seed": s_fit, "corex_runtime_s": fit.runtime_s,
            "n_discovery_obs": int(len(members)),
            "fit_failed": False,
        })
        umis = mis[j, :n_h]
        sup = metrics.cumulative_mi_support(umis, cfg["primary_q"], valid_h)
        for r, (unit, score) in enumerate(zip(sup.indices, sup.scores)):
            support_rows.append({
                "cluster": k, "factor_index": int(j), "threshold":
                cfg["primary_q"], "mi_rank": r, "unit": int(unit),
                "mi_score_nats": float(score),
                "mi_score_bits": float(score * NATS_TO_BITS),
                "support_size": sup.size, "support_density": sup.density,
                "achieved_fraction": sup.achieved_fraction,
                "degenerate": sup.degenerate,
            })
    store.write_multiple(cond, {"factors": factor_rows, "supports": support_rows})
    fit_dir = config.RAW_DIR / "imagenet" / "corex_fits"
    fit_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fit_dir / f"cluster_{k:03d}.npz",
                        mis_clipped=mis, mis_raw=mis_raw, tcs=fit.tcs,
                        order=np.asarray(order), n_h=n_h,
                        valid_h=valid_h, valid_l=valid_l,
                        members=members.astype(np.int64))
    return {"cluster": k, "status": "fit", "n_disc_obs": int(len(members)),
            "tcs": fit.tcs.tolist()}


def run_fits(store: ResultStore, cfg: dict, cache_path: Path,
             failure_log: FailureLog, max_workers: int | None = 16) -> dict:
    tasks = [{"stage": "fit", "cluster": k} for k in range(cfg["n_clusters"])]
    t0 = time.time()
    run_parallel(_fit_condition, tasks, max_workers, failure_log, "imagenet",
                 stage_name="corex_fit")
    return {"runtime_s": time.time() - t0, "n_clusters": cfg["n_clusters"]}


def _fit_condition(cond: dict) -> dict:
    """Worker entry: one condition = one cluster fit."""
    from clustering import KMEANS_SEED, N_CLUSTERS, PHATE_SEED

    cfg = {"n_factors": N_FACTORS, "primary_q": PRIMARY_Q,
           "n_clusters": N_CLUSTERS, "phate_seed": PHATE_SEED,
           "kmeans_seed": KMEANS_SEED}
    cache_path = _default_cache_path()
    return fit_one_cluster(cond["cluster"], cfg, cache_path)


def _default_cache_path() -> Path:
    return config.RAW_DIR / "imagenet" / "resnet18_imagenet_val_cache.npz"
