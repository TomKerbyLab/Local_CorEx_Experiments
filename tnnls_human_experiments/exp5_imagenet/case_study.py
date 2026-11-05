"""Experiment 5 stage 6: African-grey case study (spec §African-grey case
study; deterministic discovery-only selection, freeze F5).

Selection rule (discovery data only; evaluation intervention results are
never consulted):
  1. ImageNet class index for African grey = AFRICAN_GREY_HF_IDX (87,
     AFRICAN_GREY_WNID), verified against the saved class mapping at import.
  2. Among clusters with >= 100 discovery images, the cluster with the
     largest proportion of African-grey images; ties by larger African-grey
     count, then lower cluster id.
  3. Within the cluster, the two retained factors with the largest MI with
     the African-grey logit.
  4. Supports at q in {0.60, 0.70, 0.80, 0.90}; 0.80 cells come from the
     main ablation stage, the remaining thresholds are evaluated here with
     their matched random + univariate controls.
  5. Secondary-cluster readout: per-cluster mean TV over off-cluster
     evaluation images for the main support.

Cluster ids are run-specific; the run records which id this run produced.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import config
from common import metrics, seeds
from common.io import FailureLog
from common.runtime import run_parallel
from common.store import ResultStore

from ablations import run_ablation_cell

AFRICAN_GREY_IDX = 87
AFRICAN_GREY_WNID = "n01817953"
CASE_THRESHOLDS = (0.60, 0.70, 0.90)  # 0.80 comes from the main stage
MIN_DISC_FOR_CASE = 100
N_SELECTED_FACTORS = 2


def select_african_grey_cluster(store: ResultStore, labels: np.ndarray) -> dict:
    """Deterministic rule on discovery data only. Returns the selection
    record; the caller writes it as a shard before any case-study fit."""
    from clustering import load_cluster_assignments
    from data import imagenet_split

    is_disc, _ = imagenet_split(labels)
    disc_idx = np.flatnonzero(is_disc)
    clusters = load_cluster_assignments(store, {"n_clusters": 100})
    rec = african_grey_rule(clusters[disc_idx], labels[disc_idx])
    return rec


def african_grey_rule(disc_cluster: np.ndarray,
                      disc_labels: np.ndarray) -> dict:
    """Pure deterministic selection rule (discovery data only; unit-testable
    without a store, dataset, or evaluation results).

    disc_cluster: per-discovery-image cluster assignment; disc_labels:
    matching true classes. Among clusters with >= MIN_DISC_FOR_CASE
    discovery images, choose the largest proportion of African-grey images;
    ties by larger count, then lower cluster id (F5).
    """
    best = None
    for k in np.unique(disc_cluster):
        k = int(k)
        members = disc_cluster == k
        n = int(members.sum())
        if n < MIN_DISC_FOR_CASE:
            continue
        n_grey = int((disc_labels[members] == AFRICAN_GREY_IDX).sum())
        prop = n_grey / n
        key = (prop, n_grey, -k)
        if best is None or key > best[0]:
            best = (key, k, n, n_grey, prop)
    if best is None:
        raise RuntimeError(
            "African-grey selection failed: no cluster with >= "
            f"{MIN_DISC_FOR_CASE} discovery images")
    return {
        "selected_cluster": int(best[1]),
        "n_discovery_images": int(best[2]),
        "n_african_grey_discovery": int(best[3]),
        "proportion": float(best[4]),
        "rule": f">= {MIN_DISC_FOR_CASE} discovery images, max proportion "
                f"of class {AFRICAN_GREY_IDX} ({AFRICAN_GREY_WNID}), ties by "
                f"count then lower id",
        "cluster_ids_run_specific": True,
        "uses_evaluation_data": False,
    }


def load_fit_mis(cluster: int) -> dict:
    path = config.RAW_DIR / "imagenet" / "corex_fits" / f"cluster_{cluster:03d}.npz"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def select_african_grey_factors(cluster: int) -> list[dict]:
    """Two retained factors with the largest MI with the African-grey logit
    (L[:, 87]), from the stored fit MI matrix (discovery data only)."""
    fit = load_fit_mis(cluster)
    mis = fit["mis_clipped"]
    n_h = int(fit["n_h"])
    order = fit["order"].tolist()
    grey_col = mis[:, n_h + 87]
    cand = sorted(order, key=lambda j: (-grey_col[j], j))
    rows = []
    for rank, j in enumerate(cand[:N_SELECTED_FACTORS]):
        rows.append({
            "cluster": cluster, "factor_index": int(j),
            "selection_rank": rank,
            "african_grey_logit_idx": 87,
            "factor_to_grey_logit_mi_nats": float(mis[j, n_h + 87]),
            "factor_to_grey_logit_mi_bits":
                float(mis[j, n_h + 87] / np.log(2.0)),
            "unit_mi_rank_within_cluster":
                int(np.where(np.asarray(order) == j)[0][0]),
        })
    return rows


def case_support_rows(cluster: int, factor_index: int) -> list[dict]:
    """Supports for one case-study factor at all four case thresholds
    (0.80's rows come from the main stage and are referenced only)."""
    fit = load_fit_mis(cluster)
    mis = fit["mis_clipped"]
    n_h = int(fit["n_h"])
    valid_h = fit["valid_h"]
    rows = []
    thresholds = sorted({0.80, *CASE_THRESHOLDS})
    for q in thresholds:
        sup = metrics.cumulative_mi_support(mis[factor_index, :n_h], q, valid_h)
        for r, (unit, score) in enumerate(zip(sup.indices, sup.scores)):
            rows.append({
                "cluster": cluster, "factor_index": factor_index,
                "threshold": q, "mi_rank": r, "unit": int(unit),
                "mi_score_nats": float(score),
                "mi_score_bits": float(score / np.log(2.0)),
                "support_size": sup.size, "support_density": sup.density,
                "achieved_fraction": sup.achieved_fraction,
                "degenerate": sup.degenerate, "stage": "case_study"})
    return rows


def run_case_study(store: ResultStore, cfg: dict, cache_path: Path,
                   failure_log: FailureLog, max_workers: int | None = 8) -> dict:
    """Selection + supports + intervention cells for the non-primary
    thresholds (0.60/0.70/0.90); the 0.80 cells already exist in the main
    ablation stage."""
    cache = _read_cache(cache_path)
    labels = cache["labels"]

    sel_cond = {"stage": "case_study_selection"}
    if store.is_done(sel_cond, ("case_study_selection",)):
        sel_row = pd.read_parquet(store.shard_path(sel_cond, "case_study_selection")
                                  ).iloc[0].to_dict()
    else:
        sel = select_african_grey_cluster(store, labels)
        sel_row = dict(sel)
        store.write(sel_cond, "case_study_selection", [sel_row])
    cluster = int(sel_row["selected_cluster"])
    factors = select_african_grey_factors(cluster)

    sup_cond = {"stage": "case_study_supports", "cluster": cluster}
    if not store.is_done(sup_cond, ("case_study_supports",)):
        rows = []
        for f in factors:
            rows.extend(case_support_rows(cluster, f["factor_index"]))
        store.write_multiple(sup_cond, {"case_study_supports": rows,
                                        "case_study_factors": factors})

    tasks = []
    for f in factors:
        for q in CASE_THRESHOLDS:
            sup_rows = case_support_rows(cluster, f["factor_index"])
            support = np.sort(
                pd.DataFrame([r for r in sup_rows if r["threshold"] == q]
                             ).sort_values("mi_rank")["unit"].to_numpy()).tolist()
            tasks.append({"stage": "ablation", "cluster": cluster,
                          "factor_index": f["factor_index"], "threshold": q,
                          "support_override": support,
                          "n_random_controls": cfg["n_random_controls"],
                          "cache_path": str(cache_path),
                          "wnid_by_index": wnids_by_index()})
    t0 = time.time()
    run_parallel(_case_cell, tasks, max_workers, failure_log, "imagenet",
                 stage_name="case_study_ablation")
    return {"selected": sel_row, "runtime_s": time.time() - t0,
            "factors": factors}


def _read_cache(cache_path: Path) -> dict:
    with np.load(cache_path) as z:
        return {"features": z["features"], "logits": z["logits"],
                "labels": z["labels"]}


def wnids_by_index() -> list[dict]:
    from data import class_mapping
    return class_mapping()


def _case_cell(cond: dict) -> dict:
    return run_ablation_cell(cond)


def secondary_cluster_report(store: ResultStore, cache_path: Path) -> list[dict]:
    """Per-cluster mean TV over OFF-cluster evaluation images for the two
    case-study factors at the primary threshold; the most-affected
    secondary cluster (max mean TV) is reported per factor."""
    cache = _read_cache(cache_path)
    sel = pd.read_parquet(store.shard_path(
        {"stage": "case_study_selection"}, "case_study_selection")).iloc[0]
    cluster = int(sel["selected_cluster"])
    from clustering import load_cluster_assignments
    clusters = load_cluster_assignments(store, {
        "n_clusters": 100, "phate_seed": 7000, "kmeans_seed": 7001})

    rows = []
    factors = select_african_grey_factors(cluster)
    with np.load(config.RAW_DIR / "imagenet" / "eval_cohorts.npz") as z:
        eval_imgs = z[f"cluster_{cluster:03d}_images"]
        is_source = z[f"cluster_{cluster:03d}_source"].astype(bool)
    for f in factors:
        fi = f["factor_index"]
        q_main = int(round(0.80 * 100))
        with np.load(config.RAW_DIR / "imagenet" / "classwise_shifts" /
                     f"cluster_{cluster:03d}_factor_{fi:02d}_q{q_main}.npz") as z:
            tv = z["local_corex_0_tv"]
        off = ~is_source
        off_imgs = eval_imgs[off]
        tv_off = tv[off]
        off_cluster_ids = clusters[off_imgs]
        df = pd.DataFrame({"cluster": off_cluster_ids, "tv": tv_off})
        per = df.groupby("cluster")["tv"].agg(["mean", "count"])\
            .sort_values("mean", ascending=False)
        for rank, (k, agg) in enumerate(per.head(10).iterrows()):
            rows.append({"case_factor": fi, "cluster": cluster,
                         "off_cluster": int(k), "mean_tv_off_cluster":
                             float(agg["mean"]), "n_images": int(agg["count"]),
                         "rank": rank, "is_most_affected_secondary":
                             rank == 0})
    return rows
