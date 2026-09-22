"""Experiment 1 runner: Gaussian grouped-covariance recovery.

Core grid: n in {50,500,5000}, alpha in {0,0.5,1}, regimes
{mostly_shared, disjoint}, p_n = 25 nuisance, 16 paired replicates
(root seeds 1000-1015). Every method sees the same generated dataset
within a replicate.

Methods:
  global      one LinearCorEx on all observations
  oracle      separate LinearCorEx per true region
  phate       PHATE-10d + k-means (k=2, n_init=50, seed derived per replicate)
  random      10 balanced random partitions per generated dataset
  raw_kmeans  k-means on standardized observed variables
  pca         PCA-10 + k-means (supplementary)

For k != 2 the spec forbids forcing a one-to-one label map: ARI/purity are
reported directly and region factors are scored from the estimated cluster
with the largest overlap with each true region; if the same cluster is
selected for both regions, that fact is recorded.

Output shards (per generated dataset replicate): "runs" and "matches".
"""

from __future__ import annotations

import functools
import json
import logging
import time

import numpy as np

from ..common import config, corex, metrics, partition, seeds
from ..common.io import FailureLog
from ..common.runtime import run_parallel, setup_logging
from ..common.store import ResultStore
from . import data as exp1_data

EXPERIMENT = "gaussian"
logger = logging.getLogger(f"tnnls.{EXPERIMENT}")

DEFAULTS = {
    "n_factors": 5,
    "phate_dim": 10,
    "k_clusters": 2,
    "n_random_partitions": 10,
    "kmeans_n_init": 50,
    "n_workers": 0,
    "support_fraction": 0.80,
    "include_pca": True,
}


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def core_grid() -> list[dict]:
    return [
        {"n_per_region": n, "alpha": alpha, "regime": regime, "n_nuisance": 25,
         "root_seed": seed}
        for n in (50, 500, 5000)
        for alpha in (0.0, 0.5, 1.0)
        for regime in exp1_data.REGIMES
        for seed in range(1000, 1016)
    ]


def nuisance_grid() -> list[dict]:
    return [
        {"n_per_region": 500, "alpha": 0.5, "regime": regime, "n_nuisance": p_n,
         "root_seed": seed}
        for regime in exp1_data.REGIMES
        for p_n in (0, 25, 100, 475)
        for seed in range(1000, 1016)
    ]


def supplementary_grid() -> list[dict]:
    """One sensitivity dimension at a time around the core condition
    (n=500, alpha=0.5, p_n=25, d=10, k=2, M=5)."""
    base = {"n_per_region": 500, "alpha": 0.5, "regime": "mostly_shared",
            "n_nuisance": 25}
    conds: list[dict] = []
    for alpha in (0.25, 0.75):
        for regime in exp1_data.REGIMES:
            for seed in range(1000, 1016):
                conds.append({**base, "alpha": alpha, "regime": regime, "root_seed": seed})
    for phate_dim in (2, 5):
        for seed in range(1000, 1016):
            conds.append({**base, "phate_dim": phate_dim, "root_seed": seed})
    for k in (1, 4, 8):
        for seed in range(1000, 1016):
            conds.append({**base, "k_clusters": k, "root_seed": seed})
    for M in (3, 8, 10):
        for seed in range(1000, 1016):
            conds.append({**base, "n_factors": M, "root_seed": seed})
    return conds


def smoke_grid() -> list[dict]:
    return [
        {"n_per_region": 200, "alpha": 0.5, "regime": "mostly_shared",
         "n_nuisance": 25, "root_seed": 1000},
        {"n_per_region": 200, "alpha": 0.5, "regime": "disjoint",
         "n_nuisance": 25, "root_seed": 1001},
        {"n_per_region": 200, "alpha": 1.0, "regime": "disjoint",
         "n_nuisance": 0, "root_seed": 1000},
    ]


GRIDS = {
    "core": core_grid,
    "nuisance": nuisance_grid,
    "supplementary": supplementary_grid,
    "smoke": smoke_grid,
}


# ---------------------------------------------------------------------------
# Per-replicate execution
# ---------------------------------------------------------------------------

def condition_keys(cond: dict) -> dict:
    keys = ("n_per_region", "alpha", "regime", "n_nuisance", "root_seed",
            "phate_dim", "k_clusters", "n_factors")
    return {k: cond[k] for k in keys if k in cond}


def run_replicate(cond: dict, cfg: dict, store: ResultStore) -> dict:
    """Run one generated-dataset replicate through all partition methods."""
    root = int(cond["root_seed"])
    s_data = seeds.derive_seed(root, "data")
    s_kmeans = seeds.derive_seed(root, "partition", "kmeans")
    s_corex_global = seeds.derive_seed(root, "corex", "global")
    s_corex_region = seeds.derive_seed(root, "corex", "region")

    ds = exp1_data.generate_dataset(
        n_per_region=int(cond["n_per_region"]),
        alpha=float(cond["alpha"]),
        regime=cond["regime"],
        n_nuisance=int(cond["n_nuisance"]),
        seed=s_data,
    )
    X, region = ds["X"], ds["region"]
    X_std, _, _ = corex.standardize_columns(X)
    p_total = X.shape[1]
    nuisance_mask = np.zeros(p_total, dtype=bool)
    nuisance_mask[ds["nuisance_idx"]] = True

    M = int(cfg["n_factors"])
    phate_dim = int(cfg["phate_dim"])
    k = int(cfg["k_clusters"])
    n_rand = int(cfg["n_random_partitions"])
    frac = float(cfg["support_fraction"])
    n_init = int(cfg["kmeans_n_init"])

    base_keys = {
        "n_per_region": cond["n_per_region"],
        "alpha": cond["alpha"],
        "regime": cond["regime"],
        "n_nuisance": cond["n_nuisance"],
        "root_seed": root,
        "rho": exp1_data.RHO,
        "n_factors": M,
        "phate_dim": phate_dim,
        "k_clusters": k,
    }

    run_rows: list[dict] = []
    match_rows: list[dict] = []

    # ---- global fit --------------------------------------------------------
    global_fit = corex.fit_linear_corex(X_std, M, seed=s_corex_global)
    run_rows.append(
        _run_row(base_keys, "global", 0, 0, global_fit, ari=1.0, purity=1.0,
                 aligned_accuracy=1.0, cluster_sizes=json.dumps({"0": len(region)}))
    )
    # global factor set is matched independently per true region
    for c in (0, 1):
        match_rows += _score_fit(
            base_keys, "global", c, 0, global_fit, ds["true_groups"],
            nuisance_mask, p_total, frac,
        )

    # ---- oracle local ------------------------------------------------------
    for c in (0, 1):
        Xc = X_std[region == c]
        fit = corex.fit_linear_corex(Xc, M, seed=s_corex_region + c)
        run_rows.append(
            _run_row(base_keys, "oracle", c, 0, fit, ari=1.0, purity=1.0,
                     aligned_accuracy=1.0,
                     cluster_sizes=json.dumps({str(c): len(Xc)}))
        )
        match_rows += _score_fit(
            base_keys, "oracle", c, 0, fit, ds["true_groups"],
            nuisance_mask, p_total, frac,
        )

    # ---- estimated partitions ---------------------------------------------
    est: list[tuple[str, np.ndarray]] = []
    labels, _ = partition.phate_kmeans_partition(
        X_std, k, seed=s_kmeans, phate_dim=phate_dim, n_init=n_init,
        n_pca=min(50, p_total - 1),
    )
    est.append(("phate", labels))
    labels, _ = partition.raw_kmeans_partition(
        X_std, k, seed=s_kmeans + 1, n_init=n_init
    )
    est.append(("raw_kmeans", labels))
    if cfg.get("include_pca", True):
        labels, _ = partition.pca_kmeans_partition(
            X_std, k, seed=s_kmeans + 2, pca_dim=phate_dim, n_init=n_init
        )
        est.append(("pca", labels))

    for method, labels in est:
        pm = metrics.partition_metrics(region, labels)
        run_rows.append(_run_row(base_keys, method, -1, 0, None,
                                 ari=pm["ari"], purity=pm["purity"],
                                 aligned_accuracy=pm["aligned_accuracy"],
                                 cluster_sizes=pm["cluster_sizes"]))
        same_cluster_selected = False
        sel_by_region = {}
        for c in (0, 1):
            sel = _select_cluster_for_region(labels, region, c, k)
            sel_by_region[c] = sel
            if sel is None:
                continue
            if k == 2 and sel_by_region.get(1 - c) == sel:
                same_cluster_selected = True
            Xc = X_std[labels == sel]
            s_fit = seeds.derive_seed(root, "corex", method, c)
            fit = corex.fit_linear_corex(Xc, M, seed=s_fit)
            run_rows.append(
                _run_row(base_keys, method, c, 0, fit, ari=pm["ari"],
                         purity=pm["purity"],
                         aligned_accuracy=pm["aligned_accuracy"],
                         cluster_sizes=pm["cluster_sizes"],
                         same_cluster_selected=same_cluster_selected)
            )
            match_rows += _score_fit(
                base_keys, method, c, 0, fit, ds["true_groups"],
                nuisance_mask, p_total, frac,
            )

    # ---- balanced random partitions ---------------------------------------
    for draw in range(n_rand):
        s_draw = seeds.derive_seed(root, "random_partition", draw)
        labels, _ = partition.balanced_random_partition(region, seed=s_draw)
        pm = metrics.partition_metrics(region, labels)
        run_rows.append(_run_row(base_keys, "random", -1, draw, None,
                                 ari=pm["ari"], purity=pm["purity"],
                                 aligned_accuracy=pm["aligned_accuracy"],
                                 cluster_sizes=pm["cluster_sizes"]))
        for c in (0, 1):
            sel = _select_cluster_for_region(labels, region, c, k)
            if sel is None:
                continue
            Xc = X_std[labels == sel]
            s_fit = seeds.derive_seed(root, "corex", "random", draw, c)
            fit = corex.fit_linear_corex(Xc, M, seed=s_fit)
            run_rows.append(
                _run_row(base_keys, "random", c, draw, fit, ari=pm["ari"],
                         purity=pm["purity"],
                         aligned_accuracy=pm["aligned_accuracy"],
                         cluster_sizes=pm["cluster_sizes"])
            )
            match_rows += _score_fit(
                base_keys, "random", c, draw, fit, ds["true_groups"],
                nuisance_mask, p_total, frac,
            )

    store.write(condition_keys(cond), "runs", run_rows)
    store.write(condition_keys(cond), "matches", match_rows)
    return {"n_runs": len(run_rows), "n_matches": len(match_rows)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_row(base_keys, method, region, draw, fit, *, ari, purity,
             aligned_accuracy, cluster_sizes, same_cluster_selected=False):
    row = {
        **base_keys,
        "partition_method": method,
        "region": region,          # -1 = no factor fit (partition-only row)
        "partition_draw": draw,
        "corex_estimator": "linear",
        "partition_ari": ari,
        "partition_purity": purity,
        "aligned_accuracy": aligned_accuracy,
        "cluster_sizes": cluster_sizes,
        "same_cluster_selected": same_cluster_selected,
        "runtime_s": None,
        "corex_converged": None,
        "n_factors_valid": None,
        "total_tc": None,
        "fit_status": "ok",
    }
    if fit is not None:
        row.update(
            runtime_s=fit.runtime_s,
            corex_converged=fit.converged,
            n_factors_valid=fit.n_factors_valid,
            total_tc=fit.total_tc(),
        )
    return row


def _select_cluster_for_region(labels, region, c, k):
    """Estimated cluster with the largest number of observations from true
    region c. For k=2 returns the aligned cluster id (0/1); records when the
    same cluster is selected for both regions via the caller."""
    labels = np.asarray(labels)
    if k == 2:
        if len(np.unique(labels)) != 2:
            return None
        aligned, _ = metrics.align_partition(labels, region)
        return c if (aligned == c).any() else None
    overlap = np.array([((labels == e) & (region == c)).sum() for e in range(k)])
    return int(np.argmax(overlap)) if overlap.sum() > 0 else None


def _score_fit(base_keys, method, region, draw, fit, true_groups,
               nuisance_mask, p_total, frac):
    """Hungarian-match a fit's factors to the 5 true groups of `region` and
    compute all recovery metrics. The assignment is fixed across metrics."""
    assoc = fit.association_scores()
    group_keys = [(region, g) for g in range(1, 6)]
    truth = np.vstack([true_groups[k] for k in group_keys])
    matched = metrics.match_factors_to_groups(assoc, truth)
    rows = []
    for m in matched:
        g = m["true_group"]
        true_group_id = group_keys[g][1]
        truth_vec = truth[g] > 0
        scores_j = assoc[m["learned_factor"]]
        topk = metrics.topk_metrics(scores_j, truth_vec, k=4)
        sup = metrics.cumulative_mi_support(scores_j, frac)
        sup_m = metrics.support_vs_truth_metrics(sup.indices, truth_vec)
        nfp = metrics.false_positive_proportion(
            np.argsort(-scores_j, kind="stable")[:4], nuisance_mask
        )
        nfp_sup = metrics.false_positive_proportion(sup.indices, nuisance_mask)
        rows.append(
            {
                **base_keys,
                "partition_method": method,
                "region": region,
                "partition_draw": draw,
                "learned_factor": m["learned_factor"],
                "true_group": true_group_id,
                "factor_tc": float(fit.tcs[m["learned_factor"]]),
                "cosine": m["cosine"],
                "auprc": m["auprc"],
                "top4_precision": topk["precision"],
                "top4_recall": topk["recall"],
                "top4_f1": topk["f1"],
                "top4_jaccard": topk["jaccard"],
                "nfp_top4": nfp,
                "support_size": sup.size,
                "support_density": sup.density,
                "support_f1": sup_m["f1"],
                "nfp_support": nfp_sup,
                "support_degenerate": sup.degenerate,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg: dict) -> dict:
    grid_name = cfg.get("grid", "core")
    grid = GRIDS[grid_name]()
    defaults = {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS and k != "grid"}}
    # per-condition overrides live in the condition dicts themselves
    run_cfg = {
        "grid": grid_name,
        "n_factors": defaults["n_factors"],
        "phate_dim": defaults["phate_dim"],
        "k_clusters": defaults["k_clusters"],
        "n_random_partitions": defaults["n_random_partitions"],
        "kmeans_n_init": defaults["kmeans_n_init"],
        "support_fraction": defaults["support_fraction"],
        "include_pca": defaults["include_pca"],
    }
    run_id = config.make_run_id(EXPERIMENT, run_cfg, 0)
    setup_logging(EXPERIMENT, run_id)
    if config.is_complete(run_id):
        logger.info("Run %s already complete; skipping (never overwrite)", run_id)
        return {"run_id": run_id, "status": "already_complete"}
    config.save_resolved_config(run_id, run_cfg, extra={"n_conditions": len(grid)})

    store = ResultStore(EXPERIMENT)
    failure_log = FailureLog(EXPERIMENT)

    # Skip conditions already done (resumability)
    pending = [c for c in grid if not store.is_done(condition_keys(c), ("runs", "matches"))]
    logger.info("Grid %s: %d conditions (%d already done)", grid_name, len(pending),
                len(grid) - len(pending))

    def work(cond, cfg=defaults, store=store):
        return run_replicate(cond, cfg, store)

    t0 = time.time()
    run_parallel(
        functools.partial(run_replicate, cfg=defaults, store=store),
        pending,
        max_workers=int(defaults.get("n_workers") or 0),
        failure_log=failure_log,
        experiment=EXPERIMENT,
        stage_name="gaussian_replicate",
    )
    elapsed = time.time() - t0
    n_ok = store.count("runs")
    failures = failure_log.read()
    n_fail = 0 if failures.empty else len(failures)
    record = {
        "grid": grid_name,
        "n_conditions": len(grid),
        "completed_shards": n_ok,
        "failures": n_fail,
        "elapsed_s": elapsed,
    }
    config.mark_complete(run_id, record)
    from ..common.io import write_run_report
    write_run_report(EXPERIMENT, run_id, len(grid) - n_fail, n_fail,
                     {"grid": grid_name, "elapsed_s": f"{elapsed:.0f}"})
    logger.info("Run %s complete: %d shards, %d failures, %.1f s",
                run_id, n_ok, n_fail, elapsed)
    return {"run_id": run_id, **record}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--grid", type=str, default="smoke",
                    choices=["core", "nuisance", "supplementary", "smoke"])
    ap.add_argument("--n-workers", type=int, default=None)
    args = ap.parse_args()
    cfg = config.load_config(args.config) if args.config else {}
    cfg["grid"] = args.grid
    if args.n_workers is not None:
        cfg["n_workers"] = args.n_workers
    main(cfg)