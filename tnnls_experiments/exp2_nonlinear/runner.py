"""Experiment 2 runner: nonlinear and higher-order recovery.

Bio CorEx is the primary estimator; Linear CorEx is a limited-capability
baseline. Partition methods: global, oracle, PHATE, balanced random (10
draws), raw-space k-means.

Routing variables are included in PHATE partitioning and in CorEx fitting
(primary condition); ground-truth scoring excludes them. A secondary
condition excludes routing variables from CorEx fitting.

For the continuous generator, the fixed discretization rule is 10-bin
equal-frequency binning estimated on the fitting data (edges saved); a
secondary variant fits BioCorex with native gaussian marginals.
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
from . import data as exp2_data

EXPERIMENT = "nonlinear"
logger = logging.getLogger(f"tnnls.{EXPERIMENT}")

# Core keys shared by discrete and continuous match rows (validation test 8)
MATCH_COLS = (
    "generator", "n_per_region", "alpha", "noise", "n_nuisance", "root_seed",
    "n_factors", "partition_method", "region", "partition_draw", "estimator",
    "corex_input", "learned_factor", "true_group", "factor_tc", "cosine",
    "auprc", "topd_precision", "topd_recall", "topd_f1", "topd_jaccard",
    "routing_fp", "nuisance_fp", "support_size", "support_density",
    "support_f1", "routing_fp_support", "nuisance_fp_support",
    "support_degenerate",
)

DEFAULTS = {
    "n_factors": 4,             # M = 4 requested factors (spec)
    "phate_dim": 10,
    "k_clusters": 2,
    "n_random_partitions": 10,
    "kmeans_n_init": 50,
    "n_bins": 10,               # discretization bins for continuous data
    "routing_in_corex": True,   # primary condition per spec
    "native_gaussian_bio": False,  # secondary continuous variant
    "support_fraction": 0.80,
    "n_workers": 0,
    "bio_max_iter": 200,
    "bio_dim_hidden": 4,        # latent states: >= 2^group-capacity for parity
    "bio_n_repeat": 5,          # EM restarts (escape trivial basin if possible)
    "diagnostic_n": 100_000,
    "capability_diagnostic": True,
}


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def discrete_core_grid() -> list[dict]:
    return [
        {"generator": "parity", "n_per_region": n, "alpha": alpha, "noise": eta,
         "n_nuisance": 14, "root_seed": seed}
        for n in (500, 5000)
        for alpha in (0.0, 0.5, 1.0)
        for eta in (0.0, 0.05, 0.15)
        for seed in range(2000, 2016)
    ]


def continuous_core_grid() -> list[dict]:
    return [
        {"generator": "sine_sum", "n_per_region": n, "alpha": alpha,
         "noise": sigma, "n_nuisance": 14, "root_seed": seed}
        for n in (500, 5000)
        for alpha in (0.0, 0.5, 1.0)
        for sigma in (0.05, 0.20, 0.50)
        for seed in range(3000, 3016)
    ]


def discrete_supplementary_grid() -> list[dict]:
    conds = []
    base = {"generator": "parity", "n_nuisance": 14}
    # small-sample regime
    for alpha in (0.0, 0.5, 1.0):
        for eta in (0.0, 0.05, 0.15):
            for seed in range(2000, 2016):
                conds.append({**base, "n_per_region": 50, "alpha": alpha,
                              "noise": eta, "root_seed": seed})
    # extra noise levels
    for eta in (0.10, 0.20):
        for seed in range(2000, 2016):
            conds.append({**base, "n_per_region": 500, "alpha": 0.5,
                          "noise": eta, "root_seed": seed})
    # nuisance sweep
    for p_n in (0, 56):
        for seed in range(2000, 2016):
            conds.append({**base, "n_per_region": 500, "alpha": 0.5, "noise": 0.05,
                          "n_nuisance": p_n, "root_seed": seed})
    return conds


def continuous_supplementary_grid() -> list[dict]:
    conds = []
    base = {"generator": "sine_sum", "n_nuisance": 14}
    for alpha in (0.0, 0.5, 1.0):
        for sigma in (0.05, 0.20, 0.50):
            for seed in range(3000, 3016):
                conds.append({**base, "n_per_region": 50, "alpha": alpha,
                              "noise": sigma, "root_seed": seed})
    for p_n in (0, 56):
        for seed in range(3000, 3016):
            conds.append({**base, "n_per_region": 500, "alpha": 0.5, "noise": 0.20,
                          "n_nuisance": p_n, "root_seed": seed})
    # product-group comparison at the core condition
    for sigma in (0.05, 0.20, 0.50):
        for seed in range(3000, 3016):
            conds.append({**base, "generator": "product", "n_per_region": 500,
                          "alpha": 0.5, "noise": sigma, "root_seed": seed})
    return conds


def smoke_grid() -> list[dict]:
    return [
        {"generator": "parity", "n_per_region": 300, "alpha": 0.5, "noise": 0.05,
         "n_nuisance": 14, "root_seed": 2000},
        {"generator": "sine_sum", "n_per_region": 300, "alpha": 0.5, "noise": 0.20,
         "n_nuisance": 14, "root_seed": 3000},
    ]


GRIDS = {
    "discrete_core": discrete_core_grid,
    "continuous_core": continuous_core_grid,
    "discrete_supplementary": discrete_supplementary_grid,
    "continuous_supplementary": continuous_supplementary_grid,
    "smoke": smoke_grid,
}


# ---------------------------------------------------------------------------
# Per-replicate execution
# ---------------------------------------------------------------------------

def condition_keys(cond: dict) -> dict:
    keys = ("generator", "n_per_region", "alpha", "noise", "n_nuisance",
            "root_seed", "corex_input", "estimator")
    return {k: cond[k] for k in keys if k in cond}


def run_replicate(cond: dict, cfg: dict, store: ResultStore) -> dict:
    root = int(cond["root_seed"])
    s_data = seeds.derive_seed(root, "data")
    s_kmeans = seeds.derive_seed(root, "partition", "kmeans")
    s_corex = seeds.derive_seed(root, "corex")

    generator = cond["generator"]
    noise = float(cond["noise"])
    if generator == "parity":
        ds = exp2_data.generate_discrete_dataset(
            int(cond["n_per_region"]), float(cond["alpha"]), noise,
            int(cond["n_nuisance"]), seed=s_data)
        eta, sigma = noise, None
    else:
        ds = exp2_data.generate_continuous_dataset(
            int(cond["n_per_region"]), float(cond["alpha"]), noise,
            int(cond["n_nuisance"]), seed=s_data,
            product_groups=(generator == "product"))
        eta, sigma = None, noise

    region = ds["region"]
    routing_idx = ds["routing_idx"]
    nuisance_idx = ds["nuisance_idx"]

    base_keys = {
        "generator": generator,
        "n_per_region": cond["n_per_region"],
        "alpha": cond["alpha"],
        "noise": noise,
        "n_nuisance": cond["n_nuisance"],
        "root_seed": root,
        "n_factors": cfg["n_factors"],
    }

    M = cfg["n_factors"]
    k = cfg["k_clusters"]
    n_rand = cfg["n_random_partitions"]
    frac = cfg["support_fraction"]

    # ----- prepare fitting matrices per condition ---------------------------
    # CorEx input variants:
    #   primary: all variables incl. routing (discrete: raw bits; continuous:
    #            standardized then binned)
    #   no_routing: routing columns dropped from CorEx fitting
    corex_input = "with_routing" if cfg["routing_in_corex"] else "no_routing"
    est_variants = []
    # estimator variants: bio (primary), linear (baseline); continuous also
    # has native gaussian bio variant if requested
    if generator == "parity":
        X_all = ds["X"]  # signal+nuisance are 0/1; routing are continuous
        # CorEx input requires nonnegative ints for discrete BioCorex: bin
        # the two routing variables with the fixed equal-frequency rule.
        X_routing = ds["X"][:, ds["routing_idx"]]
        X_routing_b, _ = corex.discretize_equal_frequency(X_routing, cfg["n_bins"])
        X_all = ds["X"].copy()
        X_all[:, ds["routing_idx"]] = X_routing_b
        variants = [("bio", "with_routing"), ("linear", "with_routing")]
        if not cfg["routing_in_corex"]:
            variants = [("bio", "no_routing"), ("linear", "no_routing")]
    else:
        X_cont = ds["X"]
        X_std, _, _ = corex.standardize_columns(X_cont)
        Xb, _ = corex.discretize_equal_frequency(X_std, cfg["n_bins"])
        variants = [("bio", "binned_with_routing"), ("linear", "binned_with_routing")]
        if cfg["native_gaussian_bio"]:
            variants.append(("bio_gaussian", "gaussian_with_routing"))
    if not cfg["routing_in_corex"] and generator != "parity":
        variants = [(e, i.replace("with_routing", "no_routing")) for e, i in variants]

    run_rows: list[dict] = []
    match_rows: list[dict] = []

    # ----- partition once (on the primary representation) -------------------
    # PHATE input: all variables (signal + routing + nuisance); standardized
    # in both generators so scales are comparable.
    if generator == "parity":
        X_phate, _, _ = corex.standardize_columns(ds["X"].astype(float))
    else:
        X_phate = X_std
    labels_phate, _ = partition.phate_kmeans_partition(
        X_phate, k, seed=s_kmeans, phate_dim=cfg["phate_dim"],
        n_init=cfg["kmeans_n_init"], n_pca=min(50, X_phate.shape[1] - 1))
    labels_raw, _ = partition.raw_kmeans_partition(
        X_phate, k, seed=s_kmeans + 1, n_init=cfg["kmeans_n_init"])

    partition_specs: list[tuple[str, np.ndarray]] = [
        ("global", None),
        ("oracle", None),
        ("phate", labels_phate),
        ("raw_kmeans", labels_raw),
    ]

    for method, labels in partition_specs:
        _fit_and_score_partitions(
            cond, cfg, ds, method, labels, region, M, k, frac,
            s_corex, base_keys, routing_idx, nuisance_idx, variants, generator,
            X_all if generator == "parity" else None,
            Xb if generator != "parity" else None,
            X_std if generator != "parity" else None,
            run_rows, match_rows, draw=0,
        )

    # random partitions
    for draw in range(n_rand):
        s_draw = seeds.derive_seed(root, "random_partition", draw)
        labels, _ = partition.balanced_random_partition(region, seed=s_draw)
        _fit_and_score_partitions(
            cond, cfg, ds, "random", labels, region, M, k, frac,
            s_corex, base_keys, routing_idx, nuisance_idx, variants, generator,
            X_all if generator == "parity" else None,
            Xb if generator != "parity" else None,
            X_std if generator != "parity" else None,
            run_rows, match_rows, draw=draw,
        )

    store.write(condition_keys(cond) | {"corex_input": "ALL", "estimator": "ALL"},
                "runs", run_rows)
    store.write(condition_keys(cond) | {"corex_input": "ALL", "estimator": "ALL"},
                "matches", match_rows)
    return {"n_runs": len(run_rows), "n_matches": len(match_rows)}


def _fit_and_score_partitions(
    cond, cfg, ds, method, labels, region, M, k, frac,
    s_corex, base_keys, routing_idx, nuisance_idx, variants, generator,
    X_all, Xb, X_std, run_rows, match_rows, draw,
):
    """Fit all estimator variants in every cell of one partition and score
    region-specific recovery.

    Cell -> region mapping (spec): global fits are matched independently for
    each true region; oracle cells map naturally; estimated partitions use
    the cluster with the largest overlap with each true region.
    """
    if method == "global":
        cell_specs = [("global", np.arange(len(region)), {0, 1})]
    elif method == "oracle":
        cell_specs = [("region0", np.flatnonzero(region == 0), {0}),
                      ("region1", np.flatnonzero(region == 1), {1})]
    else:
        aligned, _ = metrics.align_partition(labels, region)
        pm = metrics.partition_metrics(region, labels)
        # for k=2 the aligned labels map clusters to regions directly; for
        # k!=2 select the cluster with largest overlap per region (no forced
        # one-to-one map; if the same cluster is selected for both regions,
        # both region scores use that cluster's factors, and the fact is
        # recorded on the run row)
        sel_by_region = {}
        for c in (0, 1):
            if k == 2:
                sel_by_region[c] = c if (aligned == c).any() else None
            else:
                overlaps = [((labels == e) & (region == c)).sum() for e in range(k)]
                sel_by_region[c] = int(np.argmax(overlaps)) if sum(overlaps) > 0 else None
        cell_specs = []
        seen = set()
        for c in (0, 1):
            sel = sel_by_region[c]
            if sel is None or sel in seen:
                continue
            seen.add(sel)
            cell_specs.append((f"cluster{sel}", np.flatnonzero(labels == sel),
                               {c for c2 in (0, 1) if sel_by_region[c2] == sel}))
        run_rows.append({
            **base_keys, "partition_method": method, "region": -1,
            "partition_draw": draw, "partition_ari": pm["ari"],
            "partition_purity": pm["purity"],
            "aligned_accuracy": pm["aligned_accuracy"],
            "cluster_sizes": pm["cluster_sizes"], "fit_status": "ok",
            "same_cluster_selected": sel_by_region.get(0) == sel_by_region.get(1),
        })

    for estimator, variant in variants:
        for cell_name, cell_idx, regions_served in cell_specs:
            s_fit = seeds.derive_seed(s_corex, estimator, method, cell_name)
            fit, fit_row, fit_status = _fit_estimator(
                cfg, ds, estimator, variant, cell_idx, M, s_fit,
                routing_idx, generator, X_all, Xb, X_std)
            if cell_name == "global":
                cell_id = -1
            else:
                cell_id = int(cell_name.replace("region", "").replace("cluster", ""))
            if fit_row is not None:
                row = {**base_keys, "partition_method": method,
                       "estimator": estimator, "corex_input": variant,
                       "region": cell_id,
                       "partition_draw": draw, "fit_status": fit_status,
                       **fit_row}
                run_rows.append(row)
            if fit is not None:
                for c in regions_served:
                    match_rows += _score_fit(
                        base_keys, method, c, draw, estimator, variant, fit,
                        ds["true_groups"], routing_idx, nuisance_idx,
                        ds["X"].shape[1], frac)
    return


def _fit_estimator(cfg, ds, estimator, variant, cell_idx, M, s_fit,
                   routing_idx, generator, X_all, Xb, X_std):
    """Fit one estimator on one partition cell. Returns (fit, row, status)."""
    import time as _time

    if generator == "parity":
        X = X_all
        if variant == "no_routing":
            keep = np.ones(X.shape[1], bool)
            keep[routing_idx] = False
            X = X[:, keep]
    else:
        if variant.startswith("gaussian"):
            X = X_std
        else:
            X = Xb
        if variant.endswith("no_routing"):
            keep = np.ones(X.shape[1], bool)
            keep[routing_idx] = False
            X = X[:, keep]
    X_cell = X[cell_idx]

    t0 = _time.time()
    try:
        if estimator == "linear":
            fit = corex.fit_linear_corex(X_cell, M, seed=s_fit, max_iter=2000)
        elif estimator == "bio":
            fit = corex.fit_bio_corex(
                X_cell, M, seed=s_fit, max_iter=cfg["bio_max_iter"],
                dim_hidden=cfg["bio_dim_hidden"], n_repeat=cfg["bio_n_repeat"])
        elif estimator == "bio_gaussian":
            fit = corex.fit_bio_corex(
                X_cell, M, seed=s_fit, marginal_description="gaussian",
                max_iter=cfg["bio_max_iter"],
                dim_hidden=cfg["bio_dim_hidden"], n_repeat=cfg["bio_n_repeat"])
        else:
            raise ValueError(estimator)
    except Exception as e:
        return None, {
            "runtime_s": _time.time() - t0, "corex_converged": False,
            "n_factors_valid": 0, "total_tc": None,
            "failure_stage": "corex_fit", "exception": str(e),
        }, "failed"
    return fit, {
        "runtime_s": fit.runtime_s, "corex_converged": fit.converged,
        "n_factors_valid": fit.n_factors_valid, "total_tc": fit.total_tc(),
        "failure_stage": None, "exception": None,
    }, "ok"


def _score_fit(base_keys, method, region, draw, estimator, variant, fit,
               true_groups, routing_idx, nuisance_idx, p_total, frac):
    """Match factors to the 4 true groups of `region`; routing + nuisance FP."""
    assoc = fit.association_scores()
    group_keys = [(region, g) for g in range(1, 5)]
    truth = np.vstack([true_groups[gk] for gk in group_keys])
    # routing columns are zero in all truth vectors and must not be scored
    matched = metrics.match_factors_to_groups(assoc, truth)
    routing_mask = np.zeros(p_total, bool)
    routing_mask[routing_idx] = True
    nuisance_mask = np.zeros(p_total, bool)
    nuisance_mask[nuisance_idx] = True
    rows = []
    for m in matched:
        g = m["true_group"]
        truth_vec = truth[g] > 0
        d_true = int(truth_vec.sum())
        scores_j = assoc[m["learned_factor"]]
        topk = metrics.topk_metrics(scores_j, truth_vec, k=d_true)
        sup = metrics.cumulative_mi_support(scores_j, frac)
        sup_m = metrics.support_vs_truth_metrics(sup.indices, truth_vec)
        top_idx = np.argsort(-scores_j, kind="stable")[:d_true]
        rfp = metrics.false_positive_proportion(top_idx, routing_mask)
        nfp = metrics.false_positive_proportion(top_idx, nuisance_mask)
        rfp_sup = metrics.false_positive_proportion(sup.indices, routing_mask)
        nfp_sup = metrics.false_positive_proportion(sup.indices, nuisance_mask)
        rows.append({
            **base_keys,
            "partition_method": method,
            "region": region,
            "partition_draw": draw,
            "estimator": estimator,
            "corex_input": variant,
            "learned_factor": m["learned_factor"],
            "true_group": group_keys[g][1],
            "factor_tc": float(fit.tcs[m["learned_factor"]]),
            "cosine": m["cosine"],
            "auprc": m["auprc"],
            "topd_precision": topk["precision"],
            "topd_recall": topk["recall"],
            "topd_f1": topk["f1"],
            "topd_jaccard": topk["jaccard"],
            "routing_fp": rfp,
            "nuisance_fp": nfp,
            "support_size": sup.size,
            "support_density": sup.density,
            "support_f1": sup_m["f1"],
            "routing_fp_support": rfp_sup,
            "nuisance_fp_support": nfp_sup,
            "support_degenerate": sup.degenerate,
        })
    return rows


# ---------------------------------------------------------------------------
# Structure diagnostics
# ---------------------------------------------------------------------------

def run_capability_diagnostic(cfg: dict) -> None:
    """Capability diagnostic for pairwise-independent structure (Exp 2).

    Demonstrates, per group size d and noise eta, that:
      (a) standard EM initializations find TC ~ 0 on pure parity groups
          (dim_hidden=4, restarts), while
      (b) a ground-truth atom-seeded latent init achieves the full parity TC,
          showing the variational family CAN represent the structure and the
          failure is in the estimator's optimization, not its expressiveness.

    Seeded fits are labeled uses_ground_truth_init=true. These rows NEVER
    enter recovery-metric summaries (they are a capability statement only).
    """
    store = ResultStore(EXPERIMENT)
    n = int(cfg.get("diagnostic_n", 100_000))
    for d in (3, 4):
        for eta in (0.0, 0.05, 0.15):
            cond = {"diagnostic": "capability", "group_size": d, "eta": eta, "n": n}
            if store.is_done(cond, ("capability",)):
                continue
            s = seeds.derive_seed(0, "capability", d, eta)
            rng = np.random.default_rng(s)
            block = exp2_data.parity_group(rng, d, n, eta)
            # (a) standard init
            fit_std = corex.fit_bio_corex(
                block, 1, seed=s, max_iter=cfg["bio_max_iter"],
                dim_hidden=cfg["bio_dim_hidden"], n_repeat=cfg["bio_n_repeat"])
            # (b) atom-seeded init (ground truth): one latent per joint atom.
            # Encode the full joint pattern uniquely (bit sums collide for
            # parity: e.g. (0,0,0) and (1,1,0) both sum to 0/2 patterns).
            powers = 2 ** np.arange(d)
            atom_keys = block @ powers
            fit_atom = corex.fit_bio_corex(
                block, 1, seed=s, max_iter=cfg["bio_max_iter"],
                dim_hidden=max(cfg["bio_dim_hidden"], 2 ** d),
                n_repeat=1, atom_keys=atom_keys)
            row = {
                "diagnostic": "capability",
                "group_size": d,
                "eta": eta,
                "n": n,
                "true_tc_bits": float(d - 1) if eta == 0 else None,
                "std_init_tc": float(fit_std.tcs[0]),
                "seeded_init_tc": float(fit_atom.tcs[0]),
                "seeded_init_uses_ground_truth": True,
                "note": ("standard EM finds ~0 TC on pairwise-independent "
                         "parity; the family can represent it when seeded "
                         "with the joint atom. Optimization failure, not "
                         "expressiveness failure."),
            }
            store.write(cond, "capability", [row])
            logger.info("capability d=%d eta=%.2f: std TC=%.4f, seeded TC=%.4f "
                        "(true %s bits)", d, eta, fit_std.tcs[0],
                        fit_atom.tcs[0], d - 1 if eta == 0 else "?")


def run_structure_diagnostics(cfg: dict) -> None:
    """Diagnostic runs with independent large samples (>=100k)."""
    store = ResultStore(EXPERIMENT)
    rows = []
    n = int(cfg.get("diagnostic_n", 100_000))
    conds = (
        [("parity", d, eta) for d in (3, 4) for eta in (0.0, 0.05, 0.15)]
        + [("sine_sum", d, sigma) for d in (3, 4) for sigma in (0.05, 0.20, 0.50)]
        + [("product", 4, sigma) for sigma in (0.20,)]
    )
    for generator, d, noise in conds:
        cond = {"generator": generator, "group_size": d, "noise": noise, "n": n}
        if store.is_done(cond, ("structure",)):
            continue
        diag = exp2_data.structure_diagnostics(generator, d, noise, n,
                                               seed=seeds.derive_seed(0, "diag", generator, d, noise))
        store.write(cond, "structure", [diag])
        rows.append(diag)
        logger.info("diagnostic %s d=%d noise=%.2f: TC=%.3f permTC=%.3f pairwiseMI=%.4f",
                    generator, d, noise, diag["group_tc"], diag["permuted_group_tc"],
                    diag["max_pairwise_mi"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg: dict) -> dict:
    grid_name = cfg.get("grid", "discrete_core")
    grid = GRIDS[grid_name]()
    defaults = {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS and k != "grid"}}
    run_cfg = {k: v for k, v in defaults.items() if k != "n_workers"}
    run_id = config.make_run_id(EXPERIMENT, {"grid": grid_name, **run_cfg}, 0)
    setup_logging(EXPERIMENT, run_id)
    if config.is_complete(run_id):
        logger.info("Run %s already complete; skipping", run_id)
        return {"run_id": run_id, "status": "already_complete"}
    config.save_resolved_config(run_id, {"grid": grid_name, **run_cfg},
                                extra={"n_conditions": len(grid)})

    if defaults.get("run_diagnostics", True):
        run_structure_diagnostics(defaults)
    if defaults.get("capability_diagnostic", True):
        run_capability_diagnostic(defaults)

    store = ResultStore(EXPERIMENT)
    failure_log = FailureLog(EXPERIMENT)
    pending = [c for c in grid
               if not store.is_done(condition_keys(c) | {"corex_input": "ALL", "estimator": "ALL"},
                                    ("runs", "matches"))]
    logger.info("Grid %s: %d conditions (%d already done)", grid_name, len(pending),
                len(grid) - len(pending))

    t0 = time.time()
    run_parallel(
        functools.partial(run_replicate, cfg=defaults, store=store),
        pending,
        max_workers=int(defaults.get("n_workers") or 0),
        failure_log=failure_log,
        experiment=EXPERIMENT,
        stage_name="nonlinear_replicate",
    )
    elapsed = time.time() - t0
    failures = failure_log.read()
    n_fail = 0 if failures.empty else len(failures)
    record = {"grid": grid_name, "n_conditions": len(grid),
              "completed_shards": store.count("runs"),
              "failures": n_fail, "elapsed_s": elapsed}
    config.mark_complete(run_id, record)
    from ..common.io import write_run_report
    write_run_report(EXPERIMENT, run_id, len(grid) - n_fail, n_fail,
                     {"grid": grid_name, "elapsed_s": f"{elapsed:.0f}"})
    return {"run_id": run_id, **record}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=str, default="smoke",
                    choices=list(GRIDS.keys()))
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--n-workers", type=int, default=None)
    ap.add_argument("--skip-diagnostics", action="store_true")
    args = ap.parse_args()
    cfg = config.load_config(args.config) if args.config else {}
    cfg["grid"] = args.grid
    if args.n_workers is not None:
        cfg["n_workers"] = args.n_workers
    if args.skip_diagnostics:
        cfg["run_diagnostics"] = False
    main(cfg)