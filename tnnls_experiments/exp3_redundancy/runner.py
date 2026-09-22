"""Experiment 3 runner: redundancy versus synergy diagnostic.

Grid: d in {3,4,5}, generator in {redundant, synergistic}, n in {500,5000},
32 replicates (root seeds 4100-4131), frozen TC-matched noise pairs.

Per replicate: generate group + 5d nuisance; estimate plug-in and
Miller-Madow TC/DTC/Omega on the true group; fit Bio CorEx with 3 factors;
rank by TC; take the highest-TC factor (no ground truth used for selection);
compute recovery metrics for identifying group members.
"""

from __future__ import annotations

import functools
import logging
import time

import numpy as np

from ..common import config, corex, metrics, seeds
from ..common import information
from ..common.io import FailureLog, write_run_report
from ..common.runtime import run_parallel, setup_logging
from ..common.store import ResultStore
from . import data as exp3_data

EXPERIMENT = "redundancy"
logger = logging.getLogger(f"tnnls.{EXPERIMENT}")

DEFAULTS = {
    "n_factors": 3,
    "bio_dim_hidden": 2,
    "bio_n_repeat": 1,
    "bio_max_iter": 200,
    "support_fraction": 0.80,
    "n_workers": 0,
}


def core_grid() -> list[dict]:
    pairs = exp3_data.selected_pairs()
    conds = []
    for d in (3, 4, 5):
        eta_r, eta_s = pairs[d]
        for generator, eta in (("redundant", eta_r), ("synergistic", eta_s)):
            for n in (500, 5000):
                for root_seed in range(4100, 4132):
                    conds.append({
                        "generator": generator, "d": d, "n": n, "eta": eta,
                        "root_seed": root_seed,
                    })
    return conds


def smoke_grid() -> list[dict]:
    pairs = exp3_data.selected_pairs()
    conds = []
    for d in (3,):
        eta_r, eta_s = pairs[d]
        for generator, eta in (("redundant", eta_r), ("synergistic", eta_s)):
            conds.append({
                "generator": generator, "d": d, "n": 1000, "eta": eta,
                "root_seed": 4100,
            })
    return conds


GRIDS = {"core": core_grid, "smoke": smoke_grid}


def condition_keys(cond: dict) -> dict:
    return {k: cond[k] for k in ("generator", "d", "n", "eta", "root_seed")}


def run_replicate(cond: dict, cfg: dict, store: ResultStore) -> dict:
    root = int(cond["root_seed"])
    s_data = seeds.derive_seed(root, "data")
    s_corex = seeds.derive_seed(root, "corex")

    ds = exp3_data.generate_dataset(
        cond["generator"], int(cond["d"]), int(cond["n"]), float(cond["eta"]),
        seed=s_data)
    X = ds["X"]
    d = ds["d"]
    group_idx = ds["group_idx"]
    nuisance_idx = ds["nuisance_idx"]

    # --- information quantities on the true group (plug-in + MM) ---------
    group = X[:, group_idx]
    tc_plugin = information.total_correlation_plugin(group)
    dtc_plugin = information.dual_total_correlation_plugin(group)
    omega_plugin = tc_plugin - dtc_plugin
    tc_mm, _ = information.total_correlation_plugin_mm(group)
    # DTC_MM: apply MM per joint entropy in the DTC identity
    h_joint_mm, h_joint = information.joint_entropy_mm(group)
    dtc_mm = (sum(information.joint_entropy_mm(np.delete(group, i, axis=1))[0]
                  for i in range(d)) - (d - 1) * h_joint_mm)
    omega_mm = tc_mm - dtc_mm

    # population values
    if cond["generator"] == "redundant":
        tc_pop = exp3_data.population_tc_redundant(d, float(cond["eta"]))
        omega_pop = None  # analytic O-info not implemented; plug-in serves
    else:
        tc_pop = exp3_data.population_tc_synergistic(d, float(cond["eta"]))
        omega_pop = None

    # --- CorEx fit ---------------------------------------------------------
    fit = corex.fit_bio_corex(
        X, int(cfg["n_factors"]), seed=s_corex,
        dim_hidden=int(cfg["bio_dim_hidden"]),
        n_repeat=int(cfg["bio_n_repeat"]),
        max_iter=int(cfg["bio_max_iter"]))

    # rank factors by explained TC (never by recovery ground truth)
    order = np.argsort(-fit.tcs)
    top = int(order[0])
    scores = fit.association_scores()[top]
    tc_top = float(fit.tcs[top])

    truth = np.zeros(X.shape[1])
    truth[group_idx] = 1.0
    auprc = metrics.auprc(scores, truth)
    topk = metrics.topk_metrics(scores, truth > 0, k=d)
    sup = metrics.cumulative_mi_support(scores, float(cfg["support_fraction"]))
    sup_m = metrics.support_vs_truth_metrics(sup.indices, truth > 0)
    nuisance_mask = np.zeros(X.shape[1], bool)
    nuisance_mask[nuisance_idx] = True
    nfp_top = metrics.false_positive_proportion(
        np.argsort(-scores, kind="stable")[:d], nuisance_mask)
    nfp_sup = metrics.false_positive_proportion(sup.indices, nuisance_mask)

    row = {
        "generator": cond["generator"],
        "d": d,
        "n": cond["n"],
        "eta": cond["eta"],
        "root_seed": root,
        "tc_plugin": tc_plugin,
        "dtc_plugin": dtc_plugin,
        "omega_plugin": omega_plugin,
        "tc_mm": tc_mm,
        "dtc_mm": dtc_mm,
        "omega_mm": omega_mm,
        "tc_population": tc_pop,
        "corex_factor_tc": tc_top,
        "corex_total_tc": fit.total_tc(),
        "corex_converged": fit.converged,
        "corex_runtime_s": fit.runtime_s,
        "corex_n_negative_mi": fit.n_negative_mi,
        "n_factors_valid": fit.n_factors_valid,
        "auprc": auprc,
        "topd_precision": topk["precision"],
        "topd_recall": topk["recall"],
        "topd_f1": topk["f1"],
        "topd_jaccard": topk["jaccard"],
        "support_size": sup.size,
        "support_density": sup.density,
        "support_f1": sup_m["f1"],
        "nfp_topd": nfp_top,
        "nfp_support": nfp_sup,
        "fit_status": "ok",
    }
    store.write(condition_keys(cond), "runs", [row])
    return {"row": row["fit_status"]}


def write_calibration_table() -> None:
    """Freeze the calibration to outputs/summaries (and raw for the record)."""
    from ..common.config import SUMMARIES_DIR

    rows = exp3_data.calibrate_noise_pairs()
    import pandas as pd

    df = pd.DataFrame(rows)
    # flag pairs that miss the 0.05-bit tolerance
    df["tc_match_within_tolerance"] = df["abs_tc_diff"] <= exp3_data.TC_MATCH_TOL
    df.to_csv(SUMMARIES_DIR / "redundancy_synergy_calibration.csv", index=False)
    return df


def main(cfg: dict) -> dict:
    grid_name = cfg.get("grid", "core")
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
    calibration = write_calibration_table()
    n_flagged = int((~calibration[calibration["selected"]]
                     ["tc_match_within_tolerance"]).sum())

    store = ResultStore(EXPERIMENT)
    failure_log = FailureLog(EXPERIMENT)
    pending = [c for c in grid if not store.is_done(condition_keys(c), ("runs",))]
    logger.info("Grid %s: %d conditions (%d done)", grid_name, len(pending),
                len(grid) - len(pending))

    t0 = time.time()
    run_parallel(
        functools.partial(run_replicate, cfg=defaults, store=store),
        pending,
        max_workers=int(defaults.get("n_workers") or 0),
        failure_log=failure_log,
        experiment=EXPERIMENT,
        stage_name="redundancy_replicate",
    )
    elapsed = time.time() - t0
    failures = failure_log.read()
    n_fail = 0 if failures.empty else len(failures)
    record = {"grid": grid_name, "n_conditions": len(grid),
              "completed_shards": store.count("runs"),
              "failures": n_fail, "elapsed_s": elapsed,
              "calibration_pairs_flagged": n_flagged}
    config.mark_complete(run_id, record)
    write_run_report(EXPERIMENT, run_id, len(grid) - n_fail, n_fail,
                     {"grid": grid_name, "elapsed_s": f"{elapsed:.0f}"})
    logger.info("Run %s complete: %d shards, %d failures, %d flagged pairs",
                run_id, store.count("runs"), n_fail, n_flagged)
    return {"run_id": run_id, **record}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=str, default="smoke", choices=list(GRIDS))
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--n-workers", type=int, default=None)
    args = ap.parse_args()
    cfg = config.load_config(args.config) if args.config else {}
    cfg["grid"] = args.grid
    if args.n_workers is not None:
        cfg["n_workers"] = args.n_workers
    main(cfg)