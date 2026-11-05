"""Global Linear CorEx comparison arm for the MNIST benchmark (added per the
paper-plan review, fb#2: "Global CorEx supports — what localization adds
within the same factor-learning framework").

Design (frozen before outcome evaluation):

- One global Linear CorEx fit per (model seed, layer, input_condition,
  budget) on the FULL 10k-discovery sample — the same conventions as the
  local fits (standardize valid hidden coords; output-aware fits see
  row-centered standardized logits stacked after H), but
  partition=global.
- Budgets: M=10 (per-fit parity with the local fits) and M=100
  (factor-budget-matched: 20 clusters x 10 factors). M=10 is the
  workflow-parity reading; M=100 is the capacity-matched arm.
- Retained factors: top-5 by explained TC (same TC rank rule as local).
- Supports: cumulative-MI at the primary thresholds [0.6, 0.7, 0.8, 0.9],
  same tie rule as local.
- Interventions: identical activation-mask machinery (TrunkEvaluator) on
  the identical test set — common held-out cohorts with every local arm.
  Per support: Local CorEx + 20 size-matched random supports (same count
  as the local output-aware arm).
- Per-cluster localization: image-level TV grouped by the fixed test-set
  cluster assignment for every intervention, so global factors' spatial
  concentration is comparable with local factors on common cohorts.
- Reporting unit: (model seed, budget) pooling for factor-level summaries;
  NOT independent network replications.
- No test-set statistic feeds fit/support/rank decisions.

Writes under the same ResultStore namespace in its own tables
(global_factors, global_supports, global_ablation, global_cluster_tv);
no existing exp4 shard is touched.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import torch

from common import config, seeds
from common.io import FailureLog, write_run_report
from common.runtime import setup_logging
from common.store import ResultStore
from benchmark import (
    LayerContext, N_CLUSTERS, NATS_TO_BITS, TrunkEvaluator,
    build_discovery_clusters, build_test_cluster_assignment,
    extract_representations, fit_and_rank, mapped_supports,
    posthoc_logit_mi, _summarize_intervention)
from classifier import MLP
from data import load_splits

RUN_NAME = "global_corex_arm"
logger = logging.getLogger(f"tnnls.{RUN_NAME}")

MODEL_SEEDS = (5100, 5101, 5102, 5103, 5104)
BUDGETS = (10, 100)
N_RETAINED_GLOBAL = 5
SUPPORT_THRESHOLDS = (0.60, 0.70, 0.80, 0.90)
N_RANDOM_CONTROLS = 20
CONDITIONS = ("output_aware", "hidden_only")


def checkpoint_path(model_seed: int):
    return config.CHECKPOINTS_DIR / "mnist" / f"mnist_seed{model_seed}.pt"


def load_model(model_seed: int, device: str) -> MLP:
    model = MLP().to(device)
    model.load_state_dict(
        torch.load(checkpoint_path(model_seed), map_location=device))
    model.eval()
    return model


def fit_seed(model_seed: int, cond: str, layer: int, budget: int) -> int:
    return seeds.derive_seed(model_seed, "corex", "global", cond, layer,
                             budget)


def build_plan(store: ResultStore, model_seed: int) -> dict:
    """(cond, layer, budget, rank) -> {"dominant": int, "supports": {q: idx}}
    from the stored global_supports shard (fit stage may have run earlier)."""
    df = store.read_all("global_supports")
    df = df[df.model_seed == model_seed]
    plans = {}
    for r in df.itertuples():
        key = (str(r.input_condition), int(r.layer), int(r.budget),
               int(r.factor_rank))
        entry = plans.setdefault(key, {"dominant": None, "supports": {}})
        entry["supports"][float(r.threshold)] = json.loads(r.support)
        entry["dominant"] = int(r.dominant_logit)
    return plans


# ---------------------------------------------------------------------------
# Stage 1: global fits + supports
# ---------------------------------------------------------------------------

def run_global_fit_stage(model_seed: int, model: MLP, splits: dict,
                         device: str, store: ResultStore,
                         budgets: tuple = BUDGETS) -> dict:
    """Fit on the full discovery sample; write global factor + support
    shards; return the ablation-stage plan."""
    cond_key = {"stage": "global_fit", "model_seed": model_seed}
    if store.is_done(cond_key, ("global_factors", "global_supports")):
        return build_plan(store, model_seed)
    with torch.no_grad():
        h1, h2, h3, logits = extract_representations(
            model, splits["discovery_x"].to(device), device)
    factor_rows, support_rows, dominants = [], [], {}
    for layer in (1, 2, 3):
        ctx = LayerContext(h1, h2, h3, logits, layer)
        for cond in CONDITIONS:
            L_for_fit = ctx.L_std if cond == "output_aware" else None
            for budget in budgets:
                s_fit = fit_seed(model_seed, cond, layer, budget)
                fr = fit_and_rank(ctx.H_std, L_for_fit, budget, s_fit,
                                  N_RETAINED_GLOBAL, valid_h=ctx.valid_h)
                if fr["logit_mis"] is None:
                    v_idx = np.flatnonzero(ctx.valid_h)
                    fr["logit_mis"], fr["dominant"], fr["gaps"] = \
                        posthoc_logit_mi(fr["fit"], ctx.H_std[:, v_idx],
                                         ctx.L_std)
                    logit_source = "post_hoc"
                else:
                    logit_source = "fit"
                common = {
                    "model_seed": model_seed, "partition": "global",
                    "budget": int(budget), "layer": layer,
                    "input_condition": cond,
                    "corex_seed": s_fit,
                    "logit_mi_source": logit_source,
                }
                for rank, j in enumerate(fr["order"]):
                    hmis = fr["hidden_mis"][j]
                    sups = mapped_supports(
                        hmis, list(SUPPORT_THRESHOLDS), ctx.valid_h,
                        seeds.derive_seed(s_fit, "tie"))
                    factor_rows.append({
                        **common, "factor_index": int(j),
                        "factor_rank": rank,
                        "factor_tc": float(fr["fit"].tcs[j]) * NATS_TO_BITS,
                        "corex_converged": fr["fit"].converged,
                        "n_negative_mi": fr["fit"].n_negative_mi,
                        "n_valid_hidden": int(ctx.valid_h.sum()),
                        "n_excluded_hidden": int((~ctx.valid_h).sum()),
                        "corex_runtime_s": fr["fit"].runtime_s,
                        "dominant_logit": int(fr["dominant"][j]),
                    })
                    dominants[(cond, layer, int(budget), rank)] = \
                        int(fr["dominant"][j])
                    for q, sup in zip(SUPPORT_THRESHOLDS, sups):
                        support_rows.append({
                            **common, "factor_index": int(j),
                            "factor_rank": rank, "threshold": float(q),
                            "support_size": len(sup.indices),
                            "support_density": len(sup.indices)
                            / ctx.H_std.shape[1],
                            "support": json.dumps(list(sup.indices)),
                            "dominant_logit": int(fr["dominant"][j]),
                        })
    store.write(cond_key, "global_factors", factor_rows)
    store.write(cond_key, "global_supports", support_rows)
    plans = build_plan(store, model_seed)
    for key, entry in plans.items():
        entry["dominant"] = dominants.get(key)
    return plans


# ---------------------------------------------------------------------------
# Stage 2: interventions on the identical held-out test set
# ---------------------------------------------------------------------------

def tv_grouped(tv: np.ndarray, test_cluster: np.ndarray) -> list:
    """Mean image-level TV per test cluster (-1 for empty clusters)."""
    out = []
    for k in range(N_CLUSTERS):
        idx = np.flatnonzero(test_cluster == k)
        out.append(float(tv[idx].mean()) if len(idx) else -1.0)
    return out


def run_global_ablation_stage(model_seed: int, model: MLP, splits: dict,
                              labels_cluster: np.ndarray, device: str,
                              store: ResultStore, cfg: dict) -> None:
    plans = run_global_fit_stage(model_seed, model, splits, device, store)
    cond_key = {"stage": "global_ablation", "model_seed": model_seed}
    if store.is_done(cond_key, ("global_ablation", "global_cluster_tv")):
        return
    trunk = TrunkEvaluator(model, splits["test_x"], device)
    labels_test = splits["test_y"].numpy()
    test_cluster = build_test_cluster_assignment(splits, store, N_CLUSTERS)
    abl_rows, tv_rows = [], []
    for (cond, layer, budget, rank) in sorted(plans):
        entry = plans[(cond, layer, budget, rank)]
        dominant = entry["dominant"]
        for q, sup in entry["supports"].items():
            if not sup:
                continue
            s = len(sup)
            base_int = {
                "model_seed": model_seed, "partition": "global",
                "budget": int(budget), "layer": layer,
                "input_condition": cond, "factor_rank": rank,
                "threshold": float(q), "support_size": s,
                "dominant_logit": dominant if dominant is not None else -1,
            }
            p_abl = trunk.ablated_probs(layer, sup)
            abl_rows.append(_summarize_intervention(
                base_int, "global_corex", 0, p_abl, trunk.baseline_probs,
                labels_test, dominant,
                removed_act_mag=trunk.mean_removed_activation(layer, sup)))
            tv = 0.5 * np.abs(p_abl - trunk.baseline_probs).sum(axis=1)
            tv_rows.append({**base_int, "method": "global_corex",
                            "random_draw": 0,
                            "cluster_tv": json.dumps(tv_grouped(tv,
                                                                test_cluster))})
            for draw in range(cfg["n_random_controls"]):
                s_r = seeds.derive_seed(model_seed, "global_rand", cond,
                                        layer, budget, rank, float(q), draw)
                rng = np.random.default_rng(s_r)
                rand_sup = rng.choice(trunk.h[layer].shape[1], size=s,
                                      replace=False).tolist()
                p_r = trunk.ablated_probs(layer, rand_sup)
                abl_rows.append(_summarize_intervention(
                    base_int, "random", draw, p_r, trunk.baseline_probs,
                    labels_test, dominant,
                    removed_act_mag=trunk.mean_removed_activation(layer,
                                                                  rand_sup)))
                tv_r = 0.5 * np.abs(p_r - trunk.baseline_probs).sum(axis=1)
                tv_rows.append({**base_int, "method": "random",
                                "random_draw": draw,
                                "cluster_tv": json.dumps(
                                    tv_grouped(tv_r, test_cluster))})
    store.write(cond_key, "global_ablation", abl_rows)
    store.write(cond_key, "global_cluster_tv", tv_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(model_seeds: tuple = MODEL_SEEDS, device: str | None = None,
         n_random: int = N_RANDOM_CONTROLS,
         budgets: tuple = BUDGETS) -> dict:
    budgets = tuple(budgets)
    run_id = f"{RUN_NAME}_seeds" + "-".join(map(str, model_seeds)) \
        + "_M" + "-".join(map(str, budgets))
    setup_logging(RUN_NAME, run_id)
    store = ResultStore("mnist")
    failure_log = FailureLog(RUN_NAME)
    cfg = {"n_random_controls": int(n_random), "n_clusters": N_CLUSTERS,
           "thresholds": list(SUPPORT_THRESHOLDS), "budgets": list(budgets),
           "n_retained": N_RETAINED_GLOBAL}
    config.save_resolved_config(run_id, cfg)
    splits = load_splits(save_dir=config.RAW_DIR / "mnist")
    labels_cluster = build_discovery_clusters(splits, store, N_CLUSTERS)
    done = 0
    for model_seed in model_seeds:
        try:
            model = load_model(model_seed, device)
            run_global_fit_stage(model_seed, model, splits, device, store,
                                 budgets=budgets)
            run_global_ablation_stage(model_seed, model, splits,
                                      labels_cluster, device, store, cfg)
            done += 1
            logger.info("global arm: seed %d done", model_seed)
        except Exception as e:  # noqa: BLE001
            import traceback
            failure_log.record(condition={"model_seed": model_seed},
                               seed=model_seed, stage="global_arm", exc=e,
                               message=traceback.format_exc(limit=3))
    write_run_report(RUN_NAME, run_id, done, len(model_seeds) - done, {})
    return {"run_id": run_id, "seeds_done": done}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=RUN_NAME)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(MODEL_SEEDS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-random", type=int, default=N_RANDOM_CONTROLS)
    ap.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    args = ap.parse_args()
    main(tuple(args.seeds), args.device, args.n_random,
         tuple(args.budgets))
