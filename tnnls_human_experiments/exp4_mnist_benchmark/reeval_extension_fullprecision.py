"""Round-4 closure of the stored hidden-MI precision issue (exp4.md
§E4.14.2, §E4.15 decision #29 / follow-up of #24).

1. Deterministically refit all 300 output-aware fit conditions (stored fit
   seeds); save the FULL-PRECISION hidden-MI vectors to
   `results/factor_hidden_mi_full_precision.npz`.
2. Verify the regenerated PRIMARY supports (q in {0.6,0.7,0.8,0.9}, built
   at full precision like the original fit stage) match the stored primary
   supports exactly.
3. Identify extension (q in {0.95, 0.99}) supports that differ from the
   6-decimal-rounded reconstruction, and re-evaluate exactly those
   intervention families (1 LC + 20 random + 1 univariate each) with the
   frozen evaluator into a SEPARATE store table `extension_summary_fp`,
   never touching the committed extension shards.
4. Write `results/extension_fullprecision_reeval.csv` comparing old vs new
   per-intervention metrics.
"""
from __future__ import annotations

import glob
import json
import sys
import numpy as np
import pandas as pd
import torch

HERE = __import__("os").path.dirname(__file__)
RAW = HERE + "/run_results/raw/mnist"
RES = HERE + "/results"
sys.path.insert(0, HERE)
from benchmark import (iter_cluster_layer_contexts, mapped_supports,  # noqa
                       fit_and_rank, run_ablation_stage_for_factor,
                       TrunkEvaluator, EXPERIMENT)
from common import seeds as S
from common import store as store_mod
from common.config import RAW_DIR
from data import load_splits
from train import load_model

PRIMARY_THRESHOLDS = (0.6, 0.7, 0.8, 0.9)
EXTENSION_THRESHOLDS = (0.95, 0.99)


def main() -> None:
    store = store_mod.ResultStore(EXPERIMENT)
    splits = load_splits(save_dir=RAW_DIR / EXPERIMENT)
    labels_cluster = pd.read_parquet(store.shard_path(
        {"stage": "clusters", "phate_seed": 5200, "kmeans_seed": 5201,
         "k": 20}, "clusters")).set_index("discovery_index")["cluster"] \
        .sort_index().to_numpy()
    labels_test = splits["test_y"].numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = {"n_random_controls": 20, "support_thresholds": []}

    # stored factor rows for comparison
    stored_f = {}
    for f in sorted(glob.glob(RAW + "/*.factors.parquet")):
        d = pd.read_parquet(f)
        for _, r in d[d.input_condition == "output_aware"].iterrows():
            stored_f[(int(r.model_seed), int(r.cluster), int(r.layer),
                      int(r.factor_index))] = (json.loads(r["hidden_mi_vector"]),
                                               int(r["factor_rank"]))
    # stored primary supports (fit-stage, full precision): mi_rank-0 rows
    # carry the first unit + sizes; reconstruct set membership from rows
    stored_sup = {}
    for f in sorted(glob.glob(RAW + "/*.supports.parquet")):
        d = pd.read_parquet(f,
                            columns=["input_condition", "threshold",
                                     "model_seed", "cluster", "layer",
                                     "factor_index", "mi_rank",
                                     "hidden_unit"])
        for (ms, cl, ly, fi), g in d[d.input_condition == "output_aware"] \
                .groupby(["model_seed", "cluster", "layer", "factor_index"]):
            for q, gg in g.groupby("threshold"):
                stored_sup.setdefault((int(ms), int(cl), int(ly), int(fi)),
                                      {})[float(q)] = set(gg.hidden_unit)

    vecs = {}
    sup_diff = []          # (seed, cluster, layer, factor, q) full != rounded
    primary_mismatch = 0
    n_cond = n_fac = 0
    plan = []              # families to re-evaluate
    for seed in (5100, 5101, 5102, 5103, 5104):
        model = load_model(seed, device=device)
        trunk = TrunkEvaluator(model, splits["test_x"], device)
        for cluster, layer, ctx in iter_cluster_layer_contexts(
                model, splits, labels_cluster,
                {"n_clusters": 20, "layers": [1, 2, 3],
                 "support_thresholds": list(PRIMARY_THRESHOLDS)}, device):
            n_cond += 1
            s_fit = int(S.derive_seed(seed, "corex", cluster, layer,
                                      "output_aware"))
            fr = fit_and_rank(ctx.H_std, ctx.L_std, 10, s_fit, 5,
                              valid_h=ctx.valid_h)
            s_tie = int(S.derive_seed(s_fit, "support_ties"))
            for rank, j in enumerate(fr["order"]):
                n_fac += 1
                key = (seed, cluster, layer, int(j))
                hmi_full = fr["hidden_mis"][rank]
                vecs[f"{seed}:{cluster}:{layer}:{int(j)}"] = \
                    hmi_full.astype(np.float64)
                # 2. primary supports (full precision, as the fit stage
                #    built them) must match the stored ones exactly
                prim = mapped_supports(hmi_full, list(PRIMARY_THRESHOLDS),
                                       ctx.valid_h, s_tie)
                stored = stored_sup.get(key, {})
                for q, sup in zip(PRIMARY_THRESHOLDS, prim):
                    if stored.get(q) != set(int(i) for i in sup.indices):
                        primary_mismatch += 1
                        print(f"  PRIMARY MISMATCH {key} q={q}")
                # 3. extension supports vs the rounded reconstruction
                sups_ext_full = mapped_supports(
                    hmi_full, list(EXTENSION_THRESHOLDS), ctx.valid_h, s_tie)
                sups_ext_rnd = mapped_supports(
                    np.round(hmi_full, 6), list(EXTENSION_THRESHOLDS),
                    ctx.valid_h, s_tie)
                for q, sfull, srnd in zip(EXTENSION_THRESHOLDS,
                                          sups_ext_full, sups_ext_rnd):
                    if set(sfull.indices) != set(srnd.indices):
                        sup_diff.append((*key, q, len(sfull.indices),
                                         len(srnd.indices)))
                        plan.append((seed, cluster, layer, rank, j, q,
                                     [int(i) for i in sfull.indices],
                                     ctx, trunk, int(fr["dominant"][rank])))
    print(f"refit: conditions={n_cond} factors={n_fac}; "
          f"primary-support mismatches: {primary_mismatch}; "
          f"extension supports differing: {len(sup_diff)}")
    np.savez_compressed(RES + "/factor_hidden_mi_full_precision.npz", **vecs)
    print("saved full-precision vectors:",
          RES + "/factor_hidden_mi_full_precision.npz", f"({len(vecs)} arrays)")

    # 4. re-evaluate the affected families (frozen evaluator, separate table)
    if not plan:
        print("no extension supports to re-evaluate")
        return
    old = pd.concat([pd.read_parquet(f) for f in
                     sorted(glob.glob(RAW + "/*.extension_summary.parquet"))],
                    ignore_index=True)
    cmp_rows = []
    key = ["model_seed", "cluster", "layer", "factor_index", "threshold",
           "method", "random_draw"]
    for (seed, cluster, layer, rank, j, q, sup, ctx, trunk,
         dominant) in plan:
        cond = {"stage": "extension", "model_seed": seed,
                "cluster": cluster, "layer": layer, "factor_index": int(j),
                "input_condition": "output_aware"}
        if not store.is_done(cond, ("extension_summary_fp",)):
            cfgq = {**cfg, "support_thresholds": [q]}
            run_ablation_stage_for_factor(
                seed, cluster, layer, "output_aware", int(j), rank,
                {q: sup}, ctx.H_std, ctx.logits, ctx.valid_h,
                int(dominant), trunk, labels_test, store, cfgq,
                table_name="extension_summary_fp")
        fp = pd.read_parquet(store.shard_path(cond, "extension_summary_fp"))
        sel = old[(old.model_seed == seed) & (old.cluster == cluster)
                  & (old.layer == layer) & (old.factor_index == j)
                  & (old.threshold == q)
                  & (old.input_condition == "output_aware")]
        oldr = sel.set_index(["method", "random_draw"])
        for _, nr in fp.iterrows():
            kk = (nr["method"], nr["random_draw"])
            if kk in oldr.index:
                orow = oldr.loc[kk]
                cmp_rows.append({
                    "model_seed": seed, "cluster": cluster, "layer": layer,
                    "factor_index": int(j), "threshold": q,
                    "method": nr["method"], "random_draw": nr["random_draw"],
                    "support_size_old": orow["support_size"],
                    "support_size_new": nr["support_size"],
                    "T_max_old": orow["T_max"], "T_max_new": nr["T_max"],
                    "T_max_d": nr["T_max"] - orow["T_max"],
                    "H_input_old": orow["H_input"],
                    "H_input_new": nr["H_input"],
                    "H_input_d": nr["H_input"] - orow["H_input"],
                })
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(RES + "/extension_fullprecision_reeval.csv", index=False)
    lc = cmp[cmp.method == "local_corex"]
    print(cmp.groupby("method")[["T_max_d", "H_input_d"]]
          .agg(["mean", "max", "min"]).round(4).to_string())
    print("rewritten LC rows:", len(lc), "| pairs:", len(cmp))
    print("saved comparison:", RES + "/extension_fullprecision_reeval.csv")


if __name__ == "__main__":
    main()