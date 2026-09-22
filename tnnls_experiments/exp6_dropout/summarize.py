"""Experiment 6 summarizer: paired contrasts over seeds (reads raw shards).

Primary unit: the paired model seed. Outputs dropout_paired_contrasts with
mean, se, 95% CI per (metric, layer).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.config import SUMMARIES_DIR
from ..common.metrics import paired_difference_summary
from ..common.store import ResultStore

EXPERIMENT = "dropout"


def paired_contrasts() -> pd.DataFrame:
    store = ResultStore(EXPERIMENT)
    reps = store.read_all("rep_stats")
    rows = []
    metric_cols = ["gaussian_tc", "gaussian_tc_norm", "corex_explained_tc",
                   "cum_frac_top5"]
    for layer in sorted(reps["layer"].unique()):
        for scope in ("global", "weighted_local"):
            sub = reps[(reps["layer"] == layer) & (reps["scope"] == scope)]
            wide = sub.pivot_table(index="root_seed", columns="condition",
                                   values=metric_cols)
            for metric in metric_cols:
                if (metric, "dropout") not in wide.columns:
                    continue
                x = wide[(metric, "dropout")].to_numpy(float)
                y = wide[(metric, "no_dropout")].to_numpy(float)
                s = paired_difference_summary(x, y)
                rows.append({
                    "layer": layer, "scope": scope, "metric": metric,
                    "dropout_mean": float(np.nanmean(x)),
                    "no_dropout_mean": float(np.nanmean(y)),
                    **s,
                })
    return pd.DataFrame(rows)


def support_density_contrasts() -> pd.DataFrame:
    store = ResultStore(EXPERIMENT)
    sup = store.read_all("supports")
    if sup.empty:
        return pd.DataFrame()
    rows = []
    # pair by (seed, layer, cluster, factor rank, threshold)
    key = ["root_seed", "layer", "cluster", "factor_rank", "threshold"]
    wide = sup.pivot_table(index=key, columns="condition",
                          values="support_density").reset_index()
    for (layer, rank, q), grp in wide.groupby(["layer", "factor_rank",
                                               "threshold"]):
        x = grp["dropout"].to_numpy(float)
        y = grp["no_dropout"].to_numpy(float)
        n = min(len(x), len(y))
        s = paired_difference_summary(x[:n], y[:n])
        rows.append({"layer": layer, "factor_rank": rank, "threshold": q,
                     "dropout_mean": float(np.nanmean(x)),
                     "no_dropout_mean": float(np.nanmean(y)), **s})
    return pd.DataFrame(rows)


def functional_contrasts() -> pd.DataFrame:
    store = ResultStore(EXPERIMENT)
    fa = store.read_all("functional_ablation")
    if fa.empty:
        return pd.DataFrame()
    lc = fa[fa["method"] == "local_corex"]
    key = ["root_seed", "layer", "cluster", "factor_rank"]
    rows = []
    for metric in ("T_max", "excess_effect_E", "log_effect_ratio_Q"):
        wide = lc.pivot_table(index=key, columns="condition", values=metric)
        if wide.empty or "dropout" not in wide.columns:
            continue
        for layer in wide.index.get_level_values("layer").unique():
            sub = wide.xs(layer, level="layer")
            x = sub["dropout"].to_numpy(float)
            y = sub["no_dropout"].to_numpy(float)
            n = min(len(x), len(y))
            s = paired_difference_summary(x[:n], y[:n])
            rows.append({"layer": layer, "metric": metric,
                         "dropout_mean": float(np.nanmean(x)),
                         "no_dropout_mean": float(np.nanmean(y)), **s})
    return pd.DataFrame(rows)


def main() -> dict:
    pc = paired_contrasts()
    sc = support_density_contrasts()
    fc = functional_contrasts()
    pc.to_csv(SUMMARIES_DIR / "dropout_paired_contrasts.csv", index=False)
    sc.to_csv(SUMMARIES_DIR / "dropout_support_contrasts.csv", index=False)
    fc.to_csv(SUMMARIES_DIR / "dropout_functional_contrasts.csv", index=False)
    print(f"paired contrasts: {len(pc)}; support contrasts: {len(sc)}; "
          f"functional contrasts: {len(fc)}")
    return {"paired": pc, "supports": sc, "functional": fc}


if __name__ == "__main__":
    main()