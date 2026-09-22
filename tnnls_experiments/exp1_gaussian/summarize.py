"""Experiment 1 summarizer: reads raw shards, never reruns models.

Produces outputs/summaries/gaussian_summary.csv with replicate-level means,
SEs, CIs, and paired contrasts vs global and oracle-local.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.config import SUMMARIES_DIR
from ..common.metrics import paired_difference_summary, summarize_replicates
from ..common.store import ResultStore

EXPERIMENT = "gaussian"


def replicate_aggregate(matches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate factor-level matches within region, then within replicate."""
    # within (replicate, method, draw, region): mean over the 5 matched factors
    keys = ["root_seed", "regime", "n_per_region", "alpha", "n_nuisance",
            "partition_method", "partition_draw", "region"]
    metric_cols = [c for c in matches.columns
                   if c.startswith(("cosine", "auprc", "top4", "nfp",
                                    "support_", "nfp_support"))]
    agg = matches.groupby(keys, dropna=False)[metric_cols].mean().reset_index()
    # within replicate: mean over regions and draws (random partitions only:
    # each draw is a control replicate, so keep draws separate for random)
    keys2 = ["root_seed", "regime", "n_per_region", "alpha", "n_nuisance",
             "partition_method", "region"]
    return agg.groupby(keys2, dropna=False)[metric_cols].mean().reset_index()


def paired_contrasts(agg: pd.DataFrame) -> pd.DataFrame:
    """Paired differences of each method vs global and vs oracle, within
    condition (regime, n, alpha, p_n, region). Replicates are root seeds."""
    rows = []
    for (regime, n, alpha, p_n, region), grp in agg.groupby(
        ["regime", "n_per_region", "alpha", "n_nuisance", "region"]
    ):
        for method in grp["partition_method"].unique():
            if method in ("global", "oracle"):
                continue
            sub = grp[grp["partition_method"] == method].set_index("root_seed")
            glb = grp[grp["partition_method"] == "global"].set_index("root_seed")
            orc = grp[grp["partition_method"] == "oracle"].set_index("root_seed")
            common_g = sub.index.intersection(glb.index)
            common_o = sub.index.intersection(orc.index)
            for metric in ("auprc", "top4_f1", "cosine"):
                if metric not in sub.columns:
                    continue
                x = sub.loc[common_g, metric].to_numpy(float)
                y = glb.loc[common_g, metric].to_numpy(float)
                s = paired_difference_summary(x, y)
                rows.append({"regime": regime, "n_per_region": n, "alpha": alpha,
                             "n_nuisance": p_n, "region": region,
                             "method": method, "metric": metric,
                             "vs": "global", **s})
                x = sub.loc[common_o, metric].to_numpy(float)
                y = orc.loc[common_o, metric].to_numpy(float)
                s = paired_difference_summary(x, y)
                rows.append({"regime": regime, "n_per_region": n, "alpha": alpha,
                             "n_nuisance": p_n, "region": region,
                             "method": method, "metric": metric,
                             "vs": "oracle", **s})
    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    store = ResultStore(EXPERIMENT)
    matches = store.read_all("matches")
    if matches.empty:
        raise SystemExit("No matches found; run the experiment first.")
    agg = replicate_aggregate(matches)
    contrasts = paired_contrasts(agg)
    agg.to_csv(SUMMARIES_DIR / "gaussian_summary.csv", index=False)
    contrasts.to_csv(SUMMARIES_DIR / "gaussian_contrasts.csv", index=False)
    print(f"aggregate rows: {len(agg)}; contrast rows: {len(contrasts)}")
    return agg


if __name__ == "__main__":
    main()