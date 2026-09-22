"""Experiment 4 summarizer: reads raw shards, never reruns models.

Primary summaries (per spec):
  1. Fraction of LC interventions that are substantial and class-concentrated,
     by layer and support threshold.
  2. Paired LC vs random differences in T_max and H_input.
  3. Paired LC vs univariate differences.
  4. Support density by layer and threshold.
  5. TC / density / magnitude relationships (saved as a tidy table).
  6. Directional concordance rate by layer.
  7. Output-aware vs hidden-only comparison at q=0.8.

Uncertainty: model seed is the primary unit; a hierarchical bootstrap
(resample seeds, then clusters within seeds) is used for factor-level
distributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.config import SUMMARIES_DIR
from ..common.store import ResultStore

EXPERIMENT = "mnist"


def prevalence_by_layer_threshold(abl: pd.DataFrame) -> pd.DataFrame:
    """Summary 1: class-concentrated prevalence by layer x threshold.

    The replicate is the model seed: prevalence is computed per seed, then
    summarized across seeds (mean, sd, se, CI).
    """
    lc = abl[(abl["method"] == "local_corex")
             & (abl["input_condition"] == "output_aware")]
    rows = []
    for (layer, q), grp in lc.groupby(["layer", "threshold"]):
        per_seed = grp.groupby("model_seed")["substantial_class_concentrated"].mean()
        n = len(per_seed)
        mean = per_seed.mean()
        se = per_seed.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        rows.append({"layer": layer, "threshold": q,
                     "prevalence": float(mean), "se": float(se),
                     "n_seeds": n, "n_interventions": len(grp)})
    return pd.DataFrame(rows)


def paired_lc_vs_control(abl: pd.DataFrame, control: str) -> pd.DataFrame:
    """Paired differences LC - control on (T_max, T_weighted, H_input, D_input)
    at matched (seed, cluster, layer, factor, threshold)."""
    lc = abl[abl["method"] == "local_corex"]
    ct = abl[abl["method"] == control]
    key = ["model_seed", "cluster", "layer", "input_condition",
           "factor_index", "threshold"]
    merged = lc.merge(ct, on=key, suffixes=("_lc", f"_{control}"))
    rows = []
    for metric in ("T_max", "T_weighted", "H_input", "D_input"):
        for layer in sorted(merged["layer"].unique()):
            sub = merged[merged["layer"] == layer]
            d = (sub[f"{metric}_lc"] - sub[f"{metric}_{control}"]).to_numpy(float)
            d = d[np.isfinite(d)]
            # seed-level aggregation: the replicate is the model seed
            per_seed = sub.groupby("model_seed").apply(
                lambda g: (g[f"{metric}_lc"] - g[f"{metric}_{control}"]).mean()
            ).to_numpy(float)
            n = len(per_seed)
            if n == 0:
                continue
            mean = float(per_seed.mean())
            se = float(per_seed.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            rows.append({
                "metric": metric, "layer": layer, "control": control,
                "mean_diff": mean, "se": se, "n_pairs": len(d), "n_seeds": n,
            })
    return pd.DataFrame(rows)


def support_density(abl: pd.DataFrame, sup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lc = sup.drop_duplicates(
        subset=["model_seed", "cluster", "layer", "input_condition",
                "factor_index", "threshold"])
    for (layer, q), grp in lc.groupby(["layer", "threshold"]):
        per_seed = grp.groupby("model_seed")["support_density"].mean()
        rows.append({"layer": layer, "threshold": q,
                     "mean_density": float(per_seed.mean()),
                     "se": float(per_seed.std(ddof=1) / np.sqrt(len(per_seed)))
                     if len(per_seed) > 1 else 0.0})
    return pd.DataFrame(rows)


def concordance_by_layer(abl: pd.DataFrame) -> pd.DataFrame:
    lc = abl[(abl["method"] == "local_corex")
             & (abl["input_condition"] == "output_aware")]
    rows = []
    for layer, grp in lc.groupby("layer"):
        per_seed = grp.groupby("model_seed")["concordant"].mean()
        rows.append({"layer": layer,
                     "concordance_rate": float(per_seed.mean()),
                     "se": float(per_seed.std(ddof=1) / np.sqrt(len(per_seed)))
                     if len(per_seed) > 1 else 0.0})
    return pd.DataFrame(rows)


def output_aware_vs_hidden_only(abl: pd.DataFrame) -> pd.DataFrame:
    """Summary 7: compare conditions at q=0.8, matched factor rank."""
    lc = abl[abl["method"] == "local_corex"]
    oa = lc[(lc["input_condition"] == "output_aware")
            & (lc["threshold"] == 0.8)]
    ho = lc[(lc["input_condition"] == "hidden_only")]
    key = ["model_seed", "cluster", "layer", "factor_rank"]
    merged = oa.merge(ho, on=key, suffixes=("_oa", "_ho"))
    rows = []
    for metric in ("T_max", "H_input"):
        for layer in sorted(merged["layer"].unique()):
            sub = merged[merged["layer"] == layer]
            d = (sub[f"{metric}_oa"] - sub[f"{metric}_ho"]).to_numpy(float)
            d = d[np.isfinite(d)]
            per_seed = sub.groupby("model_seed").apply(
                lambda g: (g[f"{metric}_oa"] - g[f"{metric}_ho"]).mean()
            ).to_numpy(float)
            rows.append({"metric": metric, "layer": layer,
                         "mean_diff": float(per_seed.mean()),
                         "se": float(per_seed.std(ddof=1) / np.sqrt(len(per_seed)))
                         if len(per_seed) > 1 else 0.0,
                         "n_pairs": len(d)})
    return pd.DataFrame(rows)


def hierarchical_bootstrap(abl: pd.DataFrame, metric: str, n_boot: int = 2000,
                          seed: int = 1) -> dict:
    """Bootstrap resampling model seeds, then clusters within seeds, for a
    factor-level mean."""
    rng = np.random.default_rng(seed)
    lc = abl[(abl["method"] == "local_corex")
             & (abl["input_condition"] == "output_aware")]
    seeds = lc["model_seed"].unique()
    stats = []
    for _ in range(n_boot):
        vals = []
        for s in rng.choice(seeds, size=len(seeds), replace=True):
            sub = lc[lc["model_seed"] == s]
            clusters = sub["cluster"].unique()
            chosen = rng.choice(clusters, size=len(clusters), replace=True)
            for c in chosen:
                g = sub[sub["cluster"] == c]
                if len(g):
                    vals.append(g[metric].mean())
        stats.append(np.mean(vals) if vals else np.nan)
    stats = np.asarray(stats, dtype=float)
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return {"mean": float(np.nanmean(stats)), "ci_low": float(lo),
            "ci_high": float(hi)}


def main() -> dict:
    store = ResultStore(EXPERIMENT)
    abl = store.read_all("ablation_summary")
    sup = store.read_all("supports")
    factors = store.read_all("factors")
    if abl.empty:
        raise SystemExit("No ablation shards; run the benchmark first.")

    out = {}
    out["prevalence"] = prevalence_by_layer_threshold(abl)
    out["lc_vs_random"] = paired_lc_vs_control(abl, "random")
    out["lc_vs_univariate"] = paired_lc_vs_control(abl, "univariate")
    out["support_density"] = support_density(abl, sup)
    out["concordance"] = concordance_by_layer(abl)
    out["oa_vs_ho"] = output_aware_vs_hidden_only(abl)
    # relationship table: TC, density, magnitude at factor level
    lc = abl[(abl["method"] == "local_corex")
             & (abl["input_condition"] == "output_aware")]
    dens = sup.drop_duplicates(
        subset=["model_seed", "cluster", "layer", "input_condition",
                "factor_index", "threshold"])[
        ["model_seed", "cluster", "layer", "input_condition", "factor_index",
         "threshold", "support_density", "support_size"]]
    rel = lc.merge(
        factors[["model_seed", "cluster", "layer", "input_condition",
                 "factor_index", "factor_tc"]],
        on=["model_seed", "cluster", "layer", "input_condition",
            "factor_index"]).merge(dens, on=[
        "model_seed", "cluster", "layer", "input_condition", "factor_index",
        "threshold"], suffixes=("", "_sup"))
    out["tc_density_magnitude"] = rel[[
        "model_seed", "cluster", "layer", "factor_index", "threshold",
        "factor_tc", "support_density", "support_size", "T_max", "H_input",
        "substantial_class_concentrated"]]

    for name, df in out.items():
        df.to_csv(SUMMARIES_DIR / f"mnist_{name}.csv", index=False)
    print({k: len(v) for k, v in out.items()})
    return out


if __name__ == "__main__":
    main()