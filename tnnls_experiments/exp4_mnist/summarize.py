"""Experiment 4 summarizer: reads raw shards, never reruns models.

Primary summaries (per spec, as pre-registered in paper/exp4.md):
  1. Fraction of LC interventions that are substantial and class-concentrated,
     by layer and support threshold.
  2. Paired LC vs random differences (T_max/U_max/H_input/flip_rate/...).
  3. Paired LC vs univariate differences.
  4. Support density by layer and threshold.
  5. TC / density / magnitude relationships (tidy table) + the Spearman
     TC-magnitude link and its support-size-adjusted version.
  6. Reduction-concordance rate by layer AND condition (descriptive; with the
     absolute shift at the associated class alongside).
  7. Output-aware vs hidden-only comparison at q=0.8 on excess-over-random,
     with support sizes reported.
  New coverage/effective-units tables: short factor retention, zero-effect
  interventions, no-silent-exclusion guarantees, effective unit counts.

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
    for metric in ("T_max", "T_weighted", "H_input", "D_input", "U_max",
                   "flip_rate"):
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
    """Reduction-concordance rate (associated class = most reduced class) by
    layer AND input condition. Descriptive only: MI association is
    sign-blind and masking is not a directed latent change, so a low rate
    is not by itself evidence against a factor — the absolute shift at the
    associated class is reported alongside (see
    associated_class_abs_shift)."""
    lc = abl[(abl["method"] == "local_corex")
             & abl["concordant"].notna()]
    rows = []
    for (layer, cond), grp in lc.groupby(["layer", "input_condition"]):
        per_seed = grp.groupby("model_seed")["concordant"].mean()
        rows.append({"layer": layer, "input_condition": cond,
                     "reduction_concordance_rate": float(per_seed.mean()),
                     "se": float(per_seed.std(ddof=1) / np.sqrt(len(per_seed)))
                     if len(per_seed) > 1 else 0.0,
                     "mean_abs_shift_at_associated_class": float(
                         grp["associated_class_abs_shift"].mean())})
    return pd.DataFrame(rows)


def output_aware_vs_hidden_only(abl: pd.DataFrame) -> pd.DataFrame:
    """Summary 7: compare conditions at q=0.8, matched factor rank.

    Fairness: the two conditions need not delete equal numbers of units,
    so each row reports support sizes for BOTH conditions AND each
    condition's excess effect over its own matched random controls.
    Differences are read on the excess over matched random, never on the
    raw effect difference.
    """
    lc = abl[abl["method"] == "local_corex"]
    rnd = abl[abl["method"] == "random"]
    rnd_mean = (rnd.groupby(
        ["model_seed", "cluster", "layer", "input_condition", "factor_index",
         "threshold"])[["T_max", "H_input"]].mean()
        .rename(columns={"T_max": "random_T_max",
                         "H_input": "random_H_input"}))
    exc = lc.merge(rnd_mean.reset_index(),
                   on=["model_seed", "cluster", "layer", "input_condition",
                       "factor_index", "threshold"])
    exc["excess_T_max"] = exc["T_max"] - exc["random_T_max"]
    exc["excess_H_input"] = exc["random_H_input"] - exc["H_input"]
    oa = exc[(exc["input_condition"] == "output_aware")
             & (exc["threshold"] == 0.8)]
    ho = exc[(exc["input_condition"] == "hidden_only")]
    key = ["model_seed", "cluster", "layer", "factor_rank"]
    merged = oa.merge(ho, on=key, suffixes=("_oa", "_ho"))
    rows = []
    for metric in ("excess_T_max", "excess_H_input"):
        for layer in sorted(merged["layer"].unique()):
            sub = merged[merged["layer"] == layer]
            d = (sub[f"{metric}_oa"] - sub[f"{metric}_ho"]).to_numpy(float)
            d = d[np.isfinite(d)]
            per_seed = sub.groupby("model_seed").apply(
                lambda g: (g[f"{metric}_oa"] - g[f"{metric}_ho"]).mean()
            ).to_numpy(float)
            rows.append({"metric": metric, "layer": layer,
                         "mean_support_size_oa": float(
                             sub["support_size_oa"].mean()),
                         "mean_support_size_ho": float(
                             sub["support_size_ho"].mean()),
                         "mean_diff": float(per_seed.mean()),
                         "se": float(per_seed.std(ddof=1) / np.sqrt(len(per_seed)))
                         if len(per_seed) > 1 else 0.0,
                         "n_pairs": len(d)})
    out = pd.DataFrame(rows)
    if len(out):
        out.attrs["note"] = ("excess over matched random controls per "
                             "condition; raw effect differences not used")
    return out


def effective_units_by_layer_threshold(factors: pd.DataFrame) -> pd.DataFrame:
    """Distribution of the effective number of units (entropy of normalized
    hidden-MI weights) by layer, condition, and factor rank: distinguishes a
    genuinely broad association from one dominant unit plus a weak tail."""
    return (factors.groupby(["layer", "input_condition", "factor_rank"])
            ["effective_n_units"].agg(["mean", "std", "count"]).reset_index())


def tc_magnitude_relationship(abl: pd.DataFrame, factors: pd.DataFrame,
                              sup: pd.DataFrame) -> pd.DataFrame:
    """Within each (layer, threshold) stratum: Spearman correlation between
    factor TC and T_max over Local CorEx interventions, and the same
    correlation after removing the (linear) dependence of T_max on support
    size — does the TC-magnitude link persist beyond 'more units selected'?
    (Empirical: nested supports guarantee only that support size grows
    monotonically in q, not the effect monotonicity.)

    factor TC reaches the intervention rows through merging the factors and
    supports tables by the factor key (the same merge as the
    tc_density_magnitude table).
    """
    from scipy.stats import spearmanr

    lc = abl[(abl["method"] == "local_corex")
             & (abl["input_condition"] == "output_aware")].copy()
    fkey = ["model_seed", "cluster", "layer", "input_condition", "factor_index"]
    lc = lc.merge(
        factors[fkey + ["factor_tc"]], on=fkey, how="inner")
    skey = fkey + ["threshold"]
    dens = sup.drop_duplicates(subset=skey)[
        skey + ["support_density", "support_size"]]
    lc = lc.merge(dens, on=skey, how="inner")
    rows = []
    for (layer, q), grp in lc.groupby(["layer", "threshold"]):
        ok = grp[np.isfinite(grp["factor_tc"]) & np.isfinite(grp["T_max"])]
        if len(ok) < 8 or ok["T_max"].std() == 0:
            continue
        rho_all = float(spearmanr(ok["factor_tc"], ok["T_max"]).statistic)
        size = ok["support_size"].to_numpy(float)
        tmax = ok["T_max"].to_numpy(float)
        resid = tmax - np.polyval(np.polyfit(size, tmax, 1), size)
        rho_resid = float(spearmanr(ok["factor_tc"], resid).statistic)
        rows.append({"layer": layer, "threshold": q, "n": len(ok),
                     "spearman_tc_Tmax": rho_all,
                     "spearman_tc_Tmax_resid_size": rho_resid})
    return pd.DataFrame(rows)


def coverage(abl: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Coverage table: short factor retention and zero-effect interventions
    stay visible, so prevalence cannot improve because difficult cases
    disappeared. One row per (model_seed, layer, input_condition); the port
    adds the excluded (tiny) clusters here."""
    lc = abl[(abl["method"] == "local_corex")]
    rows = []
    for k, grp in lc.groupby(["model_seed", "layer", "input_condition"]):
        rows.append({
            "model_seed": k[0], "layer": k[1], "input_condition": k[2],
            "n_lc_interventions": int(len(grp)),
            "n_clusters_present": int(grp["cluster"].nunique()),
            "n_zero_effect": int((grp["T_max"] <= 1e-11).sum()),
            "n_zero_effect_h_input_nan": int(
                (~np.isfinite(grp["H_input"]) & (grp["T_max"] <= 0)).sum()),
        })
    n_factors = factors.groupby(
        ["model_seed", "layer", "input_condition", "cluster"]
    ).size().rename("n_factors_fit").reset_index()
    if len(rows):
        cov = pd.DataFrame(rows)
        counts = n_factors.groupby(
            ["model_seed", "layer", "input_condition"])["n_factors_fit"].min()
        cov = cov.merge(counts.reset_index(),
                        on=["model_seed", "layer", "input_condition"],
                        how="left")
        return cov
    return pd.DataFrame()


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
    out["effective_units"] = effective_units_by_layer_threshold(factors)
    out["tc_magnitude"] = tc_magnitude_relationship(abl, factors, sup)
    out["coverage"] = coverage(abl, factors)
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