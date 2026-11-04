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

Uncertainty: model seed is the primary unit (across-seed summaries with
small-sample intervals). A cluster-sampling bootstrap was removed in the
round-4 revision (exp4.md §E4.10): the seed × cluster design is crossed
(same clusters under every seed), and the nested resampler was unused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import SUMMARIES_DIR
from common.store import ResultStore

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
        for threshold in sorted(merged["threshold"].unique()):
            for layer in sorted(merged["layer"].unique()):
                sub = merged[(merged["layer"] == layer)
                             & (merged["threshold"] == threshold)]
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
                    "metric": metric, "layer": layer, "threshold": float(threshold),
                    "control": control,
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


def tc_magnitude_relationship(abl: pd.DataFrame,
                              factors: pd.DataFrame) -> pd.DataFrame:
    """Within each (layer, threshold) stratum: Spearman correlation between
    factor TC and T_max over Local CorEx interventions, and the same
    correlation after removing the (linear) dependence of T_max on support
    size — does the TC-magnitude link persist beyond 'more units selected'?
    (Empirical: nested supports guarantee only that support size grows
    monotonically in q, not the effect monotonicity.)
    """
    from scipy.stats import spearmanr

    lc = abl[(abl["method"] == "local_corex")
             & (abl["input_condition"] == "output_aware")].copy()
    fkey = ["model_seed", "cluster", "layer", "input_condition", "factor_index"]
    lc = lc.merge(factors[fkey + ["factor_tc"]], on=fkey, how="inner")
    # support_size travels on the ablation rows themselves (matched-key base)
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


def cutoff_sensitivity(abl: pd.DataFrame) -> pd.DataFrame:
    """Supplementary (predeclared): prevalence over the T_max x H_input
    cutoff grid {0.05, 0.10, 0.15} x {0.40, 0.50, 0.60}, by layer and
    threshold. Zero-effect rows (T_max = 0) fail every T cutoff and
    NaN-entropy rows fail every H cutoff, so no cutoff combination can
    resurrect a degenerate intervention."""
    lc = abl[abl["method"] == "local_corex"]
    rows = []
    for (layer, q), grp in lc.groupby(["layer", "threshold"]):
        per_seed = grp.groupby("model_seed").apply(
            lambda g, group_key=None: pd.Series({
                f"T{Tt:.2f}_H{Ht:.2f}": float(
                    ((g["T_max"] >= Tt) & (g["H_input"] <= Ht)).mean())
                for Tt in (0.05, 0.10, 0.15)
                for Ht in (0.40, 0.50, 0.60)})
        )
        for cutoff in per_seed.columns:
            rows.append({"layer": layer, "threshold": q, "cutoff": cutoff,
                         "prevalence": float(per_seed[cutoff].mean()),
                         "se": float(per_seed[cutoff].std(ddof=1)
                                      / np.sqrt(len(per_seed)))
                         if len(per_seed) > 1 else 0.0})
    return pd.DataFrame(rows)


def profile_cohort_summary(store, cluster_compositions: dict) -> pd.DataFrame:
    """Feedback round 2, Phase C input: per-factor held-out cohort effects.

    For every (seed, fit-cluster, layer, condition, factor): mean per-image
    magnitude a(x) = 0.5||p^A - p^0||_1 and flip rate on three cohorts
    defined from the profile grid + the fixed test-cluster assignment —
    (1) OWN: images assigned to the factor's discovery cluster (all
    classes); (2) SAME-DIGIT-OTHER: images of the discovery cluster's
    dominant digit assigned to other clusters; (3) OTHER-DIGITS: the rest.
    Averages are per image (mass never favors larger cohorts); cohort
    sizes are reported; paired differences vs the matched random-mean
    grid are computed cell-aligned before averaging (LC - random_mean on
    identical cells). Exploratory analysis (results were already
    examined).
    """
    prof = store.read_all("profiles")
    if prof.empty:
        return pd.DataFrame()
    dom_digit = {int(c): int(np.argmax(v)) for c, v in
                 cluster_compositions.items()}
    key = ["model_seed", "cluster", "layer", "input_condition", "factor_index"]
    rows = []
    lc = prof[prof.method == "local_corex"]
    rm = prof[prof.method == "random_mean"]
    merged = lc.merge(rm, on=key + ["eval_cluster", "true_class"],
                      suffixes=("_lc", "_rm"))
    for _, fgrp in merged.groupby(key):
        k = fgrp[key].iloc[0]
        fit_c = int(k["cluster"])
        dom_c = dom_digit[fit_c]
        cell = fgrp.eval_cluster
        cls = fgrp.true_class
        own = (cell == fit_c)
        sdo = (~own) & (cls == dom_c)
        oth = (~own) & (cls != dom_c)
        for name, mask in (("own_cluster", own),
                           ("same_digit_other_clusters", sdo),
                           ("other_digits", oth)):
            g = fgrp[mask]
            if g.empty:
                continue
            for stat, lc_col, rm_col in (("a_mean", "a_mean_lc", "a_mean_rm"),
                                         ("flip_rate", "flip_rate_lc",
                                          "flip_rate_rm")):
                n = g.n_images_lc
                rows.append({
                    **{kk: k[kk] for kk in key},
                    "cohort": name, "stat": stat,
                    "lc": float((g[lc_col] * n).sum() / n.sum()),
                    "random": float((g[rm_col] * n).sum() / n.sum()),
                    "diff": float(((g[lc_col] - g[rm_col]) * n).sum()
                                  / n.sum()),
                    "n_images": int(n.sum()),
                })
    return pd.DataFrame(rows)


def assoc_bits_column(factors: pd.DataFrame) -> pd.Series:
    """Association strength of each factor: its dominant-class logit MI in
    bits (NaN when the factor has no dominant logit)."""
    import json as _json
    import re as _re

    def _bits(row):
        d = row.get("dominant_logit")
        if d is None or d < 0:
            return np.nan
        c = f"logit_mi_{int(d)}"
        return float(row[c]) if c in row.index else np.nan

    return factors.apply(_bits, axis=1)


def profile_grid_summary(store) -> pd.DataFrame:
    """Feedback round 2, Phase C input: the per-cell profile grids,
    aggregated over the top-5 factors of each fit cluster and over seeds.

    One row per (layer, fit cluster, input condition, eval cluster, true
    class): n_images (identical across factors/seeds for a cell), and
    equal-weight factor-mean LC and random-mean a_mean plus their
    difference. (Top-5 equal-weight averaging is an exploratory summary
    choice; per-factor views live in the notebook.)
    """
    prof = store.read_all("profiles")
    if prof.empty:
        return pd.DataFrame()
    lc = prof[prof.method == "local_corex"].groupby(
        ["layer", "cluster", "input_condition", "eval_cluster", "true_class"]
    )[["a_mean", "n_images"]].mean().reset_index()
    rm = prof[prof.method == "random_mean"].groupby(
        ["layer", "cluster", "input_condition", "eval_cluster", "true_class"]
    )["a_mean"].mean().reset_index().rename(columns={"a_mean": "random_mean"})
    out = lc.merge(rm, on=["layer", "cluster", "input_condition",
                           "eval_cluster", "true_class"], how="left")
    out["diff"] = out["a_mean"] - out["random_mean"]
    out = out.rename(columns={"a_mean": "lc_mean"})
    # discovery-side association annotations per fit cluster (top-5 means):
    # dominant-class MI in bits and the gap to the second class
    factors = store.read_all("factors")
    if not factors.empty:
        fac = factors.copy()
        fac["assoc_bits"] = assoc_bits_column(fac)
        ann = fac.groupby(["layer", "input_condition", "cluster"]).agg(
            assoc_bits_mean=("assoc_bits", "mean"),
            gap_mean=("logit_mi_gap", "mean")).reset_index()
        out = out.merge(ann, on=["layer", "input_condition", "cluster"],
                        how="left")
    return out


def factor_assoc_table(factors: pd.DataFrame) -> pd.DataFrame:
    """Compact per-factor association/structure table (Phase D input):
    association strength (bits), gap, effective units, TC, source."""
    fac = factors.copy()
    fac["assoc_bits"] = assoc_bits_column(fac)
    cols = ["model_seed", "cluster", "layer", "input_condition",
            "factor_index", "factor_rank", "factor_tc", "dominant_logit",
            "logit_mi_gap", "effective_n_units", "logit_mi_source",
            "assoc_bits"]
    return fac[[c for c in cols if c in fac.columns]]


def coverage(abl: pd.DataFrame, factors: pd.DataFrame,
             cluster_sizes: dict | None = None) -> pd.DataFrame:
    """Coverage table: short factor retention and zero-effect interventions
    stay visible, so prevalence cannot improve because difficult cases
    disappeared. One row per (model_seed, layer, input_condition).
    With cluster_sizes, tiny-cluster exclusions (predeclared n < 100 rule)
    are named here rather than silently shrinking the grid."""
    lc = abl[(abl["method"] == "local_corex") if "method" in abl.columns
             else [False] * len(abl)]
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
    if not rows:
        return pd.DataFrame()
    cov = pd.DataFrame(rows)
    counts = n_factors.groupby(
        ["model_seed", "layer", "input_condition"])["n_factors_fit"].min()
    cov = cov.merge(
        counts.reset_index(), on=["model_seed", "layer", "input_condition"],
        how="left")
    if cluster_sizes:
        tiny = {c: n for c, n in cluster_sizes.items() if n < 100}
        cov.attrs["tiny_clusters_excluded"] = tiny
        cov.attrs["note"] = (
            f"tiny-cluster rule (<100 discovery obs): "
            f"excluded {sorted(tiny)}" if tiny
            else "no clusters excluded (all sizes >= 100)")
    return cov


def _discovery_cluster_sizes(store) -> dict | None:
    """Per-cluster discovery counts from the clusters shards; None when
    unavailable or when shards from multiple partitions (different k) are
    merged, in which case the size table would be ambiguous."""
    try:
        z = store.read_all("clusters")
    except Exception:
        return None
    if z.empty or z["discovery_index"].nunique() != 10_000:
        return None
    counts = z["cluster"].value_counts().sort_index()
    return {int(c): int(n) for c, n in counts.items()}




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
    out["tc_magnitude"] = tc_magnitude_relationship(abl, factors)
    out["coverage"] = coverage(
        abl, factors, cluster_sizes=_discovery_cluster_sizes(store))
    out["cutoff_sensitivity"] = cutoff_sensitivity(abl)
    try:
        clz = store.read_all("clusters")
        comp = {int(c): np.bincount(g.astype(int), minlength=10)
                for c, g in (clz.groupby("cluster")["true_label"]
                             if "true_label" in clz.columns else [])}
    except Exception:
        comp = {}
    if comp:
        out["profile_cohorts"] = profile_cohort_summary(store, comp)
        out["profile_grid"] = profile_grid_summary(store)
        out["factor_assoc"] = factor_assoc_table(factors)
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