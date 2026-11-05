"""Experiment 5 summarizer: reads raw shards, never reruns models.

Tidy tables (spec `imagenet_*` list + freeze F2-F6):
  imagenet_split, imagenet_clusters, imagenet_cluster_composition,
  imagenet_cluster_size_distribution, imagenet_factors, imagenet_supports,
  imagenet_ablation_summary, imagenet_top_probability_shifts
  (+ .parquet files with all methods / all clusters),
systematic survey (7 spec summaries with the F3/F4 adjustments):
  imagenet_systematic_lc_vs_random, imagenet_systematic_lc_vs_univariate
  (cluster-level paired deltas with cluster bootstrap CIs),
  imagenet_localization_contrasts (L_j for LC side by side with the mean L
  of its size-matched random controls), imagenet_tc_magnitude,
  imagenet_dominant_class_concordance, imagenet_coverage,
  imagenet_case_study (+ imagenet_case_study_secondary_clusters).

Aggregation unit for interval estimates = the discovery cluster (F4);
top-3 factors are averaged within cluster before resampling.
Factors from clusters with fewer than 30 in-cluster evaluation observations
are excluded from systematic tables and reported in coverage instead (F2).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from common import config
from common.config import SUMMARIES_DIR
from common.store import ResultStore

EXPERIMENT = "imagenet"
RUN_GIT = config.git_commit() or "unknown"
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 7100
MIN_EVAL_OBS_FOR_SYSTEMATIC = 30
PRIMARY_Q = 0.80


def _write_csv(df: pd.DataFrame, name: str) -> str:
    df.to_csv(SUMMARIES_DIR / f"{name}.csv", index=False)
    return str(SUMMARIES_DIR / f"{name}.csv")


# ---------------------------------------------------------------------------
# Raw tables
# ---------------------------------------------------------------------------

def split_table(store) -> pd.DataFrame:
    z = store.read_all("split").copy()
    integ = store.read_all("split_integrity")
    if len(integ):
        z["labels_sha256"] = str(integ["labels_sha256"].iloc[0])
    z["git_commit"] = RUN_GIT
    return z


def clusters_table(store) -> pd.DataFrame:
    return store.read_all("clusters")


def composition_table(store) -> pd.DataFrame:
    z = store.read_all("cluster_composition")
    sizes = store.read_all("cluster_sizes")
    if len(sizes):
        z = z.merge(sizes, on="cluster", how="outer")
    return z


def size_distribution(store) -> list[dict]:
    import clustering

    sizes = store.read_all("cluster_sizes")
    if len(sizes) == 0:
        return []
    report = clustering.size_distribution_report(sizes.to_dict("records"))
    rows = []
    for which, st in report.items():
        rows.append({"split": which, "stat": "min", "value": st["min"]})
        rows.append({"split": which, "stat": "max", "value": st["max"]})
        rows.append({"split": which, "stat": "mean", "value": st["mean"]})
        for p, v in st["deciles"].items():
            rows.append({"split": which, "stat": p, "value": v})
        if st.get("n_below_50") is not None:
            rows.append({"split": which, "stat": "n_clusters_below_50",
                         "value": st["n_below_50"]})
    return rows


def factors_table(store) -> pd.DataFrame:
    z = store.read_all("factors")
    z["git_commit"] = RUN_GIT
    return z


def supports_table(store) -> pd.DataFrame:
    main = store.read_all("supports")
    case = store.read_all("case_study_supports")
    if len(case):
        case = case[case["threshold"] != PRIMARY_Q]
        main = pd.concat([main, case], ignore_index=True)
    main["git_commit"] = RUN_GIT
    return main


def ablation_table(store) -> pd.DataFrame:
    return store.read_all("ablation_summary")


def cohort_meta(store) -> pd.DataFrame:
    return store.read_all("cohort_meta")


# ---------------------------------------------------------------------------
# Systematic survey
# ---------------------------------------------------------------------------

def systematic_factor_set(abl: pd.DataFrame) -> set:
    """(cluster, factor) cells with a valid LC row at the primary threshold
    whose cohort has >= MIN_EVAL_OBS_FOR_SYSTEMATIC source observations."""
    lc = abl[(abl["method"] == "local_corex")
             & (abl["threshold"] == PRIMARY_Q)
             & (_fit_failed_mask(abl))
             & abl["tv_source"].notna()
             & (abl["n_source"] >= MIN_EVAL_OBS_FOR_SYSTEMATIC)]
    return set(zip(lc["cluster"].astype(int), lc["factor_index"].astype(int)))


def _fit_failed_mask(abl: pd.DataFrame) -> pd.Series:
    """True where the cell's CorEx fit failed. Rows of the normal path carry
    no fit_failed field; a missing column means no failure."""
    if "fit_failed" in abl.columns:
        return ~abl["fit_failed"].fillna(False).astype(bool)
    return pd.Series(True, index=abl.index)


def lc_rows(abl: pd.DataFrame, systematic: set) -> pd.DataFrame:
    lc = abl[(abl["method"] == "local_corex")
             & (abl["threshold"] == PRIMARY_Q)
             & (_fit_failed_mask(abl))
             & abl["tv_source"].notna()].copy()
    mask = [(int(c), int(fi)) in systematic
            for c, fi in zip(lc["cluster"], lc["factor_index"])]
    return lc[mask]


def cluster_bootstrap_ci(per_cluster: pd.Series, b: int = BOOTSTRAP_B,
                         seed: int = BOOTSTRAP_SEED) -> dict:
    v = pd.Series(per_cluster).dropna().to_numpy(float)
    if len(v) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "se": float("nan"), "n_clusters": 0}
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, len(v), len(v))].mean()
                      for _ in range(b)])
    return {
        "mean": float(v.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
        "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"),
        "n_clusters": int(len(v)),
    }


def paired_lc_vs_control(abl: pd.DataFrame, control: str, systematic: set) -> list[dict]:
    """Paired LC - control at the cluster level (freeze F4): per-factor diff,
    averaged within cluster, bootstrap CI over clusters. The pairing uses the
    identical evaluation observations (same cohort) for every member."""
    lc = lc_rows(abl, systematic)
    ctl = abl[abl["method"] == control]
    out_rows = []
    for stat_col in ("tv_source", "localization_contrast",
                     "mean_dp_dominant_class", "top5_set_change_rate",
                     "H_input"):
        pairs = []
        for _, row in lc.iterrows():
            c = ctl[(ctl["cluster"] == row["cluster"])
                    & (ctl["factor_index"] == row["factor_index"])]
            if len(c) == 0:
                continue
            pairs.append({"cluster": int(row["cluster"]),
                          "factor_index": int(row["factor_index"]),
                          "lc": float(row[stat_col]),
                          "ctl": float(c[stat_col].mean()),
                          "diff": float(row[stat_col] - c[stat_col].mean())})
        df = pd.DataFrame(pairs)
        agg = df.groupby("cluster")["diff"].mean()
        ci = cluster_bootstrap_ci(agg)
        out_rows.append({
            "control": control, "statistic": stat_col,
            "mean_diff": ci["mean"], "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"], "se": ci["se"],
            "n_clusters": ci["n_clusters"], "n_factor_pairs": len(df),
            "bootstrap": f"cluster-level, B={BOOTSTRAP_B}",
            "git_commit": RUN_GIT,
        })
        if len(df):
            df.to_csv(SUMMARIES_DIR /
                      f"imagenet_pairdiff_lc_vs_{control}_per_{stat_col}.csv",
                      index=False)
    return out_rows


def localization_contrasts(abl: pd.DataFrame, systematic: set) -> pd.DataFrame:
    """L_j distribution: LC rows side by side with the mean L of their
    size-matched random controls (F3)."""
    lc = lc_rows(abl, systematic)
    ctl = abl[(abl["method"] == "random") & abl["tv_source"].notna()]
    ctl_mean = (ctl.groupby(["cluster", "factor_index"], as_index=False)
                .agg(localization_contrast=("localization_contrast", "mean"),
                     support_size_mean=("support_size", "mean")))
    ctl_mean["method"] = "random_mean_of_5"
    cols = ["cluster", "factor_index", "method", "localization_contrast",
            "support_size_mean"]
    lc2 = lc[["cluster", "factor_index", "localization_contrast",
              "support_size"]].rename(
        columns={"support_size": "support_size_mean"}).copy()
    lc2["method"] = "local_corex"
    out = pd.concat([lc2[cols], ctl_mean[cols]], ignore_index=True)
    out["git_commit"] = RUN_GIT
    return out


def tc_magnitude(abl: pd.DataFrame, factors: pd.DataFrame,
                 systematic: set) -> pd.DataFrame:
    lc = lc_rows(abl, systematic)
    fcols = factors[["cluster", "factor_index", "factor_tc_nats",
                     "unit_mi_total_nats"]]
    m = lc.merge(fcols.drop_duplicates(["cluster", "factor_index"]),
                 on=["cluster", "factor_index"], how="left")
    rows = []
    for level, d, xcols in (
            ("factor", m, ("factor_tc_nats", "unit_mi_total_nats")),
            ("cluster_mean",
             m.groupby("cluster")[["factor_tc_nats", "tv_source"]]
             .mean().reset_index(), ("factor_tc_nats",))):
        for xcol in xcols:
            rho, p = stats.spearmanr(d[xcol], d["tv_source"],
                                     nan_policy="omit")
            rows.append({"level": level, "x": xcol, "y": "tv_source",
                         "spearman_rho": float(rho), "p_value": float(p),
                         "n": len(d)})
    return pd.DataFrame(rows).assign(git_commit=RUN_GIT)


def concordance(abl: pd.DataFrame, systematic: set) -> pd.DataFrame:
    """Fraction of factors whose c*_j is among the 5 most-decreased output
    classes for the most-affected true class (LC at primary threshold)."""
    lc = lc_rows(abl, systematic)
    if not len(lc):
        return pd.DataFrame()
    lc = lc.assign(val=lc["dominant_in_top5_decreased"].astype(float))
    per_cluster = lc.groupby("cluster")["val"].mean()
    ci = cluster_bootstrap_ci(per_cluster)
    return pd.DataFrame([{
        "statistic": "frac_dominant_among_top5_decreased",
        "mean": ci["mean"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
        "n_clusters": ci["n_clusters"], "n_factors": int(len(lc)),
        "git_commit": RUN_GIT}])


def coverage(store, abl: pd.DataFrame, cohorts: pd.DataFrame,
             factors: pd.DataFrame) -> list[dict]:
    """F2 failure-as-data rows."""
    rows = []
    if len(factors) and "fit_failed" in factors.columns:
        failed = factors[factors["fit_failed"].astype(bool)]
        for _, r in failed.iterrows():
            rows.append({"kind": "corex_fit_failed_or_skipped",
                         "cluster": int(r["cluster"]), "factor_index": -1,
                         "detail": str(r.get("failure_reason", ""))})
    if len(cohorts):
        cohort = cohorts.groupby("cluster")["n_source"].first()
        for k, n in cohort.items():
            if n < 50:
                rows.append({"kind": "low_sample_cluster", "cluster": int(k),
                             "factor_index": -1,
                             "detail": f"n_source={int(n)} < 50"})
    if len(abl):
        for _, r in abl[(abl["method"] == "local_corex")
                        & (abl["threshold"] == PRIMARY_Q)].iterrows():
            if not r.get("fit_failed", False) and \
                    r["tv_source"] is not None and pd.notna(
                    r["tv_source"]) and int(r["n_source"]) < \
                    MIN_EVAL_OBS_FOR_SYSTEMATIC:
                rows.append({"kind": "systematic_excluded_lt30_eval_obs",
                             "cluster": int(r["cluster"]),
                             "factor_index": int(r["factor_index"]),
                             "detail": f"n_source={int(r['n_source'])}"})
            elif pd.notna(r.get("fit_failed")) and r.get("fit_failed"):
                rows.append({"kind": "ablation_cell_fit_failed",
                             "cluster": int(r["cluster"]),
                             "factor_index": int(r["factor_index"]),
                             "detail": str(r.get("failure_reason", ""))})
        fitted = set(zip(factors["cluster"].astype(int) if len(factors) else [],
                         factors["factor_index"].astype(int))) \
            if len(factors) else set()
        valid = set(zip(
            abl[(abl["method"] == "local_corex") & (abl["threshold"] == PRIMARY_Q)
                & abl["tv_source"].notna()]["cluster"].astype(int),
            abl[(abl["method"] == "local_corex") & (abl["threshold"] == PRIMARY_Q)
                & abl["tv_source"].notna()]["factor_index"].astype(int))) \
            if len(abl) else set()
        for k, fi in sorted(fitted - valid):
            rows.append({"kind": "fit_row_without_valid_ablation", "cluster": k,
                         "factor_index": fi, "detail": ""})
    return rows


def case_study_tables(store):
    sel = store.read_all("case_study_selection")
    fac = store.read_all("case_study_factors")
    sup = store.read_all("case_study_supports")
    try:
        sec = store.read_all("case_study_secondary_clusters")
    except Exception:
        sec = pd.DataFrame()
    return sel, fac, sup, sec


def main() -> dict:
    store = ResultStore(EXPERIMENT)
    abl_all = ablation_table(store)
    factors = factors_table(store)
    cohorts = cohort_meta(store)
    systematic = systematic_factor_set(abl_all)

    out = {}
    out["imagenet_split"] = _write_csv(split_table(store), "imagenet_split")
    out["imagenet_clusters"] = _write_csv(clusters_table(store),
                                          "imagenet_clusters")
    out["imagenet_cluster_composition"] = _write_csv(
        composition_table(store), "imagenet_cluster_composition")
    out["imagenet_cluster_size_distribution"] = _write_csv(
        pd.DataFrame(size_distribution(store)),
        "imagenet_cluster_size_distribution")
    out["imagenet_factors"] = _write_csv(factors, "imagenet_factors")
    out["imagenet_supports"] = _write_csv(supports_table(store),
                                          "imagenet_supports")

    abl = abl_all.copy()
    abl["git_commit"] = RUN_GIT
    out["imagenet_ablation_summary"] = _write_csv(
        abl, "imagenet_ablation_summary")

    shifts = store.read_all("top_shifts")
    if len(shifts):
        sel = store.read_all("case_study_selection")
        case_cluster = int(sel["selected_cluster"].iloc[0]) if len(sel) else -1
        subset = shifts[(shifts["method"] == "local_corex")
                        | (shifts["cluster"] == case_cluster)].copy()
        subset["git_commit"] = RUN_GIT
        subset.to_parquet(
            SUMMARIES_DIR / "imagenet_top_probability_shifts_full.parquet",
            index=False)
        shifts.assign(git_commit=RUN_GIT).to_parquet(
            SUMMARIES_DIR / "imagenet_top_probability_shifts_all.parquet",
            index=False)
        out["imagenet_top_probability_shifts"] = _write_csv(
            subset, "imagenet_top_probability_shifts")
        out["imagenet_top_probability_shifts_all_parquet"] = str(
            SUMMARIES_DIR / "imagenet_top_probability_shifts_all.parquet")

    out["imagenet_systematic_lc_vs_random"] = _write_csv(
        pd.DataFrame(paired_lc_vs_control(abl, "random", systematic)),
        "imagenet_systematic_lc_vs_random")
    out["imagenet_systematic_lc_vs_univariate"] = _write_csv(
        pd.DataFrame(paired_lc_vs_control(abl, "univariate", systematic)),
        "imagenet_systematic_lc_vs_univariate")
    out["imagenet_localization_contrasts"] = _write_csv(
        localization_contrasts(abl, systematic),
        "imagenet_localization_contrasts")
    out["imagenet_tc_magnitude"] = _write_csv(
        tc_magnitude(abl, factors, systematic), "imagenet_tc_magnitude")
    out["imagenet_dominant_class_concordance"] = _write_csv(
        concordance(abl, systematic), "imagenet_dominant_class_concordance")
    cov = pd.DataFrame(coverage(store, abl, cohorts, factors))
    out["imagenet_coverage"] = _write_csv(cov, "imagenet_coverage")

    sel, fac, sup, sec = case_study_tables(store)
    if len(sel):
        case_cluster = int(sel["selected_cluster"].iloc[0])
        case_abl = abl[(abl["cluster"] == case_cluster)]
        parts = [x for x in (sel, fac, sup, case_abl) if len(x)]
        out["imagenet_case_study"] = _write_csv(
            pd.concat(parts, ignore_index=True), "imagenet_case_study")
        if len(sec):
            out["imagenet_case_study_secondary_clusters"] = _write_csv(
                sec, "imagenet_case_study_secondary_clusters")
    return out


if __name__ == "__main__":
    print(json.dumps(main(), default=str))
