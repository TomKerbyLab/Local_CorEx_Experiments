"""Compare the committed E5 `paired_lc_vs_control` convention (control
rows averaged over all thresholds) against a threshold-matched (q = 0.80)
pairing, regenerate the corrected tables, and quantify the change.

Run with the project venv from the repository root:
    .venv/bin/python paper/figures/check_e5_pairing_fix.py

Reads the E5 `ablation_summary` store (expects
tnnls_human_experiments/exp5_imagenet/run_results/raw/imagenet/*.ablation_summary.parquet)
and writes corrected + comparison tables into outputs/e5_pairing_check/.
"""

import glob
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
STORE = (ROOT / "tnnls_human_experiments/exp5_imagenet/run_results"
              "/raw/imagenet")
OUT = ROOT / "outputs/e5_pairing_check"
BOOT = 2000
SEED = 7100

STATS = ("tv_source", "localization_contrast", "mean_dp_dominant_class",
         "top5_set_change_rate", "H_input")


def boot_ci(v):
    v = pd.Series(v).dropna().to_numpy(float)
    rng = np.random.default_rng(SEED)
    means = np.array([v[rng.integers(0, len(v), len(v))].mean()
                      for _ in range(BOOT)])
    return {"mean": float(v.mean()),
            "ci_low": float(np.percentile(means, 2.5)),
            "ci_high": float(np.percentile(means, 97.5)),
            "se": float(pd.Series(v).std(ddof=1) / np.sqrt(len(v))),
            "n_clusters": int(len(v))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = ROOT / "outputs/summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_parquet(p) for p in
              sorted(glob.glob(str(STORE / "*.ablation_summary.parquet")))]
    abl = pd.concat(frames, ignore_index=True)

    lc = abl[(abl.method == "local_corex") & (abl.threshold == 0.8)
             & abl.tv_source.notna() & (abl.n_source >= 30)]
    print("LC systematic rows:", len(lc), "pairs:",
          lc.groupby(["cluster", "factor_index"]).ngroups,
          "clusters:", lc.cluster.nunique())

    # control-coverage verification (round-3 review): every eligible LC row
    # must have its expected controls at q = 0.80; missing matches are
    # reported, never skipped silently.
    EXPECTED = {"random": 5, "univariate": 1}
    ctl_q_all = {c: abl[(abl.method == c) & (abl.threshold == 0.8)]
                 for c in EXPECTED}
    missing_report = []
    for c, n_exp in EXPECTED.items():
        table = ctl_q_all[c].groupby(["cluster", "factor_index"]).size()
        n_with = table.reindex(
            pd.MultiIndex.from_frame(
                lc[["cluster", "factor_index"]]), fill_value=0)
        n_missing = int((n_with == 0).sum())
        wrong = int(((n_with > 0) & (n_with != n_exp)).sum())
        missing_report.append({"control": c, "expected_draws": n_exp,
                               "lc_rows": len(lc), "n_no_controls": n_missing,
                               "n_wrong_count": wrong,
                               "max_observed_draws": int(table.max())})
        print(f"control-coverage {c}: missing on {n_missing}/{len(lc)} "
              f"LC rows; wrong-count cells {wrong}: "
              f"max draws observed {int(table.max())}")
    pd.DataFrame(missing_report).to_csv(
        OUT / "control_coverage_check.csv", index=False)

    rows_old, rows_new = [], []
    for control in ("random", "univariate"):
        ctl_all = ctl_q_all[control]
        ctl_all_allq = abl[((abl.method == control))]
        ctl_q = ctl_q_all[control]
        for stat in STATS:
            for label, ctl in (("threshold-averaged (committed)", ctl_all_allq),
                               ("q=0.80 matched (fixed)", ctl_q)):
                d = []
                for _, row in lc.iterrows():
                    c = ctl[(ctl.cluster == row.cluster)
                            & (ctl.factor_index == row.factor_index)]
                    if len(c) == 0:
                        continue
                    d.append(row[stat] - c[stat].mean())
                agg = pd.Series(d, index=lc.index).groupby(lc.cluster).mean()
                ci = boot_ci(agg)
                row = {"control": control, "statistic": stat,
                       "convention": label, "n_pairs": len(d), **ci}
                if label.startswith("threshold-averaged"):
                    rows_old.append(row)
                else:
                    rows_new.append(row)
                print(f"{control:10s} {label:28s} {stat:24s} "
                      f"mean {ci['mean']:+.4f} CI [{ci['ci_low']:+.4f}, "
                      f"{ci['ci_high']:+.4f}] n {len(d)}")

    pd.DataFrame(rows_old).to_csv(OUT / "systematic_convention_committed.csv",
                                  index=False)
    pd.DataFrame(rows_new).to_csv(OUT / "systematic_convention_q080.csv",
                                  index=False)

    # canonical corrected tables (the source for Figure 8): same schema as
    # the committed imagenet_systematic_*.csv, computed with the fixed
    # q = 0.80 pairing. One file per control, all five statistics.
    for control in ("random", "univariate"):
        ctl_q = ctl_q_all[control]
        sys_rows = []
        for stat in STATS:
            d = []
            pair_index = []
            for _, row in lc.iterrows():
                c = ctl_q[(ctl_q.cluster == row.cluster)
                          & (ctl_q.factor_index == row.factor_index)]
                if len(c) == 0:
                    continue
                d.append(row[stat] - c[stat].mean())
                pair_index.append((row.cluster, row.factor_index))
            pidx = list(zip(*pair_index))
            agg = pd.Series(d, index=pidx[0]).groupby(level=0).mean() \
                if d else pd.Series(dtype=float)
            ci = boot_ci(agg)
            sys_rows.append({
                "control": control, "statistic": stat,
                "mean_diff": ci["mean"], "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"], "se": ci["se"],
                "n_clusters": ci["n_clusters"], "n_factor_pairs": len(d),
                "bootstrap": f"cluster-level, B={BOOT}",
                "git_commit": "pairing-fix",
            })
            df = pd.DataFrame({"cluster": pidx[0],
                               "factor_index": pidx[1], "diff": d})
            df.to_csv(summaries /
                      f"imagenet_pairdiff_lc_vs_{control}_q080_{stat}.csv",
                      index=False)
        pd.DataFrame(sys_rows).to_csv(
            summaries / f"imagenet_systematic_lc_vs_{control}_q080.csv",
            index=False)
    print("canonical corrected tables written to", summaries)
    print("comparison tables written:", OUT)


if __name__ == "__main__":
    main()
