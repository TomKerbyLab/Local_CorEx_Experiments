"""Experiment 2 summarizer and figures (reads raw shards, never refits)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..common.config import FIGURES_DIR, SUMMARIES_DIR
from ..common.store import ResultStore

EXPERIMENT = "nonlinear"

METHOD_ORDER = ["global", "oracle", "phate", "raw_kmeans", "random"]


def aggregate(matches: pd.DataFrame) -> pd.DataFrame:
    keys = ["generator", "root_seed", "n_per_region", "alpha", "noise",
            "n_nuisance", "partition_method", "partition_draw", "region",
            "estimator", "corex_input"]
    metric_cols = ["cosine", "auprc", "topd_f1", "topd_precision",
                   "topd_recall", "factor_tc", "routing_fp", "nuisance_fp",
                   "support_f1", "support_size", "support_density"]
    # within (replicate, method, cell): mean over the 4 matched factors
    agg = matches.groupby(keys, dropna=False)[metric_cols].mean().reset_index()
    # within replicate: mean over regions and random draws
    keys2 = [k for k in keys if k not in ("partition_draw", "region")]
    return agg.groupby(keys2, dropna=False)[metric_cols].mean().reset_index()


def main() -> dict:
    store = ResultStore(EXPERIMENT)
    matches = store.read_all("matches")
    if matches.empty:
        raise SystemExit("No matches; run the experiment first.")
    agg = aggregate(matches)
    agg.to_csv(SUMMARIES_DIR / "nonlinear_summary.csv", index=False)

    # structure diagnostics table
    structure = store.read_all("structure")
    structure.to_csv(SUMMARIES_DIR / "nonlinear_structure_diagnostics.csv",
                     index=False)
    print(f"aggregate rows: {len(agg)}; diagnostics rows: {len(structure)}")

    # ---- figure: discrete and continuous rows -----------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    est_label = {"bio": "Bio CorEx", "linear": "Linear CorEx",
                 "bio_gaussian": "Bio (gaussian)"}

    for row, generator in enumerate(("parity", "sine_sum")):
        sub_gen = agg[(agg["generator"] == generator)
                      & (agg["estimator"] == "bio")
                      & (agg["n_per_region"] == 5000)]
        # panel 1: AUPRC by partition method and separation
        ax = axes[row, 0]
        for method in METHOD_ORDER:
            s = sub_gen[sub_gen["partition_method"] == method]
            g = s.groupby("alpha")["auprc"].agg(["mean", "sem"])
            if len(g):
                ax.errorbar(g.index, g["mean"], yerr=g["sem"], marker="o",
                            capsize=3, label=method)
        ax.set_xlabel("separation alpha")
        ax.set_ylabel("matched AUPRC")
        ax.set_title(f"{generator}: AUPRC by method (n=5000, Bio)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        # panel 2: AUPRC vs noise (oracle, bio vs linear)
        ax = axes[row, 1]
        for est in ("bio", "linear"):
            s = agg[(agg["generator"] == generator)
                    & (agg["partition_method"] == "oracle")
                    & (agg["estimator"] == est)
                    & (agg["n_per_region"] == 5000)
                    & (agg["alpha"] == 0.5)]
            g = s.groupby("noise")["auprc"].agg(["mean", "sem"])
            if len(g):
                ax.errorbar(g.index, g["mean"], yerr=g["sem"], marker="o",
                            capsize=3, label=est_label[est])
        ax.set_xlabel("noise")
        ax.set_ylabel("matched AUPRC")
        ax.set_title(f"{generator}: oracle-local Bio vs Linear", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

        # panel 3: Bio CorEx by separation for all methods, linear at alpha=0.5
        ax = axes[row, 2]
        s = agg[(agg["generator"] == generator)
                & (agg["estimator"] == "bio")
                & (agg["n_per_region"] == 5000)
                & (agg["alpha"] == 1.0)]
        for method in METHOD_ORDER:
            g = s[s["partition_method"] == method].groupby("noise")["auprc"].agg(["mean", "sem"])
            if len(g):
                ax.errorbar(g.index, g["mean"], yerr=g["sem"], marker="o",
                            capsize=3, label=method)
        ax.set_xlabel("noise")
        ax.set_title(f"{generator}: AUPRC vs noise (alpha=1)", fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Experiment 2: nonlinear and higher-order recovery")
    fig.tight_layout()
    out = FIGURES_DIR / "nonlinear_multipanel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    return {"summary": len(agg), "figure": str(out)}


if __name__ == "__main__":
    main()