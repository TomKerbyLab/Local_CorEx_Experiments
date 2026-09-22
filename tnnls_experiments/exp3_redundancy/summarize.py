"""Experiment 3 summarizer and figures (supplementary limitations panel)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..common.config import FIGURES_DIR, SUMMARIES_DIR
from ..common.metrics import summarize_replicates
from ..common.store import ResultStore

EXPERIMENT = "redundancy"


def main() -> dict:
    store = ResultStore(EXPERIMENT)
    runs = store.read_all("runs")
    if runs.empty:
        raise SystemExit("No runs; execute the experiment first.")

    # replicate-level aggregate: one row per condition (each replicate is a row)
    runs.to_csv(SUMMARIES_DIR / "redundancy_synergy_runs.csv", index=False)

    # summary by (generator, d, n)
    keys = ["generator", "d", "n"]
    metrics_cols = ["tc_plugin", "omega_plugin", "corex_factor_tc", "auprc",
                    "topd_f1", "support_f1", "nfp_support"]
    rows = []
    for keys_vals, grp in runs.groupby(keys):
        rec = dict(zip(keys, keys_vals))
        for m in metrics_cols:
            s = summarize_replicates(grp[m].to_numpy())
            rec.update({f"{m}_{k}": v for k, v in s.items()})
        rows.append(rec)
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARIES_DIR / "redundancy_synergy_summary.csv", index=False)

    # ---- compact limitations panel ----------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = {"redundant": "#1f77b4", "synergistic": "#d62728"}
    for gen, grp in runs.groupby("generator"):
        agg = grp.groupby("d")["corex_factor_tc"].agg(["mean", "sem"])
        axes[0].errorbar(agg.index, agg["mean"], yerr=agg["sem"], marker="o",
                         capsize=3, color=colors[gen], label=gen)
        agg = grp.groupby("d")["omega_plugin"].agg(["mean", "sem"])
        axes[1].errorbar(agg.index, agg["mean"], yerr=agg["sem"], marker="o",
                         capsize=3, color=colors[gen], label=gen)
        agg = grp.groupby("d")["topd_f1"].agg(["mean", "sem"])
        axes[2].errorbar(agg.index, agg["mean"], yerr=agg["sem"], marker="o",
                         capsize=3, color=colors[gen], label=gen)
    axes[0].set_xlabel("group size d"); axes[0].set_ylabel("CorEx factor TC (bits)")
    axes[1].set_xlabel("group size d"); axes[1].set_ylabel("O-information (bits)")
    axes[1].axhline(0, color="k", lw=0.8, alpha=0.5)
    axes[2].set_xlabel("group size d"); axes[2].set_ylabel("support F1 (top-d)")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("Exp 3: TC-matched redundancy vs synergy — CorEx finds both; "
                 "O-information separates them")
    fig.tight_layout()
    out = FIGURES_DIR / "redundancy_synergy_panel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    return {"runs": len(runs), "summary_rows": len(summary)}


if __name__ == "__main__":
    main()