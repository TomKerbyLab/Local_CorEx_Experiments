"""Experiment 1 figures: reads summary outputs, never raw CorEx fits.

Multipanel figure per spec:
  row 0: matched AUPRC vs sample size (columns = regime x alpha, p_n=25)
  row 1: matched AUPRC vs nuisance dimension (n=500, alpha=0.5)
  row 2: partition quality (ARI vs purity) by method
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ..common.config import FIGURES_DIR, SUMMARIES_DIR
from ..common.store import ResultStore

EXPERIMENT = "gaussian"

METHOD_COLORS = {
    "global": "#7f7f7f",
    "oracle": "#2ca02c",
    "phate": "#1f77b4",
    "random": "#d62728",
    "raw_kmeans": "#9467bd",
    "pca": "#8c564b",
}
METHOD_LABELS = {
    "global": "Global", "oracle": "Oracle", "phate": "PHATE",
    "random": "Random", "raw_kmeans": "Raw k-means", "pca": "PCA",
}
REGIMES = ["mostly_shared", "disjoint"]


def _errorbar_by(ax, sub, metric, xcol, color, label):
    g = sub.groupby(xcol)[metric].agg(["mean", "sem"])
    ax.errorbar(g.index, g["mean"], yerr=g["sem"], marker="o", capsize=3,
                color=color, label=label)


def _plot_by_n(ax, sub, metric, title):
    for method, grp in sub.groupby("partition_method"):
        if method == "pca":
            continue
        _errorbar_by(ax, grp, metric, "n_per_region",
                     METHOD_COLORS.get(method, "k"),
                     METHOD_LABELS.get(method, method))
    ax.set_xscale("log")
    ax.set_xlabel("n per region")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.3)


def main():
    agg = pd.read_csv(SUMMARIES_DIR / "gaussian_summary.csv")
    core = agg[agg["n_nuisance"] == 25]

    fig, axes = plt.subplots(3, 6, figsize=(20, 10), sharey="row")

    # Row 0: AUPRC vs n, columns = (regime, alpha)
    cols = [(r, a) for r in REGIMES for a in (0.0, 0.5, 1.0)]
    for i, (regime, alpha) in enumerate(cols):
        sub = core[(core["regime"] == regime) & (core["alpha"] == alpha)]
        if sub.empty:
            continue
        _plot_by_n(axes[0, i], sub, "auprc", f"{regime}, α={alpha}")
    axes[0, 0].set_ylabel("matched AUPRC")
    axes[0, 0].legend(fontsize=7)

    # Row 1: AUPRC vs nuisance dimension (n=500, alpha=0.5)
    for i, regime in enumerate(REGIMES):
        sub = agg[(agg["n_per_region"] == 500) & (agg["alpha"] == 0.5)
                  & (agg["regime"] == regime)]
        if sub.empty or sub["n_nuisance"].nunique() < 2:
            continue
        for method, grp in sub.groupby("partition_method"):
            if method == "pca":
                continue
            _errorbar_by(axes[1, i], grp, "auprc", "n_nuisance",
                         METHOD_COLORS.get(method, "k"),
                         METHOD_LABELS.get(method, method))
        axes[1, i].set_xscale("log")
        axes[1, i].set_xlabel("nuisance dimension")
        axes[1, i].set_title(f"{regime}, α=0.5, n=500", fontsize=9)
        axes[1, i].grid(alpha=0.3)
    axes[1, 0].set_ylabel("matched AUPRC")

    # Row 2: partition ARI vs purity by method
    runs = ResultStore(EXPERIMENT).read_all("runs")
    if not runs.empty:
        part = runs[runs["region"] == -1]
        for i, regime in enumerate(REGIMES):
            ax = axes[2, i]
            sub = part[part["regime"] == regime]
            for method, grp in sub.groupby("partition_method"):
                style = dict(s=6, alpha=0.3) if method == "random" else dict(s=10)
                ax.scatter(grp["partition_ari"], grp["partition_purity"],
                           color=METHOD_COLORS.get(method, "k"),
                           label=METHOD_LABELS.get(method, method), **style)
            ax.set_xlabel("partition ARI")
            ax.set_ylabel("purity")
            ax.set_title(f"{regime}: partition quality", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)

    for j in range(2, 6):
        axes[2, j].axis("off")

    fig.suptitle("Experiment 1: Gaussian grouped-covariance recovery")
    fig.tight_layout()
    out = FIGURES_DIR / "gaussian_multipanel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    main()