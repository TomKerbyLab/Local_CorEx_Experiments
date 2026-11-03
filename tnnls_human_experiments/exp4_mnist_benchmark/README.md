# Experiment 4: MNIST factor-ablation benchmark — results record

Motivation, pre-registered methodology, decision log, and results live in
[`../../paper/exp4.md`](../../paper/exp4.md) (§E4.1–§E4.15). The reference
implementation and run target is `../../tnnls_experiments/exp4_mnist/`; this
directory preserves the **executed full grid's outputs** (the raw shard
directory `outputs/raw/mnist/` is regenerable and not tracked) and renders
the predeclared display set. The runnable package port (house-style runner)
may follow; the reference implementation + committed results is the run
record.

## Executed run (2026-11-03)

- Command: `python -m tnnls_experiments.exp4_mnist.train --seeds 5100..5104`,
  then `python -m tnnls_experiments.exp4_mnist.benchmark --config
  tnnls_experiments/configs/exp4_full.yaml --stages clusters,fits,ablations,shifts`.
- Grid: 5 seeds × 20 PHATE/k-means clusters (k=20, seeds 5200/5201) ×
  layers H1/H2/H3 × top-5 factors × q ∈ {0.6, 0.7, 0.8, 0.9}
  (output-aware) + q = 0.80 (hidden-only, 5 random controls).
- Outcome: **600/600 fits** (5 factors each, all converged, no shortfalls),
  **141,000/141,000 interventions** (7,500 LC + 127,500 random + 6,000
  univariate), zero zero-effect rows, no tiny-cluster exclusions, five
  probability-shift parquets (2.82M rows each). Wall ≈ 49 min.
- Run id `mnist.3d97ba99ce.0` (`results/full_run_report.txt`,
  `results/full_run_resolved_config.json`); a 2-seed q=0.8 feasibility
  pilot preceded the grid (shards preserved untracked under
  `outputs/raw/mnist_pilot_2seed_q08/`).

## Layout

| Path | Role |
| --- | --- |
| `results/mnist_*.csv` | The eleven summary tables from `tnnls_experiments/exp4_mnist/summarize.py` (primary 1–7 + coverage, effective units, TC-magnitude, cutoff sensitivity) |
| `results/mnist_seed*_meta.json` | Per-seed training records (test acc 0.9860–0.9864, zero flags) |
| `figures/fig01–fig06*.png` | The display set rendered by `figures_exp4.py` |
| `figures_exp4.py` | Renders all figures from `results/` only (no models) |
| `analysis_exp4_results.ipynb` | Exploration notebook over the committed CSVs (executed; displays the tables per pre-registered reading) |

## Regeneration

```bash
# summaries + figures + notebook all depend only on the raw shards or the CSVs:
uv run python -m tnnls_experiments.exp4_mnist.summarize   # from repo root
cd tnnls_human_experiments/exp4_mnist_benchmark && python figures_exp4.py
```

The raw shards/parquets live in `outputs/raw/mnist/` (untracked,
regenerable in ~50 min from the committed code/configs/checkpoint records).