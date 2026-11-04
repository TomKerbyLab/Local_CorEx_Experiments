# Experiment 4: MNIST factor-ablation benchmark — human-pipeline port + results record

Motivation, pre-registered methodology, decision log, and results live in
[`../../paper/exp4.md`](../../paper/exp4.md) (§E4.1–§E4.15).

**This directory is now the run target**: a self-contained port (the
house pattern of `../exp3_redundancy/`) — flat modules, a local
`common/` subpackage, package-local regenerable outputs
(`run_results/`), and the committed record in `results/` + `figures/`.
The reference implementation `tnnls_experiments/exp4_mnist/` stays as
the spec record of the executed runs (see §E4.13).

## Port fixes folded in (per paper/exp4.md §E4.14/§E4.15)

- stage-aware completeness gate (`is_complete_for_stages` + stage-union
  markers, hollow-run guard): new stages can run after the main grid;
- MNIST under the repository-relative `data/mnist_classifier/`
  (decision #6; no cwd-relative downloads);
- the spec's `mnist_models` store table (one row per trained seed);
- the duplicated `load_splits` definition removed (audit item 5).

## Executed runs (2026-11-03)

- **Full grid on the reference implementation** (record + CSVs in
  `results/`): run id `mnist.3d97ba99ce.0`;
  5 seeds × 20 PHATE clusters (k=20, seeds 5200/5201) × layers H1–H3 ×
  top-5 factors × q ∈ {0.6,0.7,0.8,0.9} output-aware + q=0.80
  hidden-only; **600/600 fits, 141,000/141,000 interventions, zero
  failures, zero exclusions**; five per-seed probability-shift parquets
  (14.1M rows); ~49 min wall. §E4.14 = the pre-registered findings;
  §E4.14.1/§E4.14.2 = the exploratory profile and q-extension sections.
- **Extensions (also on the reference):** test-cluster assignment
  (checks 0.975/0.966; note the PHATE out-of-sample transform is
  thread-nondeterministic — rates vary a few % between runs, both
  passes); per-factor profile grids (768k cells); q = 0.95/0.99
  supplementary (66k rows; specificity collapse at 0.99 documented with
  support-overlap diagnostics).
- **Panel:** `figures/exp4_qualitative_reconstruction_panel.png` — the
  predeclared qualitative reconstruction (cluster maximizing
  min(n₄,n₉) = cluster 0; factor per layer by max{I(Z;L₄), I(Z;L₉)};
  factor-score-directed symmetric perturbations decoded through the
  mirror autoencoder (`train_mirror_ae.py`, test recon MSE 0.0215);
  explanatory only — never factor selection.

## Layout

| Path | Role |
| --- | --- |
| `results/mnist_*.csv` | The committed summary tables (primary 1–7 + coverage, effective units, TC-magnitude, cutoff sensitivity, profile cohorts/grid, factor assoc, extension diagnostics) |
| `results/mnist_seed*_meta.json` | Per-seed training records (test acc 0.9860–0.9864, zero flags) |
| `figures/` | Committed display set (fig01–fig07 + the qualitative panel) |
| `figures_exp4.py` | Renders the predeclared figures from `results/` only |
| `analysis_exp4_results.ipynb` | Exploration notebook (executed; per-reading tables + exploratory section) |
| `common/` | The ported shared infrastructure (config/corex/metrics/seeds/io/runtime/store/information/partition) |
| `classifier.py` `data.py` `train.py` | Spec-exact MLP, fixed splits (seed 5000), training (5100–5104) |
| `benchmark.py` | Stage runner: clusters, test_clusters, fits, ablations, extensions, profiles, shifts |
| `interventions.py` `decode_panel.py` `train_mirror_ae.py` | Metrics + predeclared panel machinery |
| `summarize.py` | Aggregates → `run_results/mnist_*.csv` (copy selected tables into `results/`) |
| `tests.py` `test_benchmark.py` `conftest.py` | Validation tests (spec tests 1–14 + machinery tests) — 36 passing |

## Usage (from this directory)

```bash
python train.py --seeds 5100 5101 5102 5103 5104   # classifiers (GPU, ~3 min each)
python -m pytest . -q                              # validation tests
python benchmark.py --config exp4_full.yaml        # full grid (~50 min on one RTX 6000)
python benchmark.py --config exp4_pilot.yaml       # feasibility pilot (~11 min)
python summarize.py                                # summary CSVs from raw shards
python figures_exp4.py                             # display set from results/
python train_mirror_ae.py && python decode_panel.py  # qualitative panel
```