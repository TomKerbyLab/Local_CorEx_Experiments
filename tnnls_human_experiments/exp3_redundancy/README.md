# Experiment 3: TC-matched redundancy vs synergy (human pipeline)

Motivation, methodology, pre-registered outcomes, and run plan live in
[`../../paper/exp3.md`](../../paper/exp3.md). The reference implementation in
`../../tnnls_experiments/exp3_redundancy/` is kept unmodified; this directory
is the port implementing the agreed redesign (§E3.4–§E3.11):

- **Equal-TC calibration** — a single population-TC target (0.5 bits), flip
  rates solved exactly by bisection on the analytic TC per (d, generator), and
  frozen before any replicate (`redundancy_data.py`).
- **Exact population references** — TC/DTC/Ω/mean pairwise MI from the exact
  pmfs, computed before every fit and recorded per replicate, so estimator
  bias is read against truth (`redundancy_data.population_quantities`).
- **Capacity arms** — main arm `dim_hidden = 2^(d-1)`, sensitivity arm
  `dim_hidden = 2`; never pooled (§E3.5).
- **Neutral tie handling** — seeded permutation before the descending sort
  (Appendix B decided rule; the true group's leading-column layout would be
  favored by the legacy rule) (`recovery_metrics.py`).
- **Two-stage discovered-support Ω** — freeze the top-factor support on the
  fit sample, estimate Ω on exactly those columns of a fresh characterization
  sample (independent derived seed); selection bias removed by data
  separation.
- **Failed fits as data** — a failed fit records one `failed_fit` info row and
  keeps the replicate's ground-truth-side content.
- **Pre-registered readout** — generator differences in top-factor TC read as
  paired differences against pre-specified ±0.05-bit equivalence margins, not
  against zero (`summarize_redundancy.equivalence_readout`).

## Layout

| File | Role |
| --- | --- |
| `redundancy_data.py` | Generators, exact pmfs, population quantities, calibration |
| `information_estimates.py` | Plug-in + Miller-Madow TC/DTC/Ω |
| `recovery_metrics.py` | Neutral-tie top-d, cumulative-MI supports, AUPRC |
| `corex_fitting.py` | Discrete Bio CorEx fits (seeded, RNG-guarded, atom-init contingency) |
| `seeds.py` | Derived seeds (sha256 over identifier parts) |
| `run_redundancy_experiment.py` | Runner: condition construction, fits, CSVs, run log |
| `parallel_runner.py` | Shard launcher (union of shards = serial run exactly) |
| `summarize_redundancy.py` | Aggregates, equivalence readout, limitations panel |
| `test_redundancy.py` | Validation tests (spec 03 items, redesigned) |

## Usage

```bash
uv run python run_redundancy_experiment.py smoke        # smoke: minutes
uv run python -m pytest test_redundancy.py -q           # validation tests
uv run python parallel_runner.py core --workers 16      # full grid (pending GO + favorable pilot)
uv run python summarize_redundancy.py --experiment core
```

Results default to `redundancy_results/<experiment>/...` per-condition CSVs,
writable atomically and restartable; shards resume by skipping completed
conditions.