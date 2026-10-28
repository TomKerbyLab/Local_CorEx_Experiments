# Nonlinear higher-order Local CorEx experiments (Exp 2)

AI-implemented, human-style build of experiment 2
(`agent_instructions/02_nonlinear_higher_order_recovery.md`). This is a
*transitional* copy: it follows the monolithic flat-script style of
`../../gaussian/`, lives here until it has been reviewed and understood,
and can then move to the normal human experiments location.

## Layout

| File | Purpose |
|---|---|
| `nonlinear_data.py` | Data generators (parity, sine-sum, product) + ground truth. 14 signal variables; 2 regions with interleaved group structures; 2 routing variables `N(nu_c, I_2)` with mean distance 4*alpha; p_n nuisance variables. Column layout: [0..13] signal, [14..15] routing, [16..] nuisance. |
| `partitioning.py` | Partition methods: oracle, PHATE(10d)+k-means (k=2, n_init=50), raw k-means, balanced random. Boolean masks per aligned region (aligned ids never index raw labels). |
| `corex_fitting.py` | Bio CorEx (primary) and Linear CorEx (baseline) fitting. Explicit integer seed REQUIRED; global numpy RNG state saved/restored around both estimators (they reseed it internally). `BioCorexAtomSeeded` for the capability diagnostic only. |
| `recovery_metrics.py` | Hungarian matching on **cosine similarity** (spec objective; the old Gaussian module maximized AUPRC), matched AUPRC, top-d P/R/F1/Jaccard, nuisance/routing false positives (top-d occupancy + score-mass), 80% cumulative-MI support (size/density/F1/FPs, degenerate handling). |
| `structure_diagnostics.py` | >= 100k-sample diagnostics in bits: max |Pearson|, max pairwise MI, group TC, permuted TC; plus the capability diagnostic (standard EM ~0 TC on pure parity vs atom-seeded init reaching full TC). Run: `python structure_diagnostics.py`. |
| `config.py` | All machine-readable knobs: fit defaults, per-experiment option tables, grid levels, root-seed bases. Plain Python (versioned, importable, validated by tests); grid loop logic stays in the runner. |
| `seeds.py` | `derived_seed`: sha256 of serialized condition identifiers — the one process-independent identifier-to-seed rule (spec 00). |
| `run_nonlinear_experiments.py` | Entry point. Runs conditions and writes restartable per-condition CSVs. Each generator gets one variant-spec table (estimator x corex-input recipe); downstream code never parses variant names. `--shard k/N` attempts only every N-th condition (deterministic stride selection; shards cover the list exactly, so N parallel workers reproduce the serial run); `--combine-only` recombines per-condition CSVs without fitting. |
| `parallel_runner.py` | Spawns N parallel shard workers of the runner for one experiment (one BLAS thread per worker), logs per shard under `<output-dir>/shard_logs/`, and recombines with `--combine-only` when all succeed. Interrupted batches resume by relaunching the same command. |
| `summarize_nonlinear.py` | Per-fit summary CSV (random draws kept separate) + the 2x3-panel main-paper figure. |
| `test_nonlinear.py` | The 10 spec validation tests + regression tests. `uv run pytest tnnls_human_experiments/AI_implemented/exp2_nonlinear/`. |

## Run

From this directory (or use `uv run python ...` from the repo root):

```bash
# smoke (2 conditions, ~1-2 min): parity + sine_sum at n=300
python run_nonlinear_experiments.py smoke --output-dir nonlinear_results

# dry run: condition and fit counts only
python run_nonlinear_experiments.py discrete_core --dry-run

# core grids (288 conditions each; Bio CorEx at n=5000 is slow)
python run_nonlinear_experiments.py discrete_core --output-dir nonlinear_results
python run_nonlinear_experiments.py continuous_core --output-dir nonlinear_results

# secondary corex-input variant (routing columns dropped from the fit)
python run_nonlinear_experiments.py discrete_core --include-no-routing

# parallel: 16 shard workers (measured: parity n=5000/region is the slow
# condition at ~2.8 h; 16 workers put both core grids well under a day)
python parallel_runner.py discrete_core --workers 16 --output-dir nonlinear_results
python parallel_runner.py continuous_core --workers 16 --output-dir nonlinear_results

# >= 100k structure + capability diagnostics
python structure_diagnostics.py --output-dir nonlinear_results

# summary CSV + main-paper figure
python summarize_nonlinear.py --results-dir nonlinear_results
```

Completed conditions are skipped by default (`--overwrite` reruns them), so
interrupted runs resume in place. Each condition writes
`root_XXXXX_runs.csv` (one row per fit) and `root_XXXXX_matches.csv`
(one row per matched factor), plus a `.meta.json` with the resolved
configuration. All conditions within a file share ONE generated dataset;
failed partitions/fits are recorded with `fit_status`, never silently
dropped, and are retried on rerun.

## Grids

- `smoke`: parity eta=0.05 and sine_sum sigma=0.20, n=300, 1 replicate each.
- `discrete_core`: n in {500,5000} x alpha in {0,0.5,1} x eta in
  {0,0.05,0.15}, seeds 2000-2015 (288 conditions).
- `continuous_core`: same but sigma in {0.05,0.20,0.50}, seeds 3000-3015,
  plus native gaussian Bio CorEx variant.
- `*_supplementary`: one sensitivity axis at a time (n=50 regime, extra
  noise levels, nuisance sweep {0,56}; product-generator comparison with
  the caveat that product groups are NOT pairwise independent).

Per condition: partitions `global`, `oracle`, `phate`, `raw_kmeans`, and
10 balanced-random draws (draws kept separate in all outputs); estimators
`bio` (M=4, dim_hidden=4, n_repeat=5) and `linear`; scoring with
nonnegative association scores, cosine-objective Hungarian matching, and
80% cumulative-MI supports.

## Conventions (spec 00)

- Derived seeds via sha256 of serialized condition identifiers
  (`derived_seed`) — never builtin `hash()` (salted per process).
- Data separation: fitting data and diagnostics data are generated
  independently (diagnostics use >= 100k fresh samples).
- TC units: the plugin diagnostics and supports are in BITS; note that
  `local_corex` reports model TCs/MI in NATS — `factor_tc` in matches rows
  holds raw model units, and the capability diagnostic converts to bits.
- Never overwrite completed conditions; record failures with condition keys.