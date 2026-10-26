# Refactor plan: simplify the nonlinear pipeline for human readability

Scope: `tnnls_human_experiments/AI_implemented/exp2_nonlinear/` only. The
reference implementation (`tnnls_experiments/exp2_nonlinear/`) and the
Gaussian pipelines are not touched.

## Goal and hard constraints

A reader should be able to follow one condition from seed to CSV row by
reading `run_nonlinear_experiments.py` top to bottom, with no helper
knowledge outside the module docstrings. Functionality must not change:

- CSV schemas, column names, and file layout stay byte-compatible.
- Every derived seed (data, partitions, per-fit model seeds) stays
  identical — same seed string -> same integer.
- Grids, Bio/Linear options, matching objective (cosine), 80% supports,
  and all recorded metrics stay identical.

## Safety net (do before anything else)

1. DONE — the golden baseline is captured at `golden_smoke/` (the two
   condition CSV pairs plus combined CSVs, summary CSV, figure, and a
   `sha256_manifest.txt` over everything except the manifest itself).
   Deterministic seeds mean a refactor that preserves behavior reproduces
   these files bit-for-bit: re-run smoke into a fresh output dir and
   `diff -r` (or re-hash) against `golden_smoke/`.
2. After EVERY step below: `uv run pytest` (23 tests) and re-run smoke +
   `diff -r` against `golden_smoke/`. A diff is a refactoring bug, full stop.
   ONE known exception (found when comparing the first rerun): the
   `runtime_s` column in `root_*_runs.csv` is wall-clock timing and can
   never reproduce; ignore it — every other byte must match.

## Step 1 — delete dead API (safest first; ~150 lines gone)

Cross-referenced usages show these are called only from their own module
(or nowhere at all):

- `recovery_metrics.py`: `evaluate_factor_recovery`,
  `factor_scores_from_model`, `recovery_result_fields`,
  `recovery_assignment_fields`, `RECOVERY_COLUMNS`. After deletion,
  `match_factors_to_groups` is the ONE entry point; the runner consumes
  `recovery['assignments']` directly (it already does).
- `partitioning.py`: `local_partition_fields`, `empty_partition_fields`,
  `PARTITION_SEED_CODES`.
- `corex_fitting.py`: `factor_tc_values`.

Also drop unused dataclass fields if any carry no reads
(`CumulativeSupport.requested_fraction` is a candidate — verify first).

## Step 2 — one rule, one place (no duplicated helpers)

Three copies of equal-frequency discretization and two of standardization
currently exist. Consolidation:

- `corex_fitting.py` keeps `standardize_columns` and renames
  `discretize_equal_frequency` -> `discretize_columns` (the single
  discretization rule; spec already locates input prep at the fitting
  layer).
- `run_nonlinear_experiments.py` and `structure_diagnostics.py` import
  both instead of carrying private copies.
- New 20-line `seeds.py`: `derived_seed` (sha256 of serialized parts,
  docstring citing spec 00). `runner` and `structure_diagnostics` import
  it (two copies exist today). Do NOT touch
  `../synthetic/experiment_common.deterministic_seed` — changing it would
  alter Gaussian-experiment seed values.

## Step 3 — extract `config.py`, then kill the string-suffix encoding

### 3a — one file of knobs (mechanical, byte-safe)

Move the machine-readable choices out of the runner into a `config.py`
module — the single place to open when planning or adjusting a run:

```python
# config.py — plain Python literals, imported by runner, summarizer, tests
DEFAULTS      # fit/estimation options shared by all experiments
EXPERIMENTS   # per-grid option tables (estimators, corex_inputs, builders)
GRID_LEVELS   # n/alpha/eta/sigma level tuples + seed ranges (2000-2015, 3000-3015)
```

Why a Python module and not a runtime-read JSON/YAML file:

- The grid builders are loop logic (seed offsets, `derived_seed`), not
  data; a data file would need a mini-DSL interpreter, which is the
  opposite of simpler.
- These grids are spec-fixed and test-pinned
  (`test_condition_grids_match_spec`); an external file adds validation
  problems (typos, silent defaults, schema drift) without buying
  editability that matters.
- The provenance requirement is already met at runtime:
  `save_condition_metadata` writes the resolved condition config beside
  every CSV.
- It mirrors the sibling pipeline (`run_gaussian_experiments.py` keeps
  `EXPERIMENTS` as module-level literals) and preserves read-top-to-bottom
  order: config, data, logic, entry point.

This step is import-path shuffling only; the safety-net diff must be clean
before 3b.

### 3b — explicit variant spec table (no name suffixes)

Variant identity is currently spread across name suffixes parsed by
`endswith('no_routing')` / `startswith('gaussian')` in three places. Replace
(`corex_input_variants`, variant plumbing in `fit_matrices`,
`fit_variant`, `score_region_fit`) with one explicit spec table per
condition, built from the config:

    {'name': 'bio_binned_with_routing', 'estimator': 'bio',
     'discretize': 'binned' | 'gaussian' | None, 'drop_routing': bool}

- `fit_variants(generator, config)` builds the table (one function, one
  decision about what variants exist per generator).
- `corex_input_matrix(dataset, spec, n_bins)` builds the fit matrix for a
  spec; scoring reads `spec['drop_routing']` (no name parsing).
- Variant display names in rows stay EXACTLY as today
  ('bio', 'binned_with_routing', 'gaussian_with_routing', ...) so CSVs,
  the summarizer, and cross-checks are unaffected.

## Step 4 — uniform cell model (the main structural simplification)

`run_condition` currently branches three ways (fixed cells /
estimated partitions / random draws) with dict-splicing
(`{**cell, 'region_id': ..., 'fields': ...}`) and a closure
(`process_cell`) that mixes fitting, row building, and scoring. Replace
with FOUR small functions that all return the same shape — a list of cell
dicts `{method, draw, name, region_id, rows, regions, fields}`:

    global_cells(rows)                  # 1 cell, regions (0, 1), id -1
    oracle_cells(region)                # 2 cells, ids 0 / 1
    estimated_cells(partition_result, region)   # argmax-overlap cells
    random_cells(draw, partition_result, region)  # same, draw=draw

`partition_cells` (argmax-overlap selection + `same_cluster_selected`) is
absorbed into `estimated_cells`/`random_cells`. `run_condition` becomes:
build all cells -> for each cell, for each variant: fit, run row, match
rows -> collapsed-partition ValueError handled once at the top. The
internal partition key 'raw' becomes 'raw_kmeans' everywhere (today the
row label is 'raw_kmeans', the dict key is 'raw').

## Step 5 — summarize_nonlinear.py query helpers

Extract one `query(summary, generator, alpha, noise, estimator, ...)` used
by both panels instead of hand-written filter chains (panel 3 inline
diagnostics filter stays small). Remove leftover unused locals. Figure
output must render identically.

## Step 6 — docstring and name pass

- Module docstrings carry the "why"; function docstrings carry the
  contract only. Delete narration that repeats code.
- Consistent vocabulary: `noise` (with `noise_name` in rows only),
  `region` = true region id, `cluster` = raw estimated label id, `cell` =
  what gets fit, `variant` = estimator x corex-input choice.

## Step 7 — sync README and tests

- README file table reflects the new `config.py`/`seeds.py` and renamed
  helpers; running instructions unchanged.
- Tests only update imports if touched by steps 2/3; the 23 tests otherwise
  stay as the behavioral contract (the grid-count test now reads
  `GRID_LEVELS` through `config.py`).

## Expected outcome

| File | Now | After |
|---|---|---|
| run_nonlinear_experiments.py | 926 | ~550 |
| recovery_metrics.py | 533 | ~330 |
| structure_diagnostics.py | 329 | ~250 |
| summarize_nonlinear.py | 340 | ~300 |
| config.py (new) | — | ~80 |
| seeds.py (new) | — | ~20 |
| total | 3,263 | ~2,530 |

## Out of scope (unchanged)

- Any behavior: seeds, grids, CSVs, metrics, figure content, Bio options.
- `tnnls_experiments/` (cross-check reference) and `../synthetic/`.
- Moving the code out of `AI_implemented/` (separate future step).