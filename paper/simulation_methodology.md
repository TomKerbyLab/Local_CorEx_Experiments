# Simulation methodology: the two synthetic Local CorEx experiments

This document describes the methodology of the two synthetic experiments in
the human experiment pipeline (`tnnls_human_experiments/`):

- **Experiment 1 — Gaussian grouped-covariance recovery**
  (`synthetic/`, originally `exp1_gaussian/`), and
- **Experiment 2 — Nonlinear higher-order recovery**
  (`AI_implemented/exp2_nonlinear/`).

Section 1–4 are written to be usable nearly verbatim as the *Methods*
subsection of the paper; the appendices carry the full parameter tables,
metric formulas, reproducibility conventions, and output layout that would
fill the *supplementary appendix*. Section 5 records how these experiments
relate to the earlier reference implementation (`tnnls_experiments/`).

Throughout, a *condition* is one replicate of one data-generating
configuration; a *cell* is one set of rows of that data on which one CorEx
model is fit; and a *variant* is one (estimator, CoreEx-input) recipe
applied to a cell.

---

## 1. Experimental program and claims

The synthetic experiments support the paper's first claim:

> **Claim.** Localizing a CorEx model improves recovery when multivariate
> dependence varies across regions of the data manifold, provided that the
> partition is informative.

Experiment 1 establishes the claim with linear (covariance-structured)
ground truth and clean analytic control. Experiment 2 establishes it under
nonlinear, pairwise-independent dependence that is invisible to pairwise
scans, and adds two diagnostics that characterize what CorEx can represent
versus what generic EM optimization finds. In both experiments, the
comparison set is: a **global** baseline (one fit on the pooled data) versus
**localized** fits per estimated region, under an **oracle** partition
(upper bound), a **data-driven** partition (realistic), and **random**
partitions (negative control).

## 2. Shared conventions (both experiments)

1. **Paired design.** One dataset is generated per condition and shared by
   every method evaluated on that condition — global and localized fits,
   every partition method, every estimator, and every input variant see the
   same data. Method differences are therefore read like-for-like.
2. **Derived seeds (spec 00 rule).** No global RNG state is used.
   Seed = `hash(root_seed, purpose, condition identifiers)`
   (sha256 of the serialized parts), giving independent, reproducible seeds
   for data generation, partitioning, and each individual model fit. Rerunning
   a condition reproduces the data bit-for-bit.
3. **Data separation.** Diagnostics and any out-of-sample quantities are
   computed on independently generated data (Experiment 2's ≥10^5-sample
   diagnostics), never on the fitting data. Ground-truth masks define which
   columns are scored; routing/label columns used by the partitioner are
   excluded from ground truth in Experiment 2.
4. **Restartability.** Each condition writes its own files atomically
   (temporary file + rename). Completed conditions are skipped on rerun, so
   interrupted batches resume without repeating finished work, and combined
   CSVs are rebuilt from per-condition files.
5. **Failures are data, not crashes.** Failed model fits are recorded as
   rows with `fit_status = failed_fit` (and the exception text); collapsed
   partitions are recorded as `failed_partition`. No silently dropped runs.
6. **Batch logging.** Each runner invocation appends run-start and run-end
   records (UTC timestamps, status, written/skipped condition counts,
   elapsed time) to `run_log.jsonl` in the output directory, so completed
   files can be attributed to a batch without relying on file modification
   times.

---

## 3. Experiment 1: Gaussian grouped-covariance recovery

### 3.1 Data-generating process

Two equally-sized Gaussian populations ("clusters") are built on
**p = 25 signal variables** in **five disjoint groups of five**. Within each
group, variables are equicorrelated at **ρ = 0.7**; between groups the
covariance is zero. Marginal variances repeat [7, 3, 10, 5, 8] across the
five groups (realistic heterogeneous scale, inherited from prior work).

Three covariance regimes control how local structure varies across
populations:

- **homogeneous** (`same_dist`): both populations share the cluster-1
  covariance matrix — local structure is identical everywhere.
- **nondisjoint**: population 2 uses *overlapping* groups; each group retains
  3 members of a population-1 group and takes 2 members from the preceding
  group. Structure differs locally but groups share variables.
- **disjoint**: each population-2 group takes exactly one member from every
  population-1 group — groups exist in both regions but are structurally
  incompatible.

Population means are group-specific fixed vectors; a **separation
parameter α ∈ [0, 1]** interpolates the population-2 mean between the two
mean vectors (α = 0: identical means; α = 1: fully separated). This stages
"the regions differ in mean" independently of "the regions differ in
covariance structure".

Optionally, **p_n independent nuisance variables** are appended (variances
cycling through the signal-variance scale; zero covariance), which grow
dimensionality without adding local structure — the false-positive pressure.
Column layout: [0–24] signal, [25+] nuisance.

### 3.2 Partitioning

The two populations are pooled. Per condition, the standardized pooled data
is split into regions three ways:

- **oracle** — the true region labels (upper bound);
- **PHATE** — k-means (k = 2) on a 5-dimensional PHATE embedding of the
  pooled data (realistic estimate);
- **random** — a random relabeling preserving the true region sizes
  (balanced random partition; negative control).

Estimated labels are aligned to the truth with the Hungarian assignment for
diagnostics only. Recorded partition diagnostics per method: ARI, purity,
mapped accuracy, per-region purity, aligned 2×2 confusion counts, and the
partition time.

### 3.3 Model fitting

Each region (and the pooled data for the global baseline) is fit with
**Linear CorEx** requesting **M = 5 factors** (`gaussianize='outliers'`,
fixed seed 42 for model inits). In the dedicated `bio_check` configuration
the **Bio CorEx** estimator (discrete marginals, `dim_hidden = 2`,
`smooth_marginals = False`) replaces it to demonstrate estimator-agnostic
behavior. Every fit is scored against the region's true factor supports.

### 3.4 Recovery assessment

Each fit yields the factors × variables association-score matrix from
mutual information (Linear CorEx: `moments['MI']`; Bio CorEx: `mis`).
True groups (plus zero columns for nuisance variables) form binary indicator
vectors. Scoring proceeds in three steps:

1. **Pairing scores.** All (true group, learned factor) pairs are scored by
   average precision (AUPRC, using factor scores as the ranking), by
   prevalence-adjusted AUPRC, and by cosine distance.
2. **Matching.** Factors are matched to groups **once**, maximizing AUPRC,
   by the Hungarian assignment. This single assignment feeds every other
   metric.
3. **Matched metrics.**
   - *matched AUPRC* and matched cosine similarity;
   - *top-k F1* at k = the true group size (support recovery: how much of
     the group appears in the factor's top-scoring variables), with nuisance
     leakage measured two ways: the share of nonnegative score mass
     attributed to nuisance variables, and the share of the top-k set
     occupied by them;
   - *recovery* (matched AUPRC normalized by the number of **true groups** —
     penalizes missing groups), *precision* (normalized by the number of
     **learned factors** — penalizes extra factors), and their harmonic
     mean (the balanced summary), all in AUPRC and cosine versions;
   - directional diagnostics (best-matching group per factor and vice
     versa) as lower bounds on recoverable performance.

### 3.5 Experimental grids (16 paired replicates per condition)

| Grid | Levels |
|---|---|
| `primary` | n ∈ {50, 500, 5000} per region × α ∈ {0, 0.5, 1} × {homogeneous, nondisjoint, disjoint} |
| `factor_sensitivity` | M ∈ {3, 4, 5, 6, 7} requested factors (at n = 500, α = 0.5, nondisjoint and disjoint) |
| `bio_check` | Bio CorEx vs Linear CorEx (both estimators on shared data) |
| `nuisance_stress` | p_n ∈ {0, 25, 100, 475} (at n = 500, α = 0.5) |

---

## 4. Experiment 2: nonlinear higher-order recovery

### 4.1 Data-generating process

Two equally-sized **regions** over **14 signal variables** whose group
structures are deliberately *incompatible* across regions — no group is
shared:

- **Region 0:** {1,2,3}, {4,5,6}, {7,8,9,10}, {11,12,13,14}
- **Region 1:** {1,4,7}, {2,8,11}, {3,5,9,12}, {6,10,13,14}

Every region-1 group cuts across several region-0 groups, so factors
capturing one region's structure cannot coincide with the other's. Each
group of size d is generated independently by one of three laws:

1. **Parity** (discrete; pairwise-independent by design). B₁,…,B_{d−1} are
   iid Bernoulli(½) and B_d = B₁ ⊕ ⋯ ⊕ B_{d−1} ⊕ E with flip noise
   E ~ Bernoulli(η). At η = 0 every *proper subset* of the group's variables
   is mutually independent while the full group carries exactly TC = 1 bit
   of total correlation for every d ≥ 2 — dependence that exists only at
   order d, beyond any pairwise scan.
2. **Sine-sum** (continuous; the pairwise-independent continuous analogue).
   Θ₁,…,Θ_{d−1} iid Uniform(−π, π); Y = sin(ΣΘᵣ) + ε, ε ~ N(0, σ²). The
   response is marginally independent of each individual angle.
3. **Product** (supplementary comparison). Y = ∏Xᵣ + ε with Xᵣ iid N(0,1):
   nonlinear and higher-order-structured but **not pairwise independent**;
   its results are always reported with that qualification and never used as
   pairwise-independent ground truth.

### 4.2 Routing and nuisance columns

Column layout of every dataset: **[0–13] signal, [14–15] routing, [16+]
nuisance.**

- **Two routing variables** R | C = c ~ N(ν_c, I₂) with ν₀ = (−2α, 0) and
  ν₁ = (2α, 0) — mean separation 4α. They carry no factor structure: they
  encode region identity so the partitioner can discover it, and they are
  **excluded from all ground-truth scoring** by construction. This mirrors
  settings where region identity is observable through some "label-like"
  coordinates rather than through the factors of interest.
- **p_n nuisance variables** (Bernoulli(½) for the discrete generator,
  N(0,1) for the continuous ones) grow dimensionality with no structure.

For parity data the signal columns are 0/1 while routing columns are
continuous; before discrete CorEx fitting the routing columns are discretized
(equal-frequency, 10 bins) so the full matrix satisfies the integer-input
requirement. For continuous data the whole matrix is standardized and then
equal-frequency discretized (10 bins) before fitting.

### 4.3 Experimental grids (16 paired replicates per condition)

| Grid | Levels |
|---|---|
| `discrete_core` | parity: n ∈ {500, 5000} × α ∈ {0, 0.5, 1} × η ∈ {0, 0.05, 0.15}, p_n = 14 (288 conditions) |
| `continuous_core` | sine-sum: same n × α shape × σ ∈ {0.05, 0.20, 0.50}, p_n = 14 (288) |
| `discrete_supplementary` | small-n (n = 50) across the core α × η grid; extra flip rates η ∈ {0.10, 0.20}; nuisance sweep p_n ∈ {0, 56} |
| `continuous_supplementary` | same small-n and nuisance-sweep axes; product-generator comparison at the core condition |
| `smoke` | 2 conditions (n = 300; parity + sine-sum) — the regression baseline for refactoring |

Root seeds: 2000+ (parity), 3000+ (sine-sum / product).

### 4.4 Partitioning

On the standardized pooled data: **oracle** (true labels); **PHATE**
(10 dimensions) + k-means (k = 2, n_init = 50); **raw k-means** directly on
the standardized variables; and **10 balanced random partitions** (kept
separate in all outputs — control replicates, not repeats).

Estimated labels are aligned to the truth (Hungarian) only for the recorded
partition diagnostics. The **fit cell** for an estimated partition is the
*raw* estimated cluster with maximal overlap with the target region
(aligned ids never index raw labels — a documented convention, since the
label map can be a swap). When both regions select the same raw cluster,
one fit serves both, recorded via `same_cluster_selected`. A collapsed
k-means output (a single estimated cluster) is recorded as
`failed_partition` rather than silently dropped.

### 4.5 Fit variants

Per condition, a spec table enumerates fit variants (estimator ×
CoreX-input); downstream code reads spec fields and never parses names:

- **bio** (primary estimator) on the standard input recipe;
- **linear** (covariance-based baseline) on the same recipe;
- **native-gaussian Bio CorEx** on the standardized (unbinned) space — a
  secondary variant for the continuous generator;
- **`no_routing`** input — the routing columns are dropped from the fit
  matrix entirely; association scores are re-expanded to full width with
  zeros so all scoring stays aligned with full-width ground truth, and
  routing false positives are recorded as not-applicable for these variants.

### 4.6 Model fitting

**M = 4 requested factors** per fit. Bio CorEx is fit with
`dim_hidden = 4` (sized to cover the 2^d parity atoms), `max_iter = 200`,
and `n_repeat = 5` EM restarts keeping the maximum-TC solution; Linear CorEx
with `max_iter = 2000` and `gaussianize='outliers'`. Two hardening rules are
worth recording: every fit receives an explicit integer seed derived from
the condition and cell (`hash(root_seed, 'corex', draw, method, cell,
estimator, variant)`) — required, because Bio CorEx reseeds the global numpy
RNG from OS entropy when unseeded — and the global RNG state is
saved/restored around every fit, so fits cannot perturb pipeline randomness
and identical inputs reproduce bit-for-bit.

### 4.7 Recovery assessment

Each fit yields the nonnegative association scores a_{ji} = I(Z_j; X_i)
(raw model scores clipped at zero — the documented transform; for no_routing
variants re-expanded with zeros at the dropped columns). Factors are matched
to the four true group masks by the Hungarian assignment **maximizing cosine
similarity**, then held fixed for all other metrics. Per matched pair:

- matched **AUPRC** (with prevalence adjustment) and cosine similarity;
- **top-d precision / recall / F1 / Jaccard** at d = the true group size;
- **nuisance false positives** and **routing false positives**, measured two
  independent ways: the share of nonnegative score mass attributed to the
  role, and the share of the top-d set occupied by it;
- the **80% cumulative-MI support**: sorting the factor's scores descending
  and taking the smallest prefix reaching 80% of the factor's total MI,
  reported with size, density, and F1 against the true group, plus the role
  fractions *within* the support. Factors whose total MI ≤ 10⁻¹² are
  `support_degenerate`, flagged, and excluded from support metrics.

Aggregation is within each replicate fit (a random-partition draw is never
averaged into another draw; collapsing over draws is left to downstream
analysis).

### 4.8 Structure and capability diagnostics (≥10^5 samples, fresh data)

Two diagnostics, run on **100,000 freshly generated observations** per cell
(independent of fitting data), at (generator, d ∈ {3,4}, noise) level:

1. **Structure diagnostic** — certifies the premise of the experiment. Per
   cell: max |Pearson|, max pairwise mutual information (10-bin
   discretization for continuous generators; raw bits for parity), plug-in
   group TC in bits, and a **permuted-TC control** (one member independently
   permuted, destroying higher-order structure while preserving marginals;
   must be ≈ 0). For parity at η = 0 this shows pairwise MI ≈ 0 with
   group TC = 1 bit; the same separation holds for the sine-sum generator.
2. **Capability diagnostic** — characterizes standard CorEx EM. Standard
   random-initialization Bio CorEx finds TC ≈ 0 on pure parity groups while
   the identical variational family, initialized with a **one-hot latent on
   the joint atom** (a ground-truth-informed init, rows flagged
   `uses_ground_truth_init`), reaches the full parity TC. Seeded rows are a
   *capability statement* and permanently excluded from recovery summaries.
   Conclusion: the failure of standard CorEx EM on pairwise-independent
   higher-order structure is an **optimization failure, not an
   expressiveness failure** — the evidence that motivates discussing
   initialization-based searches for localized structure.

All model-reported TCs are converted nats → bits before comparison.

---

## 5. Relation to the reference implementation (`tnnls_experiments/`)

An independent, spec-driven implementation of both experiments exists under
`tnnls_experiments/` (built from `agent_instructions/01` and `02`). At the
methodology level — generators, region/group structures, routing and
nuisance columns, partition methods and parameters, estimator options,
factor counts, matching objective, support definition, and grid levels — it
matches the human implementation described here. The differences are
infrastructural (YAML configs, parquet shard storage, `.done.json` run
gating, process-pool execution, seed-derivation string formats), so its
results agree with the human pipeline *statistically*, not bitwise. It is
retained as a cross-implementation robustness reference and will be retired
once the human pipeline's material is finalized.

---

## Appendix A. Parameter tables

### A.1 Experiment 1 (Gaussian)

| Parameter | Value |
|---|---|
| Signal variables p_s | 25 (5 disjoint groups of 5) |
| Within-group correlation ρ | 0.7 |
| Marginal variances | [7, 3, 10, 5, 8] repeated per group |
| Cluster-1 means (per group) | [5, 10, 1, 10, 3] |
| Cluster-2 means (per group) | [2.5, 6, 3, 1, 10] |
| Regimes | homogeneous / nondisjoint (3 shared + 2 from preceding group) / disjoint (1 from each group) |
| Mean separation | α ∈ {0, 0.5, 1} interpolating cluster means |
| Nuisance p_n | 0 (primary); {0, 25, 100, 475} (stress) |
| Factors M | 5 (sensitivity: 3–7) |
| Estimator | Linear CorEx (`gaussianize='outliers'`, seed 42); Bio CorEx in `bio_check` |
| PHATE | 5 components; k-means k = 2 |
| Random partitions | per dataset, size-preserving |
| Replicates | 16 per condition (derived seeds) |

### A.2 Experiment 2 (nonlinear)

| Parameter | Value |
|---|---|
| Signal variables | 14 |
| Region groups | R0: {1,2,3},{4,5,6},{7–10},{11–14}; R1: {1,4,7},{2,8,11},{3,5,9,12},{6,10,13,14} |
| Generators | parity (η), sine-sum (σ), product (σ, supplementary) |
| Noise levels | η ∈ {0, 0.05, 0.15} (+ {0.10, 0.20} supplementary); σ ∈ {0.05, 0.20, 0.50} |
| Routing | 2 variables, N(ν_c, I₂), ν = (±2α, 0); mean distance 4α |
| Nuisance p_n | 14 (core); sweep {0, 56} (supplementary) |
| Sample sizes n per region | 5000 / 500 (core); 50 (small-n) |
| Replicates | 16 per condition (parity seed base 2000; continuous 3000) |
| Factors M | 4 |
| Bio CorEx | discrete marginals, dim_hidden = 4, max_iter = 200, n_repeat = 5 |
| Linear CorEx | max_iter = 2000 |
| Discretization | equal-frequency, 10 bins (fixed pre-run rule; bin edges saved) |
| Standardization | z-scored columns (variance floor 1e-12); near-constant columns pass zeros |
| PHATE | 10 components; k-means k = 2, n_init = 50 |
| Random partitions | 10 balanced draws per dataset |
| Support | 80% of total factor MI |
| Diagnostics | 100,000 fresh samples per (generator, d, noise) cell |

## Appendix B. Metric definitions

**Association scores.** For learned factor Z_j and observed variable X_i,
the nonnegative score a_{ji} = I(Z_j; X_i). Linear CorEx returns
−½ log(1 − ρ²_{j,i}) (the Gaussian MI); Bio CorEx returns a discrete MI
estimate (clipped at zero after raw values are stored). For no_routing
variants, zero columns are re-inserted at dropped positions.

**Matching.** Hungarian (one-to-one) assignment of learned factors to true
groups, maximizing cosine similarity in Experiment 2 and maximum average
precision in Experiment 1, held fixed for every other metric.

**Per-matched-pair metrics.**

- AUPRC: `average_precision_score(group, factor_scores)`;
  prevalence-adjusted: (AP − prevalence) / (1 − prevalence).
- Cosine distance/similarity between the true-group indicator and the
  factor-score row (zero vectors handled as worst-case).
- top-d precision / recall / F1 / Jaccard, where the predicted support is
  the d highest-scoring variables of the matched factor, d = group size
  (stable descending sort, ties by lower index).
- Role false positives (nuisance or routing):
  - *score-mass share*: Σ a_{ji} over role columns / Σ a_{ji} overall
    (nonnegative scores);
  - *top-d occupancy*: mean of the role mask on the top-d indexes;
  - *support occupancy*: the same within the cumulative-MI support.
- 80% cumulative-MI support: with s_{(1)} ≥ ⋯ ≥ s_{(p)} the sorted scores
  and T = Σᵢ max(s_{(i)},0), the smallest prefix with
  Σᵢ≤m s_{(i)} / T ≥ 0.8; degenerate factors (T ≤ 1e-12) are flagged and
  excluded.

**Aggregate recovery.** With matched values m₁,…,m_k: *recovery* =
Σⱼ mⱼ / (#true groups) — penalizes missing groups; *precision* =
Σⱼ mⱼ / (#learned factors) — penalizes extra factors; *balanced* = harmonic
mean of the two. Reported for AUPRC and cosine.

**Partition diagnostics.** ARI (on raw estimated labels), purity
(max-contingency mass), mapped accuracy (after Hungarian label alignment),
per-region purity, aligned confusion counts, and partition time.

## Appendix C. Reproducibility conventions

- **Seeds.** Derived from condition identifiers by sha256 hashing, one seed
  per purpose: data generation, partitioning (plus per-draw variants), and
  every individual model fit. No global RNG use; global RNG state is
  saved/restored around CorEx fits (both estimators manipulate it
  internally). Experiment 1 derives data seeds from
  (rep, n, α, regime, nuisance) tuples; Experiment 2 from the root seed and
  a purpose tag.
- **Batch logging.** `run_log.jsonl` (one JSON line per event) in each
  output directory: `run_start` (UTC timestamp, experiment, repetitions) and
  `run_end` (UTC timestamp, status `completed|failed` even when the run
  crashes, conditions written/skipped, elapsed seconds). This is timing
  metadata only and never part of experimental outputs (it is excluded from
  golden-baseline regeneration).
- **Restartability.** Atomic per-condition writes; skip-if-complete on
  rerun; combined CSVs rebuilt from per-condition files; failed fits and
  failed partitions recorded in the result tables.
- **Golden smoke baseline (Experiment 2).** `golden_smoke/` holds the
  byte-frozen output of the 2-condition smoke run plus a sha256 manifest;
  the known non-reproducible field is the wall-clock `runtime_s` column.

## Appendix D. Output layout

Per condition (both experiments): a `runs` CSV (one row per attempted fit —
including failures — with condition keys, partition diagnostics, fit status,
runtime, convergence, factor validity, total TC) and a `matches` CSV (one
row per matched factor–group pair with all per-pair metrics); Experiment 2
also writes a per-condition `.meta.json` with the resolved configuration and
defaults. Combined CSVs are rebuilt per experiment after each run.
`summarize_*` scripts aggregate within-fit means (random draws kept
separate) and produce the paper figures; Experiment 2's main figure is a
2 × 3 panel (discrete / continuous rows: AUPRC vs α by partition method;
Bio vs Linear by noise under oracle fits; the structure-diagnostics panel).