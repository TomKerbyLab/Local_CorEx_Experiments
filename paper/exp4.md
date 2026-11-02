# Experiment 4: factor-ablation benchmark on MNIST — the behavioral-probe instrument on a real network

Companion to `simulation_methodology.md` (Experiments 1–2) and `exp3.md`
(Experiment 3). This document is the Experiment 4 pre-registration: motivation,
methodology, grid, and outcome-neutral readings — modeled on the Experiment 3
document. §E4.12 records implementation status; §E4.13 (results) is written
only after the executed run. Reference implementation
`tnnls_experiments/exp4_mnist/`; the human-pipeline package is the run target
and will implement any redesign agreed below.

**Open decisions for sign-off are collected in §E4.14.** Everything not
listed there is frozen.

## E4.1 Motivation and role in the program

Experiment 3 established (§E3.9) that the paper's neural-network experiments
operationalize the identify-then-intervene architecture: Local CorEx
identifies node groups; a second, structure-appropriate instrument
characterizes what they are doing. On a learned network the joint is not
tractable, so the second instrument is a **behavioral probe**: perturb the
learned factor's support, observe the model's prediction change against
matched controls. Experiment 4 is that experiment, in its principal
quantitative form.

> **Question (E4).** Across discovery clusters, hidden layers, and highly
> ranked Local CorEx factors, do cumulative-MI supports identify distributed
> sets of hidden units whose deletion produces substantial and
> class-concentrated changes in classifier behavior — beyond size-matched
> random deletions and beyond a univariate ranking of units by correlation
> with one output class?

The restraint fixed by §E3.9 applies verbatim here: the benchmark's metrics
(behavioral shift magnitude, class concentration, specificity against random
controls, directional concordance) characterize **influence**, never
"redundant role" or "synergistic role." The word "type" does not appear in
any exp4 claim; the instrument does not sign the interaction among the nodes.

What is *new* relative to the spec's formulation (04_mnist factor ablation):
the experiment is additionally an end-to-end statement for the paper —
recovery → supports → intervention is the full pipeline applied to a real
trained network, so a failure at any stage (bad recovery, diffuse supports,
small effects) is reportable as such. Hypotheses 1–5 of the spec collapse
into the five pre-registered readings of §E4.11.

## E4.2 Data and separation discipline

Canonical MNIST (60,000/10,000). The 60,000 training images are split
stratified (seed 5000, exact indices recorded) into:

- 45,000 classifier-training;
- 5,000 classifier-validation (checkpoint selection and early stopping only);
- 10,000 **discovery** — drives PHATE clustering, CorEx fitting, factor
  ranking, support construction, and qualitative selection;
- the 10,000 canonical test images are **untouched** and used only for final
  intervention evaluation.

Test labels and predictions never enter factor fitting or support
construction (validated by test, and by construction the fit functions accept
only discovery arrays). This is the program's data-separation convention
(§E3.7's fresh-characterization discipline) in its network form: the
intervention stage is the "fresh sample" for the discovered supports.

## E4.3 Classifier and training

Spec-exact MLP: h1 = ReLU(BN₁(W₁x)) dim 500, dropout 0.5; h2 dim 400; h3 dim
300; logits. H1/H2/H3 are the post-BN, post-ReLU, **pre-dropout**
representations; at evaluation dropout is off. CE loss on logits.

Five independently initialized models, root seeds 5100–5104: Adam, lr 10⁻³,
batch 128, max 100 epochs, early stopping on validation loss (patience 10,
lowest-val checkpoint restored). No ablation result influences checkpoint
selection. Test accuracy < 97% is a recorded **failure FLAG**, not silent
retraining (per seed). Minibatch order is derived from (seed, epoch) by the
program's hash convention.

## E4.4 Discovery clusters

One fixed partition reused for every model seed and layer: pixels scaled
[0,1], PHATE embedding to 10 dimensions (root seed 5200, n_pca 100), k-means
k = 20, n_init 50 (seed 5201). Digit labels are not used to construct the
partition — they are used only afterward to describe cluster composition and
to evaluate effects by class. PHATE coordinates, cluster labels, sizes, and
digit compositions are stored (`mnist_clusters`).

**Tiny-cluster policy (open, §E4.14):** clusters are sized as they fall; the
reference skips only empty ones. A predeclared minimum-size rule is proposed
below and should be frozen before the full run.

## E4.5 CorEx fits and factor ranking

For every (model seed s, cluster k, layer l ∈ {1,2,3}):

1. Run the classifier in eval mode on the cluster's discovery images; keep
   H₁, H₂, H₃ and pre-softmax logits L.
2. Standardize every hidden coordinate and logit **using only that
   cluster's observations**. Coordinates with variance < 10⁻¹² are flagged
   and **excluded from the fit** (original indices retained in metadata;
   `n_valid_hidden` / `n_excluded_hidden` recorded).
3. **Output-aware condition (primary):** V = [H | L] (standardized; logits
   included but never true labels — the label-free and output-aware framing).
   **Hidden-only condition (required ablation):** V = H.
4. Linear CorEx, M = 10 requested factors, deterministic fit seed derived
   from (model seed, cluster, layer, condition) via the hash convention.
5. Rank factors **by explained TC descending** (never by ablation effect);
   retain the top 5.

Factor metadata per retained factor: fit seed, convergence, TC, all
factor-to-hidden MI scores, all factor-to-logit MI scores, dominant
associated output class c* = argmax_c I(Z‸j; L_c), the gap between the two
largest logit MIs, valid/excluded count. Two estimator notes:

- **Logit-association source is recorded.** In the output-aware condition
  the factor-to-logit MIs are fit-side association scores
  (`logit_mi_source = "fit"`). In the hidden-only condition they are
  **post-hoc** estimates: factor scores Z (the linear latents of the fitted
  model on the same standardized cluster sample) against each standardized
  logit column, with the Gaussian MI −½log(1 − ρ²) — the same estimator
  family as the fit-side scores (`logit_mi_source = "post_hoc"`). The two
  sources are never pooled in summaries.
- **Units (open, §E4.14):** Linear CorEx reports TC/MI in nats. The
  Experiment 3 lesson (§E3.11 run-record correction) is that fit-side
  information quantities are converted to **bits at the fit boundary** so
  they are directly comparable to bit-scale quantities elsewhere in the
  paper; supports are unaffected (cumulative fractions are scale-invariant).
  Proposed default: bits.

**TC-ranking ties (open, §E4.14):** the program's neutral tie rule (seeded
permutation before the descending sort; Experiment 2/3 convention) is
proposed as the predeclared tie policy for TC ranking and for c* argmax
ties.

## E4.6 Hidden-unit supports

Supports are constructed from **factor-to-hidden MI only**; logit columns
never enter a deletion support. Cumulative-MI thresholds q ∈ {0.60, 0.70,
0.80, 0.90} in the output-aware condition, q = 0.80 only in the hidden-only
condition. Supports are nested by construction, and stored with requested
and achieved fractions, size sⱼ(q), density sⱼ(q)/d_l (d_l ∈ {500, 400, 300}),
ordered hidden-unit indices, and MI scores (`mnist_supports`).

## E4.7 Intervention semantics

**Local CorEx intervention:** a deterministic mask, inserted immediately
after the layer's BN + ReLU and before dropout/next affine, zeroes the
support coordinates: hⱼ,lᵢ = 0 for i ∈ Sⱼ(q), hⱼ,lᵢ unchanged otherwise.
Evaluation mode; **no** rescaling of survivors, no BN-statistic changes, no
downstream-bias edit, no retraining, no test-set recomputation of supports.
Activation masking is canonical; equivalence with outgoing-weight deletion
is unit-tested. The all-ones-mask baseline predictions are cached once per
model seed.

The trunk evaluator (cached test-set h_l, remaining layers re-run from the
masked activations) is numerically equivalent to the model-mask forward pass
by unit test across all layers; dropout is identity in eval mode, so the
trunk is deterministic.

## E4.8 Controls

**Matched random supports:** for every LC support of size s, 20 supports
sampled uniformly without replacement from the *valid* hidden coordinates of
the same layer (5 in the hidden-only condition). Seeds derived from the
complete intervention key by the hash convention — identical supports are
regenerated on resume (deterministic, no RNG-state dependence).

**Univariate logit-association support:** for the factor's c*ⱼ, the top-s
units by |corr(Hᵢ, L_c*ᵢ)| within the same discovery cluster (ties → lower
index), one per LC support size. Output-aware condition only (hidden-only
factors take their c* from the post-hoc estimates; the spec defines the
univariate control for the primary condition, and the hidden-only arm runs
at q = 0.80 with random controls only).

**Random-control probability:** one-sided empirical ĥ_p = (1 + Σ 1[g_r ≥/≤
g_LC]) / (R + 1), reversed inequality for entropies; R = 20 makes these
coarse — read descriptively, never as p-values.

## E4.9 Evaluation and metrics

Every intervention is evaluated on all 10,000 test images. For true digit y,
the class-mean probability vectors give the signed shift Δ_{y,c} =
p̄ᴬ_{y,c} − p̄⁰_{y,c} (full 10×10 matrix stored per intervention). Aggregates:

- **Effect/degree:** T_y = ½Σ_c|Δ_{y,c}|; T_max, T_weighted.
- **Input-class concentration:** r_y = T_y/ΣT_u; H_input = −Σ r_y log r_y /
  log 10, D_input = max_y r_y.
- **Output-mass concentration:** q_{y,c} = |Δ_{y,c}|/Σ_u|Δ_{y,u}|; H_output,
  D_output at y* = argmax T_y; c⁻/c⁺ = argmin/argmax_c Δ_{y*,c}.
- **Directional concordance:** 1(c⁻ = c*ⱼ), plus the rank of c*ⱼ among Δ_{y*,c}
  ordered most-to-least negative (for both conditions — c*ⱼ is fit-side in
  output-aware, post-hoc in hidden-only).
- **Accuracy family:** accuracy change, per-class recall change,
  confusion-matrix L1 change, mean cross-entropy change.

**Predeclared criterion (frozen):** *substantial and class-concentrated* ⇔
T_max ≥ 0.10 and H_input ≤ 0.50. Sensitivity table over T_max ∈ {0.05, 0.10,
0.15} × H_input ∈ {0.40, 0.50, 0.60} is predeclared as supplementary. Report
continuous metrics regardless of classification; no cutoff may be changed
after inspecting full-run outcomes.

**Storage:** `mnist_models` (training record), `mnist_clusters`,
`mnist_factors`, `mnist_supports`, `mnist_ablation_summary` (one row per
intervention), `mnist_probability_shifts` (one row per intervention × y × c
with class-mean baseline probability, ablated probability, and signed
change — parquet). One deviation from the spec's wording is accepted
(§E4.14): the probability-shift table stores **class-mean** probabilities
(the p̄ vectors), not per-image probabilities; per-image values remain
reconstructible from checkpoints, and storing 100 million rows of them is
not informative for any predeclared display.

## E4.10 Grid and uncertainty

Primary grid: 5 model seeds × 20 clusters × 3 layers × 5 retained factors ×
4 thresholds = **6,000 Local CorEx interventions**, plus their 6,000 × 20 =
120,000 matched random-support evaluations and 6,000 univariate supports;
hidden-only adds 5 × 20 × 3 × 5 factors × 1 threshold = 1,500 LC
interventions + 7,500 random controls. Fit stage: 600 output-aware + 600
hidden-only Linear CorEx fits.

The **model seed is the replicate** (n = 5): prevalence and paired
differences are computed per seed and summarized across seeds (mean, SE,
95% interval). For factor-level distributions, the hierarchical bootstrap
resamples seeds, then clusters within seeds. The 6,000 interventions are
never treated as independent observations.

## E4.11 Pre-registered readings (outcome-neutral)

Five readings; the paper takes whichever the data support. Nothing here is a
result.

1. **vs. matched random:** LC supports produce larger (T_max, T_weighted)
   and more class-concentrated (H_input, D_input) changes than matched
   random supports — supported → recovery provides a useful input to the
   behavioral instrument; unsupported → the recovery→intervention link is
   weak on this network, a discovery-limitation finding.
2. **vs. univariate:** paired LC − univariate differences at matched keys —
   multivariate factor membership adds behavioral specificity beyond
   single-logit correlation ranking, or does not.
3. **Layer/condition structure:** intervention effects (magnitude against
   random, prevalence, concordance, support density) differ systematically
   by layer (H1/H2/H3) and by input condition (output-aware vs. hidden-only
   comparison at q = 0.8, including post-hoc-association concordance) — or
   they do not.
4. **Threshold dose-response:** increasing q increases support density and
   effect magnitude, possibly at the cost of specificity — characterize the
   trade-off across the predeclared q grid.
5. **Concordance:** the factor-associated output class is often the most
   reduced class (rate by layer and condition) — or it is not.

The instrument-restraint statement (§E4.1) forbids any reading that
interprets prevalence or concordance as labeling an interaction type; a
supported Reading 5 is stated as directional influence, nothing more.

## E4.12 Implementation status and run record

- Reference implementation `tnnls_experiments/exp4_mnist/` (commit 01d1a40):
  spec-faithful; validation tests 1–13 covered (trunk equivalence,
  masking semantics, metric hand-checks, end-to-end smoke mechanics),
  test 14 (resume determinism of random masks) holds by construction and is
  asserted at port time.
- **Reference-implementation corrections (2026-11-02, pre-run):**
  1. Hidden-only condition now computes the spec-required post-hoc
     I(Zⱼ; L_c) association (§E4.5) — previously dominant_logit was unset
     for hidden-only factors and concordance could not be computed there.
  2. `mnist_probability_shifts` now carries class-mean baseline/ablated
     probabilities alongside the signed change (§E4.9).
  3. Near-zero-variance hidden coordinates are now excluded from the CorEx
     fit itself (spec §CorEx inputs, step 3); previously they were flagged
     but still fit.
  4. Dead code removed from the probability-shift expander.
- Benchmarks at the real fit shape (n ≈ 500 cluster sample × 510 columns,
  M = 10): Linear CorEx fit 0.2 s, converged — the 1,200-fit fits-stage is
  minutes. Ablation grid ≈ 282k interventions × partial forward pass on 10k
  test images (trunk path): budgeted well under one GPU-hour on one RTX 6000
  (exact smoke/pilot timings below).
- Executed, in order (all before full-grid commitment): classifier training
  smoke (seed 5100) — **188 s** (84 epochs, early-stopped; test accuracy
  0.9864, above the 97% flag threshold); mechanics smoke `--smoke` (1 seed
  × 3 clusters × layer 3 × 4 factors, both conditions, 3 random controls)
  — [timings + artifact check placeholder]; favorable pilot (2 model seeds
  × full 20 clusters × all layers, one threshold) — [placeholder]; then the
  full grid behind the user's GO.

## E4.13 Results

*(written after the executed run; nothing claimed here now)*

## E4.14 Open decisions requiring sign-off

| # | Decision | Proposed default |
|---|---|---|
| 1 | Fit-side TC/MI units | **bits**, converted at the fit boundary (Exp 3 lesson; supports unaffected) |
| 2 | TC-ranking / c* tie policy | **neutral tie rule** (seeded permutation before descending sort) |
| 3 | Tiny-cluster rule | clusters with **< 100 discovery observations are recorded and excluded from the intervention grid** (predeclared; expected to exclude 0–2 of 20; the cluster table always reports all 20) |
| 4 | Probability-shift table granularity | **class-mean probabilities** (deviation from the spec's per-image wording; per-image is checkpoint-reconstructible) |
| 5 | Hidden-only arm scope | univariate control **omitted** for hidden-only (spec defines it for the primary condition); post-hoc concordance **included** |
| 6 | MNIST data location | downloaded once under the repo `data/` directory (not a cwd-relative `mnist_classifier/`) |