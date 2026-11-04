# Experiment 4: factor-ablation benchmark on MNIST — the behavioral-probe instrument on a real network

Companion to `simulation_methodology.md` (Experiments 1–2) and `exp3.md`
(Experiment 3). This document is the Experiment 4 pre-registration: motivation,
methodology, grid, and outcome-neutral readings — modeled on the Experiment 3
document. §E4.13 records implementation status; §E4.14 (results) is written
only after the executed run. Reference implementation
`tnnls_experiments/exp4_mnist/`; the human-pipeline package is the run target
and will implement any redesign agreed below.

**Feedback record:** four external reviews shaped this document. Round 1
(2026-11-03, first version of `exp4_feedback.md`) endorsed the design and
requested interpretive/analysis revisions — all adopted, marked **[rev]**;
round 2 corrected the first in-cluster analysis (§E4.15, decisions #9–15).
Round 3 (2026-11-03, second version of `exp4_feedback.md`) audited the
round-1/2 analyses and conclusions; the reviewer had intention context but
no code access, so **every computable claim was verified against the code
and the committed tables before adoption** (`analysis_feedback_round3.py`
reproduces every number quoted below), its confirmed arithmetic/typo
corrections were applied, statements it showed to be under-supported were
narrowed or withdrawn, and its requested recomputations were run from the
committed shards. Round-3 changes are marked **[rev3]** and recorded in
§E4.15. Round 4 (2026-11-04, current version) audited the round-3
revision; its remaining items — the panel formula (an implementation
error, fixed and the panel regenerated), the extension's authoritative
version and full-precision closure, the cluster-identity routing check,
mixed-cluster cohort counts, the crossed-design bootstrap, and wording —
were likewise verified first, then adopted as **[rev4]** (§E4.15,
decisions #28–34). Round 5 (2026-11-04, current version) confirmed the
round-4 fixes and raised three remaining consistency items (a
mis-quoted per-seed range, a now-overreached mixed-cluster
interpretation, and an inaccurate change summary) plus small edits;
all verified first, then adopted as **[rev5]** (§E4.15, decisions
#35–39). The consolidated account of record is §E4.16.

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

The experiment also carries the pipeline's three distinguishable questions
**[rev]** — which the analyses now test separately rather than implicitly
(§E4.12, Reading 3): (i) how highly ranked a factor is (explained TC), (ii)
how broadly its information is distributed across units (support size /
density / effective number of units), (iii) what deleting its support does
to behavior.

Two boundary statements held regardless of outcome **[rev]**. MNIST supplies
no ground-truth hidden-unit factor structure against which to declare
recovery successful or unsuccessful: the only comparisons available are
behavioral, against the specified controls. And unsuccessful outcomes get
the neutral reading — "the discovered supports did not yield stronger
behavioral selectivity under this deletion probe" — because small deletion
effects can reflect robustness, compensating pathways, or an intervention
that poorly exposes the discovered dependence; they do not diagnose
discovery. Any claim that *locality itself* improves discovery would
require a Global CorEx comparison, which is deliberately not added: no such
claim is in scope (§E4.15, decision 8).

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

At port time, MNIST is downloaded once under the repository-relative
`data/` directory (`mnist_classifier/` is the reference implementation's
cwd-relative root, §E4.15 decision 6).

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

**Tiny-cluster rule (frozen):** clusters with fewer than 100 discovery
observations are recorded in `mnist_clusters` and excluded from the
intervention grid (a feasibility rule — some cluster sizes vary; the
threshold is frozen before any intervention result exists). Every excluded
cluster is named in the coverage report (§E4.10).

## E4.5 CorEx fits and factor ranking

For every (model seed s, cluster k, layer l ∈ {1,2,3}):

1. Run the classifier in eval mode on the cluster's discovery images; keep
   H₁, H₂, H₃ and pre-softmax logits L.
2. **[rev] Logits are row-centered before any use:** L̃_c = L_c − (1/10)Σ_u
   L_u per image. Softmax is invariant to adding an image-dependent
   constant to every logit, so the common logit mode is behaviorally
   irrelevant, yet raw-logit associations can shift with it; centering
   removes exactly that component (a deduction from the softmax definition,
   not a heuristic). Centering applies to the output-aware fit input, the
   post-hoc associations, and the univariate control alike. **[rev3]** (The
   round-1 wording claimed that standardizing each centered column by its
   own spread afterwards only mixes a small residual of the mode back in —
   that is mathematically incorrect and is withdrawn: row-centered logits
   satisfy Σ_c L̃_c = 0 by construction, and after column standardization
   X_c = (L̃_c − μ_c)/s_c the ten columns still satisfy Σ_c s_c X_c = 0
   exactly, so the linear dependence persists regardless of scaling.)
3. Standardize every hidden coordinate and centered logit **using only
   that cluster's observations**. Coordinates with variance < 10⁻¹² are
   flagged and **excluded from the fit** (original indices retained in
   metadata; `n_valid_hidden` / `n_excluded_hidden` recorded).
4. **Output-aware condition (primary):** V = [H | L̃-centered-standardized]
   (logits included but never true labels — the label-free and
   output-aware framing). **Hidden-only condition (required ablation):**
   V = H.
5. Linear CorEx, M = 10 requested factors, deterministic fit seed derived
   from (model seed, cluster, layer, condition) via the hash convention.
6. **[rev] Retain up to five factors by explained TC descending, among
   factors with TC > 0.** Ties in TC are broken by the program's neutral
   tie rule (seeded permutation before the descending sort, applied
   consistently to support construction as well). A condition with no
   positive-TC factor defines no interventions: the shortfall is recorded
   in coverage (§E4.10), never padded with noise factors. Factor ranking is
   never by ablation effect.
7. Store, per retained factor: fit seed, convergence, TC (bits), all
   factor-to-hidden MI scores (full-width vector, bits — enabling the
   §E4.12 Reading 3 analyses), all factor-to-logit MI scores (bits),
   dominant associated output class c* = argmax_c I(Zⱼ; L_c), the gap
   between the two largest logit MIs, valid/excluded count, and the
   **effective number of units** eⱼ **[rev]**: exp of the entropy of the
   normalized positive hidden-MI weights — ≈1 for one dominant unit plus a
   weak tail, ≈k for k equal contributors.

Two estimator notes:

- **Logit-association source is recorded.** In the output-aware condition
  the factor-to-logit MIs are fit-side association scores
  (`logit_mi_source = "fit"`). In the hidden-only condition they are
  **post-hoc** estimates: factor scores Z (the linear latents of the fitted
  model on the same standardized cluster sample) against each standardized
  centered logit column, with the Gaussian MI −½log(1 − ρ²) — the same
  estimator family as the fit-side scores (`logit_mi_source = "post_hoc"`).
  The two sources are never pooled in summaries.
- **Units:** fit-side TC/MI are converted to **bits at the fit boundary**
  (the Experiment 3 lesson, §E3.11 run-record correction); supports are
  unaffected (cumulative fractions are scale-invariant).
- **Scores are model-based quantities, not exact information measures
  [rev3].** The ten (row-centered, standardized) logit columns remain
  exactly linearly dependent (Σ_c s_c X_c = 0 above), and at H3 the logits
  are deterministic linear functions of the hidden activations, so the
  augmented fit input is rank-deficient independently of centering. Linear
  CorEx never inverts the joint covariance of the inputs — it updates
  per-factor weights from per-variable Gaussian moments — so the dependence
  does not break the fit, and every executed fit completed with convergence
  recorded. The TCs and MI scores it reports are nevertheless variational
  and Gaussian model-based association scores under the fit's conditioning
  family; they are read as such, never as exact information quantities of
  the empirical joint distribution. (The estimator emits its
  near-singular-covariance advisory on a minority of fits; this occurs in
  both conditions, so it is not attributable to the logit block alone, and
  fits still report convergence.)

## E4.6 Hidden-unit supports

Supports are constructed from **factor-to-hidden MI only**; logit columns
never enter a deletion support. Cumulative-MI thresholds q ∈ {0.60, 0.70,
0.80, 0.90} in the output-aware condition, q = 0.80 only in the hidden-only
condition. Supports are nested by construction, with MI-score ties broken by
the same seeded permutation as factor ranking **[rev]**, and stored with
requested and achieved fractions, size sⱼ(q), density sⱼ(q)/d_l (d_l ∈
{500, 400, 300}), ordered hidden-unit indices, and MI scores in bits
(`mnist_supports`). Degenerate factors (total MI ≤ 10⁻¹²) get empty,
flagged supports and define no interventions. **[rev3] Precision:** the
supports in this stage are built from the fit's **full-precision
in-memory** MI scores; the stored `hidden_mi_vector` serializes the same
vector rounded to 6 decimals for display and reuse. Quantified (by a
deterministic refit of all 300 output-aware fit conditions, §E4.13):
6-decimal rounding would shift 15 of 3,000 reconstructed extension
supports, by 1–4 units each (13 at q = 0.99, 2 at q = 0.95); every other
support is unchanged. **[rev5] Closed:** full-precision vectors are now
stored (`results/factor_hidden_mi_full_precision.npz`, saved by
`reeval_extension_fullprecision.py`, which also verified the regenerated
primary supports match the stored ones exactly and re-evaluated the 15
affected extension families with their matched controls — §E4.14.2);
the earlier "listed as follow-up" state belongs to the round-3 execution
history, and the corrected extension values used here are the
full-precision ones.

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
units by |corr(Hᵢ, L̃_c*ᵢ)| within the same discovery cluster — on the
**centered** logits **[rev]**, ties → lower index — one per LC support
size. Output-aware condition only (hidden-only factors take their c* from
the post-hoc estimates; the spec defines the univariate control for the
primary condition, and the hidden-only arm runs at q = 0.80 with random
controls only — **[rev]** accepted: this limits which comparisons that arm
can answer, and §E4.12 Reading 4 claims nothing beyond it).

**Random-control probability:** one-sided empirical ĥ_p = (1 + Σ 1[g_r ≥/≤
g_LC]) / (R + 1), reversed inequality for entropies; R = 20 makes these
coarse — read descriptively, never as p-values.

**Size matching controls unit count, not removed activation magnitude
[rev].** Mean |h_l,i| over the test set is recorded for every ablated
support (`mean_removed_activation_mag` per intervention), so LC-vs-random
removal magnitudes are always visible. Predeclared sensitivity trigger: if
the median LC-vs-random per-stratum (layer × threshold) difference exceeds
a factor of 1.5, an activity-matched random control arm is run as
supplementary before any aggregate summary is written; otherwise the
size-matched design stands as primary.

## E4.9 Evaluation and metrics

Every intervention is evaluated on all 10,000 test images. For true digit y,
the class-mean probability vectors give the signed shift Δ_{y,c} =
p̄ᴬ_{y,c} − p̄⁰_{y,c} (full 10×10 matrix stored per intervention). Aggregates:

- **Effect/degree:** T_y = ½Σ_c|Δ_{y,c}|; T_max, T_weighted.
- **Input-class concentration:** r_y = T_y/ΣT_u; H_input = −Σ r_y log r_y /
  log 10, D_input = max_y r_y.
- **Output-mass concentration:** q_{y,c} = |Δ_{y,c}|/Σ_u|Δ_{y,u}|; H_output,
  D_output at y* = argmax T_y; c⁻/c⁺ = argmin/argmax_c Δ_{y*,c}.
  **[rev] Interpretive bounds (exact):** each Δ_{y,·} sums to zero over c,
  so D_output ≤ 0.5 and (for nonzero effects) H_output ≥ log 2 / log 10 ≈
  0.301. Values are interpreted against these bounds, not against a
  [0, 1]-free-scale intuition.
- **Cancel-free individual effects [rev]:** U_y = mean over images of class
  y of ½||p_iᴬ − p_i⁰||₁ (U_y ≥ T_y always, by the triangle inequality),
  U_max/U_weighted, and prediction-flip rates (argmax change, overall and
  per class). A small T_y alongside a large U_y means substantial
  individual effects with little net directional change — invisible to the
  class-mean metric. These are accumulated during evaluation; per-image
  probability vectors are not stored.
- **Directional concordance (descriptive):** 1(c⁻ = c*ⱼ), the rank of c*ⱼ
  among Δ_{y*,c} ordered most-to-least negative, and **[rev]** the signed
  and absolute shift at the associated class, |Δ_{y*, c*ⱼ}|. Two cautions,
  fixed: selecting c* = argmax_c I(Z; L_c) identifies the *most associated*
  logit — the Gaussian-MI score depends on ρ² and is sign-blind, and
  zeroing a support is not a directed change of the latent factor — so the
  association does not by itself predict which direction the probability
  moves. A low reduction-concordance rate is therefore not counted against
  a factor's interpretability without the absolute-shift reading.
- **Accuracy family:** accuracy change, per-class recall change,
  confusion-matrix L1 change, mean cross-entropy change.

**Predeclared criterion (frozen):** *substantial and class-concentrated* ⇔
T_max ≥ 0.10 and H_input ≤ 0.50. Sensitivity table over T_max ∈ {0.05, 0.10,
0.15} × H_input ∈ {0.40, 0.50, 0.60} is predeclared as supplementary. Report
continuous metrics regardless of classification; no cutoff may be changed
after inspecting full-run outcomes.

**Storage:** `mnist_models` (training record), `mnist_clusters`,
`mnist_factors` (with full hidden-MI vectors), `mnist_supports`,
`mnist_ablation_summary` (one row per intervention), `mnist_probability_shifts`
(one row per intervention × y × c with class-mean baseline probability,
ablated probability, and signed change — parquet). One deviation from the
spec's wording is accepted (§E4.15, decision 4): the probability-shift table
stores **class-mean** probabilities (the p̄ vectors), plus the per-column
individual-effect summaries U_y and flip rates; per-image probabilities
remain reconstructible from checkpoints and are not informative for any
predeclared display.

## E4.10 Failure handling and coverage

Every degenerate outcome has a predeclared treatment; none is silently
dropped, and none can silently improve prevalence denominators:

1. **Nonconverged fits** (max_iter exhausted): convergence is recorded per
   fit (`corex_converged`); the fit's factors are analyzed as-is.
   Nonconvergence is a diagnostic, never an exclusion.
2. **Zero-total-MI / degenerate factors:** flagged
   (`support_degenerate`); empty supports define no interventions.
3. **Fewer than five usable factors:** conditions retain only factors with
   TC > 0, up to five; shortfalls are recorded in the coverage table.
   A zero-factor condition contributes to coverage only.
4. **Duplicate supports** (two factors whose supports coincide): permitted;
   their intervention keys remain distinct, and their random controls are
   regenerated per key.
5. **Exactly-zero-effect interventions** (all Δ_{y,c} = 0): classified
   `substantial_class_concentrated = False`; H_input is left undefined
   (NaN) — a zero-effect intervention can never become class-concentrated
   through an entropy convention — and its rate is reported in the coverage
   table.
6. **Coverage reporting:** per (model seed, layer, condition): the number
   of LC interventions, clusters present, zero-effect interventions, and
   factors retained; excluded tiny clusters are named; prevalence is always
   reported alongside its denominator, on the same table. Failed fits
   keep the condition visible with the failure flag (the program's
   failed-fits-are-data convention, §E3.7). **[rev3] Denominator
   convention:** prevalence is always computed over the interventions
   actually evaluated, which in the executed run equals all planned slots
   because no degenerate outcome fired (§E4.14); coverage reports planned
   vs retained counts explicitly so that a read-out over retained
   interventions alone (which could rise when unusable factors disappear)
   is never confused with a prevalence change.

**Uncertainty targets, stated separately [rev]:** with five model seeds,
per-condition results are reported per seed in addition to across-seed
summaries with small-sample intervals; these intervals describe
**training-seed variation conditional on this dataset and the fixed
discovery partition**. **[rev4] The cluster-sampling bootstrap promised in
the round-1 wording is removed:** a hierarchical scheme resampling
clusters within each resampled seed treats the shared twenty clusters as
nested, but the design is **crossed** — the same discovery clusters occur
under every model seed — and the implemented resampler (never called; now
deleted from the summarizer) was the nested form. No cluster-sampling
interval is reported; if one is wanted, the resampling must preserve the
shared cluster identities across seeds (resample the seed and cluster
factors separately — a crossed-design scheme). The 6,000
interventions are never treated as independent observations.

## E4.11 Grid and uncertainty

Primary grid: 5 model seeds × 20 clusters × 3 layers = **300 output-aware +
300 hidden-only CorEx fits** **[rev]**. Output-aware interventions: 300 ×
5 factors × 4 thresholds = **6,000 Local CorEx evaluations**, plus 120,000
matched random supports and 6,000 univariate supports. Hidden-only: 300 × 5
(when all five are usable) × 1 threshold = 1,500 LC + 7,500 random. Total
**141,000 intervention evaluations** (plus the fit stage). Conditions with
fewer than five positive-TC factors contribute fewer LC rows; the shortfall
is visible in coverage, so listed totals are ceilings.

The **model seed is the replicate** (n = 5): prevalence and paired
differences are computed per seed and summarized across seeds (mean, SE,
95% small-sample interval). **[rev4]** No cluster-sampling interval is
reported (the seed × cluster design is crossed; see §E4.10, uncertainty
targets).

**Feasibility pilot [rev]:** between smoke and full grid, the pilot is a
*feasibility* run (2 model seeds × all 20 clusters × all layers × one
threshold): its function is mechanics and computational feasibility only —
progression to the full grid depends on mechanics and budget, and any
design change during or after the pilot must be documented in §E4.13
before results are looked at. (The training smoke already inspected test
accuracy, 0.9864 vs the predeclared 97% flag; that threshold was fixed in
the original spec and no design change resulted from it.)

## E4.12 Pre-registered readings (outcome-neutral)

Six readings; the paper takes whichever the data support. Nothing here is a
result. **[rev3] Status boundary:** §E4.1–§E4.12 is the frozen primary
plan; §E4.13 is the execution record; §E4.14 reports the executed outcomes;
§E4.14.1–§E4.14.2 are explicitly exploratory extensions designed after the
outcomes were inspected. The test set remains held out from all fitting;
its intervention outcomes have nonetheless been inspected repeatedly since
the run, and no further primary claims were added from those inspections.

1. **vs. matched random:** LC supports produce larger (T_max, T_weighted,
   U_max, flip rate) and more class-concentrated (H_input, D_input) changes
   than size-matched random supports. Supported → recovery provides a
   useful input to the behavioral instrument relative to the specified
   controls. Unsupported → the neutral conclusion: "the discovered supports
   did not yield stronger behavioral selectivity under this deletion probe"
   — robustness, compensating pathways, or probe insensitivity are the
   candidate explanations, and no recovery claim is attached (§E4.1). Note
   that size-matched controls share unit count, not removed activation;
   the removal-magnitude diagnostic and its conditional sensitivity arm
   (§E4.8) accompany this reading.
2. **vs. univariate:** paired LC − univariate differences at matched keys —
   multivariate factor membership adds behavioral specificity beyond
   single-logit (centered) correlation ranking, or does not.
3. **TC, distributedness, and behavior [rev]:** the three-question reading,
   predeclared explicitly:
   - support-size and density distributions by layer and threshold;
   - the relationship between factor TC and intervention magnitude
     (Spearman within each layer × threshold stratum);
   - whether that relationship persists after adjusting for support
     size (Spearman between TC and the T_max-residualized-on-support-size
     values) — a TC effect beyond "more units selected";
   - the effective number of units (§E4.5), separating genuinely broad
     associations from one dominant unit plus a weak tail.
4. **Layer/condition structure:** intervention effects (magnitude,
   prevalence, concordance, support density) differ systematically by layer
   (H1/H2/H3) and by input condition. The output-aware vs. hidden-only
   comparison at q = 0.8 (including post-hoc-association concordance) is
   **fair in what it claims [rev]**: the two conditions can delete different
   numbers of units and their factors need not correspond by rank, so the
   comparison is made on each condition's **excess over its own matched
   random controls**, with support sizes reported alongside — never on the
   raw effect difference.
5. **Threshold dose-response [rev]:** nested supports guarantee that
   support size increases monotonically in q; whether deletion effects
   increase with q, and at what cost in specificity, is an empirical
   question — assessed on the predeclared q grid, not assumed.
6. **Concordance (descriptive, [rev]):** the associated class is the most
   reduced class (rate by layer and condition) — reported alongside the
   absolute shift at the associated class. The MI association is
   sign-blind and masking is not a directed latent change, so neither a
   high nor a low reduction-concordance rate alone licenses an
   interpretability claim; the pair of quantities is reported together.

The instrument-restraint statement (§E4.1) forbids any reading that
interprets prevalence or concordance as labeling an interaction type; a
supported Reading 6 is stated as directional influence, nothing more.

## E4.13 Implementation status and run record

- Reference implementation `tnnls_experiments/exp4_mnist/` (commit 01d1a40):
  spec-faithful; validation tests 1–13 covered (trunk equivalence,
  masking semantics, metric hand-checks, end-to-end smoke mechanics),
  test 14 (resume determinism of random masks) holds by construction and is
  asserted at port time.
- **Reference-implementation corrections (2026-11-02, pre-run):**
  1. Hidden-only condition computes the spec-required post-hoc
     I(Zⱼ; L_c) association (§E4.5).
  2. `mnist_probability_shifts` carries class-mean baseline/ablated
     probabilities alongside the signed change (§E4.9).
  3. Near-zero-variance hidden coordinates are excluded from the CorEx
     fit itself (spec §CorEx inputs, step 3); previously they were flagged
     but still fit.
  4. Dead code removed from the probability-shift expander.
- **Reviewer-revision changes (2026-11-03, pre-run, all validated by the
  test suite):** centered logits everywhere discovery-side logits are used
  (§E4.5); positive-TC-only factor retention with neutral TC and
  support-construction ties (§E4.5, §E4.6); U_y/flip-rate metrics and the
  associated-class absolute shift (§E4.9); effective number of units and
  full hidden-MI vectors stored per factor (§E4.5); removal-activation
  magnitude per intervention with the predeclared activity-matching
  trigger (§E4.8); fit-side TC/MI reported in bits at the boundary
  (§E4.5); summarizer additions (cancel-free paired comparisons,
  TC-magnitude link with support-size adjustment, effective-units table,
  coverage table, fair output-aware-vs-hidden-only excess comparison).
- Benchmarks at the real fit shape (n ≈ 500 cluster sample × 510 columns,
  M = 10): Linear CorEx fit 0.2 s, converged — the 600-fit fits-stage is
  minutes. Ablation grid ≈ 141k interventions × partial forward pass on 10k
  test images (trunk path): budgeted well under one GPU-hour on one RTX
  6000 (exact smoke/pilot timings below).
- Executed, in order (all before full-grid commitment): classifier training
  smoke (seed 5100) — **188 s** (84 epochs, early-stopped; test accuracy
  0.9864, above the 97% flag threshold; threshold was predeclared in the
  original spec — no design change resulted from this inspection); smoke
  rerun after the reviewer revision (1 seed × 3 clusters × layer 3 × both
  conditions × 3 random controls): 33 exp4 + 19 shared + 12 exp3-human
  tests pass; per-seed stages 6 s; all ten summary tables regenerate
  (`tc_magnitude` is empty on the smoke grid — its per-stratum n < 8
  minimum); feasibility pilot (2026-11-03; 2 model seeds × all 20 clusters
  × layers H1–H3 × q = 0.80, full random-control counts — **240/240 fit
  conditions completed**, every condition retaining all 5 positive-TC
  factors (no shortfalls), zero failed fits, zero zero-effect rows, no
  tiny-cluster exclusions (all 20 clusters ≥ 100 discovery obs); ~300 s
  per model seed, ~10.5 min total including the PHATE/k-means stage; all
  predeclared summary tables regenerate from pilot shards; the pilot's
  data were examined only for mechanics and coverage verification — no
  conclusions are read from them); **full-grid budget from measured
  rates: ≈ 1.5–2 GPU-hours serial for the 141k-evaluation grid, plus
  ~5 min for the three remaining model trainings**; then the full grid
  behind the user's GO.
- **Full grid executed (2026-11-03, user GO):** all five model seeds
  trained first (best-epoch test accuracies 0.9864 / 0.9860 / 0.9864 /
  0.9863 / 0.9862 — every seed above the 97% flag threshold, zero flags);
  the store was started clean (the pilot's single-threshold supports
  could not be resumed into the four-threshold grid without the
  audit-known config-mismatch hazard; pilot shards kept at
  `outputs/raw/mnist_pilot_2seed_q08`); **600/600 fit conditions with 5
  factors each (no shortfalls), zero nonconverged fits, 141,000/141,000
  intervention evaluations (6,000 LC output-aware + 120,000 output-aware
  random + 6,000 univariate + 1,500 LC hidden-only + 7,500 hidden-only
  random **[rev3]** — the previously listed "127,500 random" plus a
  trailing "7,500 random" double-counted the hidden-only controls and
  summed to 148,500), zero
  zero-effect rows, no tiny-cluster exclusions, and five per-seed
  probability-shift parquets of 2.82M rows each** (14.1M total). Wall
  time ~49 min (≈585 s per seed). All seven primary summary tables (plus
  coverage, effective-units, and TC-magnitude) regenerate from the
  shipped shards; results in §E4.14.
- **Results and displays (2026-11-03):** the summarizer gained per-threshold
  paired comparisons and the predeclared cutoff-sensitivity table
  (§E4.14); results committed with the human-pipeline record
  `tnnls_human_experiments/exp4_mnist_benchmark/` (results CSVs, model
  metadata, run report, resolved config) and rendered by
  `figures_exp4.py` → committed PNGs `fig01`–`fig06`; the exploration
  notebook `analysis_exp4_results.ipynb` (results-only dependencies) is
  executed and committed. The qualitative reconstruction panel (the
  spec's predeclared cluster rule, seed 5100) remains the one pending
  display.
- **Qualitative panel executed (2026-11-03):** the predeclared rule selects
  cluster 0 (the 4/9-mixed cluster, min(n₄, n₉) = 213); per layer the
  retained factor (seed 5100) with the largest max{I(Z;L₄), I(Z;L₉)} =
  0.17 / 0.86 / 0.78 bits **[rev3 recomputed, confirmed]**; the panel
  decodes symmetric factor-score perturbations (±1σ, ±2σ) of a fixed
  held-out cohort image through a **mirror autoencoder** — encoder = the
  frozen trained classifier, decoder = a mirror MLP trained for 18 s on
  the 45k training pixels (test reconstruction MSE 0.0215; the legacy
  program's `build_decoder_from_classifier` + AE mechanism,
  self-contained). The perturbations morph the decoded digit visibly
  along the factor direction (e.g. the H3 row slides cleanly between 4-
  and 9-like forms); the panel is explanatory and never participates in
  factor selection. Committed as
  `figures/exp4_qualitative_reconstruction_panel.png`
  (`decode_panel.py`, `train_mirror_ae.py`). **[rev3] Precise perturbation
  semantics** — **[rev4] corrected formula, panel regenerated:** the kept
  factor's loading s_h on the standardized hidden coordinates comes from a
  deterministic refit (the stored fit seed); the factor score on the
  layer's standardized hidden coordinates is z = s_h·h_std, linear in the
  raw activations with gradient g = s_h ⊘ σ_l; the minimum-Euclidean-norm
  raw-activation change achieving the score change δ is
  Δh = δ·g/‖g‖² — **the round-1..3 implementation used Δh = δ·σ_l ⊙ s_h/
  ‖σ_l ⊙ s_h‖², which achieves neither minimum-norm objective and
  overshoots the intended change by 2.0–3.9× on the plotted factors
  (measured); `decode_panel.py` now implements the minimum-norm solution
  with a runtime assertion that the achieved change equals δ, and the
  committed panel was regenerated with the corrected formula.** The score
  perturbed is the factor's **hidden-coordinate component**: the
  output-aware factor's logit loading is not controlled (logits respond
  only through the forward pass from layers 1/2). δ ∈ {0, ±1σ_Z, ±2σ_Z}
  in units of the factor-score std over the discovery cluster **where
  the score is the hidden-coordinate component (σ_Z = std of s_h·h_std
  over the discovery cluster, the quantity the code perturbs) — keeping
  the ±1σ/±2σ labels aligned with the corrected intervention [rev5]**;
  layer-1/2
  perturbations propagate forward through the frozen classifier, and the
  mirror AE always decodes from H₃ — its dimensions match by mirror
  architecture, which does not establish matched coordinate meanings, so
  the display is read as the decoder's response. The δ = 0 column is the
  unperturbed reconstruction, shown beside every perturbation.
- **Human-pipeline port complete (2026-11-03):**
  `tnnls_human_experiments/exp4_mnist_benchmark/` is now the run target in
  the house pattern (flat self-contained modules, local `common/`,
  package-local regenerable `run_results/`); the reference implementation
  stays as the spec record of the executed runs, and the port folds in the
  audit's remaining items: the stage-aware completeness gate (new stages
  can run after the main grid; hollow-run guard — the bare gate had
  blocked the profiles/extensions stages entirely), the repository-
  relative MNIST root under `data/` (decision #6), the `mnist_models`
  store table, and the duplicate-`load_splits` removal; the missing
  validation test 14 (resumed runs regenerate identical random masks) is
  now asserted. Port verified: 36/36 tests and a `--smoke` end-to-end run
  into its own outputs.
- **Simplification pass (2026-11-03):** dead machinery removed from the
  port (`evaluate_intervention` — the unused pre-trunk canonical path —
  `sensitivity_table` (superseded by `cutoff_sensitivity` in the
  summarizer), the one-line `concordance` helper, and unused imports);
  `tc_magnitude_relationship`'s unused `sup` argument dropped. Larger
  refactors identified for later: a shared per-(seed, cluster, layer)
  context for the four stage loops; a single-pass probability-shift
  expander.
- **All four simplifications implemented and re-run (2026-11-03, port
  only — the reference stays as the executed-run spec record):** (1) the
  four stage loops now share `LayerContext` + one
  `iter_cluster_layer_contexts` traversal (one model pass per cluster,
  identical representations across stages); (2) `write_probability_shifts`
  is a single pass over all shards (one parquet per seed); (3) the
  neutral-tie support mapping is one shared `mapped_supports`
  implementation (fit stage + extension can no longer drift); (4) factor
  rows now store the **exact validity mask and the factor's hidden-WS
  loading vector** — the extension stage uses the exact mask (no more
  MI≠0 reconstruction; the 1,158 logged reconstructs vanish). Two
  port-time bugs found and fixed by the re-run's own gates: the
  extension stage now pins its own threshold set (a full-grid config's
  primary thresholds skipped every row) and the evaluator never writes
  an empty shard (an empty completion had gated off legitimate retries).
  **Output-identity verification: the re-run (fresh store, 600 fits +
  141,000 interventions + 66k extension + 768k profile rows + 5 shift
  parquets; ~75 min wall) reproduces all thirteen committed summary
  tables with max |difference| = 0.00e+00**; the extension's own numbers
  move by ≤ 0.002 (the reconstruction approximation's full effect —
  §E4.14.2's figures are unchanged at quoted precision). This is the
  codebase now exercising the stage-aware gate for real: the extension
  and profiles stages ran after the completed main grid via the normal
  CLI.
- **Round-3 recomputations (2026-11-03, from the committed re-run store):
  [rev3]** every number quoted in §E4.14–§E4.14.2 was recomputed directly
  from the shipped shards and committed results tables by
  `analysis_feedback_round3.py`. In the course of this, the deterministic
  refit of all 300 output-aware fit conditions (same stored fit seeds,
  same cluster representations) reproduced every stored 6-decimal
  hidden-MI vector exactly (max |rounded refit − stored| = 0.00e+00),
  independently confirming fit determinism; full-precision vs rounded MI
  changes 15 of 3,000 reconstructed extension supports by 1–4 units
  (§E4.6, §E4.14.2). The fit-estimator's near-singular-covariance advisory
  was observed on a minority of fits in **both** conditions, so it is not
  attributed to the logit block (§E4.5).
- **Round-4 closures (2026-11-04):** the qualitative panel was
  regenerated with the corrected minimum-norm perturbation formula
  (runtime-asserted achieved factor-score change = δ for every plotted
  row; committed PNG replaced); the full-precision hidden-MI vectors were
  saved (`results/factor_hidden_mi_full_precision.npz`), the regenerated
  primary supports were verified to match the stored primary supports
  exactly (0 mismatches over 1,500 factors × 4 thresholds), and the 15
  rounding-affected extension families were re-evaluated with their
  matched controls into the separate `extension_summary_fp` table (330
  rows; comparison CSV `results/extension_fullprecision_reeval.csv`);
  a deterministic PHATE + k-means refit reproduces the stored discovery
  labels and the stored test assignments at agreement 1.0000 over all
  10,000 observations each (routing check, decision #30). The unused
  nested cluster bootstrap was removed from the summarizer (decision #32).

## E4.14 Results

All numbers are from the executed full grid (§E4.13) — recomputed directly
from the committed store and tables for the round-3 revision **[rev3]**:
600/600 fits (all
converged, 5 positive-TC factors each), 141,000 interventions, zero
zero-effect rows, zero exclusions — the predeclared degenerate-outcome
machinery (§E4.10) never fired, so every prevalence below is over the full
5 × 20 × 5 = 500-intervention strata. Model-seed spread is small throughout
(SEs quoted); the five per-seed rows are available in the committed tables.

**Finding 1 — Reading 1 supported: recovery feeds the instrument, far beyond
size-matched random deletion.** Paired LC − random differences are favorable
at every metric × layer × threshold; per-seed values agree tightly (e.g.
T_max excess at H2, q = 0.9: per-seed means 0.170/0.184/0.162/0.168/0.147
across the five seeds **[rev3]** — corrects an erroneous 0.146 for seed
5100), random-mean 0.005–0.006 at the same sizes. Matched
random deletion of 50–130 units is behaviorally inert (T_max ≈ 0.006)
while LC supports of the same size move the classifier's class means by up
to **0.96** (largest single-cell probability shift **[rev3]**, corrected
from the previously quoted 0.98).
**Cancellation, precisely [rev3].** The quoted "T/U ≈ 0.93–0.97" is the
ratio of **maxima** (mean T_max/U_max per stratum, 0.94 at q = 0.6 rising
to 0.97 at q = 0.9); the ratio of **weighted averages** is lower
(T_weighted/U_weighted 0.84–0.88) — both are net-shift statistics, and a
ratio of maxima answers a different question than an average-of-classes
ratio. Reconciling with §E4.14.1's per-image analysis, at q = 0.8:
cancellation is *not* concentrated at the associated class (median
T_{c*}/U_{c*} = 0.95) or the discovery cluster's dominant digit (0.96);
it concentrates in the small-effect classes (median T_y/U_y 0.75 for the
remaining classes) and in a minority of factors — T_{c*}/U_{c*} < 0.8 for
25.5% of factors and < 0.5 for 4.7% (per-layer associated-class medians
0.82/0.98/0.96 at H1/H2/H3); the largest absolute gaps (U_y − T_y up to
0.017) sit at specific H1 factors (e.g. cluster 15/digit 9, cluster
0/digit 9, clusters 13–14/digit 5). So the class-mean metric understates
individual effects most where effects are small and at H1 — per-image
effects are real there (§E4.14.1) while net class-mean shifts are small
for those factors; this is the reconciliation, now measured rather than
asserted. The activity-matching trigger (§E4.8) never fired (stratum
median LC/random removed-activation ratios 1.00–1.02 **[rev3]**, corrected
from 1.01–1.03 — similar median removed magnitude, which does not rule
out all activation-related confounding, particularly within class and
cluster cohorts), so the size-matched comparisons are not grossly
confounded with removed magnitude. This is the empirical support for
identify-then-intervene on a real network (§E3.9's design argument, now
measured).

**Finding 2 — Reading 5: the dose-response has no specificity penalty.**
Support density rises with q at every layer (mean 0.08–0.12 at q = 0.6 →
0.25–0.34 at q = 0.9; mean sizes ~31–43 → ~100–147 units **[rev3]**
corrected ranges), T_max rises monotonically, and mean H_input falls
(e.g. H2: 0.75 → 0.56): taking more of the factor's MI yields stronger AND
more class-concentrated interventions. The spec's anticipated specificity
trade-off did not appear in this grid.

**Finding 3 — Reading 4 (layer structure): influence concentrates at
H2/H3; H1 is less concentrated and weaker on average — not silent
[rev3].** Prevalence under the conjunctive criterion increases with q
mostly at H2/H3 (q = 0.9: H2 0.376, H3 0.370; H1 ≤ 0.06 at every q). LC
absolute effects at q = 0.9: T_max 0.113/0.171/0.155 (mean H_input
0.76/0.56/0.59). A mean T_max of 0.113 is not a quiet layer: H1's low
prevalence reflects the conjunction (magnitude AND class concentration)
plus heterogeneity — at q = 0.9, 5.8% of H1 factors are strong and
class-selective while 34.8% are strong but broad (T_max ≥ 0.10, H_input >
0.50), and the per-image cohort analysis (§E4.14.1) shows real H1
effects. The high-threshold results reinforce the same distinction. The
interpretive content (revised): under this probe, class-concentrated
influence concentrates in the deeper layers; H1 factors act more
diffusely across classes on average. See the factor-level heterogeneity
display below.

**Finding 4 — Reading 2: the multivariate-vs-univariate contrast is local
to H3 [rev3 corrected ranges].** LC beats the univariate logit-association
control in paired differences at H3, with the gap growing in q: ΔT_max
+0.006 → +0.053 and ΔH_input −0.02 → −0.12 from q = 0.6 to 0.9. (The
previously quoted "+0.02–0.03 / −0.05–−0.08 across q" held only near q =
0.8.) At H1/H2 differences are small and sign-mixed across q: within
±0.021 in T_max and ±0.011 in H_input (corrected from the previously
quoted tighter ranges). Claim scoped exactly there: multivariate factor
membership adds behavioral specificity at the last hidden layer; elsewhere
the size-matched univariate ranking performs comparably.

**Finding 5 — Reading 4 (condition structure, fair comparison): logits in
the fit are behaviorally small [rev3 relabel + caveat].** On
excess-over-matched-random at q = 0.8 (support sizes ~79/51/57 units
output-aware, ~76/48/57 hidden-only), condition differences are ≤ 0.006 in
T_max excess and ≤ 0.012 in H_input excess — small relative to the
excesses themselves. The behavioral specificity does not depend on
including logits in the fit (which also keeps the label-free framing
uncompromised); this does **not** establish that the two conditions
discover equivalent factors.

**Finding 6 — Reading 3: the TC–magnitude association is positive after the
predeclared size adjustment, but weaker under a standard partial
correlation; supports are broadly distributed [rev3].** As executed, the
size adjustment was Spearman(TC, T_max **residualized on support size by
linear regression**) — a residual-on-magnitude adjustment, not a standard
partial correlation (which removes the covariate from both variables, on
ranks for a partial Spearman). Raw Spearman(TC, T_max) is weak (−0.18 to
+0.23 across strata **[rev3]**, corrected from +0.27); under that
residualized adjustment it is positive at every stratum (0.29–0.48) — the
original calculation, retained under its exact description; **[rev3]** under
a standard partial Spearman it is nonnegative at 11 of 12 strata (−0.01 to
+0.32, weakest at H1, q = 0.6; per-seed values vary with SD ≈ 0.06 across
the five seeds), and within support-size quartile bands the
median within-band Spearman is 0.12–0.42. These checks reduce the
size-related explanation but do not exclude it — the result is a modest,
method-sensitive association measured after the size adjustment, and it is
reported as "associated after the specified size
adjustment," never "size-mediated" (a causal reading stronger than the
measurement); "not an artifact of support size" **[rev4 wording]** would
exceed what the partial/band checks establish. Effective number of units
(factors) is 66–128 — no factor is a single dominant unit (≈1 would be);
H1 associations are broadest (~100 of
500 units). Two statements the evidence does support: factor-to-unit
associations are spread across many units, and deleting the selected
support changes classifier behavior. Together they do **not** establish
that the effect comes from distributed deletion rather than a few heavy
units — a focused head-vs-tail or singleton ablation would test that, and
is recorded as not run (§E4.15, decision #26).

**Finding 7 — Reading 6, descriptive: directional concordance tracks
layer [rev3 reference corrected].** Reduction-concordance (associated
class = most reduced class): H1 0.27, H2 0.75, H3 0.55 (output-aware;
hidden-only within ±0.05 everywhere). Reference **[rev3]**: the
committed runs' own matched random supports' most-reduced class agrees
with the parent factor's associated class at 0.10–0.13 per layer — an
empirical reference from the same strata, replacing the earlier flat
"chance is 0.10" (a uniform guess, since associated and most-reduced
classes need not have uniform frequency). With the absolute shift at the
associated class (0.022/0.068/0.047) rising and falling with it. Per
§E4.12 R6: descriptive directional-influence language only; MI
association is sign-blind and masking is not a directed latent change.

**Factor-level heterogeneity display (round-3 addition [rev3] — all
1,500 output-aware factors per threshold; no selection, no pooling).**
At q = 0.8, factors split into 15.3% **strong & class-selective**
(T_max ≥ 0.10 and H_input ≤ 0.50 — the predeclared criterion), 2.5%
**strong & broad** (T_max ≥ 0.10, H_input > 0.50), and 82.3% **weak
under this probe** (T_max < 0.10); by layer (L1/L2/L3): 6.0/20.4/19.4%
strong & selective, 4.0/0.8/2.6% strong & broad. The same split at q =
0.9: 5.8/37.6/37.0% strong & selective and 34.8/5.0/16.2% strong &
broad — the layer story of Finding 3 rests on class concentration, not
on H1 factors being inert. Discovery-side association strength
(dominant-class logit MI in bits) relates to magnitude and
concentration separately: full-set Spearman(assoc, T_max) = +0.60 vs
Spearman(assoc, H_input) = −0.57 at q = 0.8 — **[rev4]** present in
every layer × seed cell (assoc/T_max +0.32 to +0.77; assoc/H_input
−0.26 to −0.65 across the 15 layer-seed cells, n = 100 each), so the
pooled correlations are not a layer-mixing artifact; association
quartiles order mean T_max 0.018 → 0.140 and mean H_input 0.847 →
0.550 (Q1 → Q4); **[rev3→4 display scope]** the percentages above are
computed per factor (no pooling) and appear here as a text/table
display; `fig07` remains its own aggregate display (top-5
fit-cluster mean over seeds) — per-factor and per-seed views live in
the executed notebook (`factor_assoc`,
`profile_cohorts`, `profile_grid` tables).

**Sensitivity items (predeclared, closed):** the cutoff sensitivity grid
(T_max ∈ {0.05,0.10,0.15} × H_input ∈ {0.40,0.50,0.60}; 108 strata,
`cutoff_sensitivity`) is monotone in both cutoffs and never exceeds 0.47
prevalence at the loosest cutoffs — the primary (0.10, 0.50) point is not
a knife-edge choice; activity-matched arm not triggered (Finding 1;
stratum-median removal-magnitude ratios 1.00–1.02 **[rev3]**).

**Scoped conclusions (influence language only, per §E4.1/§E4.12):**

- On this network, cumulative-MI supports of top-ranked Local CorEx factors
  identify unit sets whose deletion yields substantial, class-concentrated
  prediction shifts relative to size-matched random and (at H3)
  univariate-association controls.
- The effects are threshold-dose-dependent without a specificity penalty
  through q = 0.9, class-concentrated influence concentrates at H2/H3
  (H1 acts more diffusely; the class-mass-free per-image view shows H1
  effects), and the effects are not materially dependent on fitting with
  logits included.
- Reading 6's concordance is directional evidence consistent with the
  factors' strongest logit association; by the §E3.9 restraint it remains
  an influence statement, not an interaction-type characterization.

### E4.14.1 Exploratory extension (2026-11-03, feedback round 2 — label: exploratory)

A second review pushed for factor heterogeneity and corrected the first
in-cluster analysis: the dominant-digit-class cut measured the class view
across the whole test set, the 0.86 "cluster-proportional null" conflated
discovery-cluster purity with predicted effect-mass distribution (it is
withdrawn), and the "factors spare their discovery cluster" reading is
retracted. The coarse class-mass cut survives only as a descriptive
distribution row (round-2 decision #9–15, §E4.15). The adopted analyses
(assignment frozen per decision #11; machinery: `test_clusters` +
`profiles` stages; validation tests) give, at q = 0.8
(`profile_cohorts`, `profile_grid`, `factor_assoc`) — **[rev3] numbers
recomputed from the committed tables (levels were previously mislabeled as
excesses; the association-prediction values were recomputed and
corrected):**

1. **Per-image effects reflect digit identity, more than cluster identity
   [rev4 wording — the earlier "digit-localized but not
   style-localized" headline claimed a negative
   style finding its contrasts were not designed to establish].**
   Mean per-image LC magnitude (levels): own cluster 0.036–0.047,
   same-digit-other-clusters 0.033–0.045, other digits 0.007–0.009
   (matched-random levels 0.002–0.009). The paired LC − matched-random
   excesses are: own 0.033–0.045, same-digit-other 0.032–0.042 (within
   ~0.005 of own, both conditions at every layer), other digits
   0.0035–0.0049 **[rev3]** (the earlier text quoted LC levels under an
   "excess" label and overstated the other-digits excess ~2×). The
   supported statement: **under the adopted cluster assignment, average
   effects were similar within the discovery cluster and among same-digit
   images outside it, and much smaller on other digits.** Whether the
   effect is digit- or style-driven at the factor level is not
   established by these average contrasts. Flip rates follow the
   same ordering (LC levels: own 1.0–3.6%, same-digit-other 1.2–3.6%,
   other digits 0.3–0.8%). **[rev3]** Paired own-vs-same-digit-other
   contrast (the average locality difference): +0.003/+0.003/−0.002 per
   image at L1/L2/L3 (output-aware; across-seed SE ≤ 0.002); a claim of
   practical equivalence would require a
   justified equivalence margin. The clusters are **not** independently
   validated as handwriting-style categories.
2. **The class-mass cut is consistent — implementation consistency is
   established; explanation is a separate, now-measured claim [rev3].**
   On cross-checked factors the per-image grids reproduce the class-mean
   delta matrices exactly (max |ΔT_y| = 0.0000): this establishes that
   the per-image and per-intervention views are the same computation —
   it does not by itself establish that cancellation explains the
   class-mass distribution. That explanation is quantified in Finding 1
   (§E4.14): per-class cancellation is mild for associated
   and dominant-digit classes (median T_y/U_y ≈ 0.95–0.96) and
   concentrates in small-effect classes and a minority of H1 factors —
   the same factors the per-image view exposes.
3. **Discovery-side association predicts held-out effects in an
   exploratory analysis designed after the outcomes were inspected
   [rev3 wording].** Strata defined only from discovery fits; the
   association strength is the factor's dominant-class logit MI in bits
   (model-based, discovery-side, label-free). Recomputed values
   **[rev3]** (corrected from 0.36 / 0.75 / 0.73 and the earlier
   quartiles): Spearman(assoc bits, own-cohort excess diff) = 0.32 /
   0.65 / 0.67 at H1/H2/H3; association quartiles order the own-cohort
   diff monotonically (e.g. H2: 0.0063 / 0.021 / 0.050 / 0.102 from Q1
   to Q4); magnitude and selectivity relationships reported separately:
   Spearman(assoc, own − other-digit excess) = 0.26 / 0.48 / 0.52.
4. **Within-digit cohort contrast for the mixed clusters [rev3, expanded
   with cohort counts and per-seed contrasts in round 4 [rev4]].**
   The three test cohorts partition the test set by construction (own =
   all images assigned to the fit cluster, all classes;
   same-digit-other and other-digits split the complement at the
   cluster's dominant digit), so they cannot overlap; the genuine concern
   is that "own" mixes modal-digit and secondary-digit images. Values
   below are paired LC − matched-random excesses (per-image weighted,
   output-aware; per-line `in_n`/`out_n` are the test-image counts of
   that digit inside/outside the fit cluster — **[rev4]** both counts
   now printed), with the per-seed paired
   within-digit contrast quoted. For the mixed clusters' **dominant**
   digits (well-populated cohorts) the contrast is small and
   **sign-varies across clusters**: cluster 0, digit 9 (in_n 485,
   out_n 524):
   0.010 vs 0.015 at H1 (contrast −0.005; per-seed −0.010 to +0.000
   **[rev5]** — the round-3 text quoted an erroneous range −0.003 to
   +0.000 that excluded −0.007 and −0.010; all quantities verified
   mutually consistent: the same statistic, factors and weights),
   0.025 vs 0.036 at H2 (−0.011), 0.091 vs 0.090 at H3 (≈ 0); cluster 1,
   digit 3 (in_n 274, out_n 736): +0.055 at H1 (per-seed +0.044 to
   +0.068), +0.005 at H2, +0.007 at H3;
   cluster 13, digit 3 (in_n 680, out_n 330): **−0.045** at H1 (outside
   larger; per-seed −0.060 to −0.020, consistent in sign), −0.014 at H2,
   −0.003
   at H3. For **secondary** digits the inside-cluster excess is larger
   at H1: cluster 0, digit 4 (in_n 281, out_n 701): 0.093 vs 0.042
   (contrast
   +0.051, per-seed +0.041 to +0.058 — consistent); cluster 1, digit 5
   (in_n 175, out_n 717): 0.145 vs 0.063 (+0.082, per-seed +0.073 to
   +0.090);
   cluster 13, digit 5 (in_n 346, out_n 546): 0.090 vs 0.029 (+0.061);
   cluster 15,
   digit 7 (in_n 28, out_n 1000): 0.128 vs 0.014 (+0.114) — small
   cohort; several
   secondary-digit cohorts are tiny (cluster 0, digit 5: in_n 10;
   digit 8: in_n 8; cluster 1, digit 9: in_n 12) and their large
   contrasts (+0.09 to +0.13 at H1) carry the corresponding sampling
   uncertainty despite the five seeds sharing the same few images.
   Contrasts narrow or close by H3 (cluster 0, digit 4: 0.085 vs 0.078).
   **[rev5] Final wording:** the within-digit contrasts differ by
   cluster and by layer — dominant- and secondary-digit contrasts have
   comparable magnitudes with opposite signs (cluster 1, digit 3:
   +0.055 vs cluster 13, digit 3: −0.045 at H1, against cluster 0,
   digit 4: +0.051) — so the supported summary is **within-digit
   response heterogeneity across clusters and layers, involving both
   dominant and secondary digits**: the small overall
   own-vs-same-digit-other contrast (item 1) coexists with these
   conditional differences and does not average them away into a
   finding of cluster-independence. This is "cohort-specific
   differences among" digit images — not style effects (clusters
   unvalidated as styles).

Assignment diagnostics (Phase A; **[rev3]** reported; **[rev4]** extended
to the full grids): the frozen Phase-A mechanics checks record a
1,000-observation self-transform agreement of 97.5% (permutation-tolerant),
test-image neighborhood k-NN consistency 0.966, median out-of-sample
nearest-neighbor distance 8.7 × 10⁻⁴ (`test_assignments_meta.json`).
**[rev4]** The two distinct identity checks are now separately verified,
over ALL observations: (a) a deterministic PHATE refit (seed 5200) +
k-means (seed 5201) reproduces the **stored discovery labels with direct
agreement 1.0000 on all 10,000 observations** — the label identities need
no permutation, so no permutation is applied and the routing of test
assignments to the factors' source-cluster IDs is direct; (b) the fresh
out-of-sample interpolation of all 10,000 test images reproduces the
**stored test assignments with agreement 1.0000** (the stored k-NN
consistency 0.966 remains the interpolation-quality diagnostic: test
images' nearest PHATE neighbor sits in the assigned cluster for 97% of
images, with boundary disagreements recorded per image). These are
mechanics checks — they do not independently validate the clusters as
handwriting-style categories.

Display: `fig07_profile_heatmaps.png` (two aligned heatmaps: fit-cluster
rows annotated with association strength; digit-class and held-out
evaluation-cluster columns; cell = LC − matched-random excess per-image
|Δ|, output-aware, top-5 fit-cluster mean over seeds; per-factor and
per-seed views in the notebook).

### E4.14.2 Supplementary threshold extension q ∈ {0.95, 0.99} (executed 2026-11-03; exploratory)

Executed per decision #14 (no q = 1.0: with nonnegative association
scores the support degenerates to the whole layer). Output-aware only;
supports reconstructed from the stored per-factor hidden-MI vectors with
the primary stage's machinery (documented approximation: vectors are
rounded to 6 decimals at the fit boundary; validity reconstructed as
"MI ≠ 0", which logs zero-MI-valid-unit mismatches in 1,158 of 6,000
factor rows, shifting the matched-random pool by at most a few units);
the frozen evaluator then ran to a separate store table — the primary
shards were never recomputed. 66,000 new rows (per factor-threshold: 1 LC + 20
random + 1 univariate; 1,500 factor conditions × 2 thresholds × 22 =
3,000 LC + 60,000 random + 3,000 univariate); zero zero-effect; wall ≈
13.5 min. **[rev4] Authoritative results:** authority is REVERSED from the
round-3 choice — the committed figures below are now the **re-run store's
extension (exact stored validity masks)** with the 15
rounding-affected families **re-evaluated at full precision**
(`reeval_extension_fullprecision.py`: full-precision hidden-MI vectors
saved to `results/factor_hidden_mi_full_precision.npz`; regenerated
primary supports match the stored primary supports exactly — 0
mismatches over 1,500 factors × 4 thresholds; the 15 changed extension
supports re-evaluated with their matched controls — 330 rows — into the
separate `extension_summary_fp` table; per-row comparison in
`results/extension_fullprecision_reeval.csv`: LC rows change by −0.017 to
+0.054 in T_max only where the support membership changed; 4 families
whose support size changed by one unit re-drew their random rows). The
quoted-precision claim is now stated exactly **[rev5]**, with the
comparison baseline identified: the figures quoted in the **round-3
document** (from the extension's first run — rounded vectors,
reconstructed validity) versus the **authoritative recomputation** (re-run
store + full-precision corrections, `analysis_feedback_round3.py` §K).
Changed at quoted precision: H_input at H2/q = 0.95 = 0.5372 → quoted
**0.54** (was 0.53); H3 LC-minus-univariate ΔT_max **+0.047–0.069** (was
+0.049–0.070); mean q = 0.99 support density **0.53** (was 0.52, medians
now 0.38 → 0.59, was 0.37 → 0.58). Unchanged at quoted precision: the
T_max means, prevalences, and all remaining §E4.14.2 values. Remaining approximations in the authoritative base: none for
validity (exact masks); support construction uses the **full-precision
vectors** for the corrected families and the 6-decimal-rounded stored
vectors elsewhere (verified identical except the 15 corrected cases —
§E4.6, §E4.15 decisions #24/#29). The first-run approximations
("MI ≠ 0" validity in 1,158 rows, rounded-vector supports throughout)
are retained above in the executed-run description for the record.

1. **Effects keep climbing steeply through q = 0.95** — mean T_max by
   layer at q = 0.95: 0.215 / 0.250 / 0.254 (H1/H2/H3; from 0.113 / 0.171
   / 0.155 at q = 0.9) — and prevalence peaks at 0.422 (H2) / 0.382 (H3)
   at q = 0.95 with specificity intact (H_input 0.54 / 0.57 **[rev4]**
   corrected at the quoted precision, see authoritative note). The
   response is NOT saturating at 0.9.
2. **At q = 0.99 specificity declines, especially at H3 — the
   anticipated trade-off finally appears [rev3 wording: "collapses" was
   stronger than the numbers; prevalence falls from 0.422 to 0.346 at
   H2, 0.382 to 0.150 at H3].** T_max still rises (0.45–0.53) but
   H_input goes back
   up (H2 0.57, H3 0.67, H1 0.80) and prevalence *falls* (H2 0.346, H3
   0.150, H1 0.000). The mechanism is visible in the supports:
   density rises to a mean 0.53 (medians: 0.38 → 0.59; mean sizes
   ~133 → ~208 units, medians 139 → 206 — full-precision recomputation
   **[rev4]**) — heavy weak-MI tails. **[rev3] Overlap, corrected**
   **[rev4 wording]:** under a
   same-size random reference the observed overlap is NOT substantially
   above chance — for two independent supports of the same sizes from
   the same valid pool the expected Jaccard is J ≈ p/(2−p) (0.35 at
   density 0.52; pair-level reference using actual sizes 0.38), so a
   median observed cross-factor Jaccard of 0.38 (p90 0.53; recomputed
   at full precision: 0.42 observed vs 0.38 reference within fits;
   cross-cluster pairs exactly at chance, 0.37 vs 0.37) is roughly what
   support size alone predicts. Near-complete supports are therefore
   large masks whose chance overlap is already high — **support size
   alone warrants the broad-lesion caution here, though it cannot by
   itself establish how factor-specific the deletion remains; the
   overlap-based mechanism claim is withdrawn** (the earlier
   "near-identical supports" reading was withdrawn in round 3). The
   self-Jaccard of a factor's 0.95/0.99 supports equals their size ratio
   (median 0.64 = s_0.95/s_0.99 — nested sets) and adds no evidence
   about functional specificity.
3. **Univariate contrast consistent with the primary grid [rev3 signs
   corrected]:** LC clearly better than the univariate control at H3
   (ΔT_max +0.047–0.069 **[rev4]**, ΔH_input **−0.047–−0.110** — the
   previously
   reported positive H_input signs were a transcription error; LC is
   larger AND more concentrated), tied or univariate-slightly-better at
   H2 (ΔT_max −0.031–−0.025 / ΔH_input ±0.01), tied at H1 (ΔT_max
   −0.013–+0.002).

Readout: **[rev4 wording]** the exploratory favorable range on this
network is q ≈ 0.9–0.95; at q = 0.99 the probe executes a **broad-support
deletion** (a cumulative pairwise-MI support never defines an isolated
factor intervention) **whose behavioral specificity is measured by the
reported outcomes, not established either way by the support
construction [rev5]**. Diagnostics in
`results/extension_support_diag.csv` and
`results/extension_fullprecision_reeval.csv` **[rev4]**.

## E4.15 Decision log (feedback of 2026-11-03 — all adopted)
| # | Decision | Status |
|---|---|---|
| 1 | Fit-side TC/MI units | **Bits**, conversion at the fit boundary, documented (§E4.5) |
| 2 | Tie policy | **Neutral tie rule** (seeded permutation), applied to TC ranking *and* support construction (§E4.5, §E4.6) |
| 3 | Tiny clusters | **n < 100 discovery obs excluded**, feasibility rule, complete coverage reporting (§E4.4, §E4.10) |
| 4 | Probability storage | **Class means** + streamed per-column individual-effect summaries (U, flips); per-image vectors not stored (§E4.9) |
| 5 | Hidden-only scope | **q = 0.8, random controls only; no univariate control**; post-hoc concordance included (§E4.8) |
| 6 | MNIST data location | **Repository-relative `data/`** at port time (§E4.2 note — reference keeps its cwd-relative root until the port) |
| 7 | Pilot | **Feasibility pilot** — mechanics/budget gate; pilot-initiated design changes documented first (§E4.11) |
| 8 | Global CorEx comparison | **Not added** — no claim that locality improves discovery is in scope (§E4.1) |

Adopted analysis/interpretation revisions beyond the six decisions:
centered logits (§E4.5), positive-TC retention rule (§E4.5), U_y/flip-rate
metrics and cancel-free paired comparisons (§E4.9, §E4.12), TC–
distributedness–behavior reading with the effective-unit diagnostic
(§E4.12 R3), fair output-aware-vs-hidden-only excess comparison (§E4.12
R4), softened unsuccessful-outcome language (§E4.1, §E4.12 R1), failure
handling and coverage table (§E4.10), separated uncertainty targets with
per-seed display (§E4.10), output-concentration bounds (§E4.9),
dose-response empirically assessed (§E4.12 R5), concordance read jointly
with absolute shift (§E4.12 R6), removal-activation diagnostic with
conditional activity-matched sensitivity arm (§E4.8), grid arithmetic
corrected (§E4.11).

**Round-2 adoptions (2026-11-03, second review of §E4.14's first cut; all
adopted):**

| # | Decision | Status |
|---|---|---|
| 9 | In-cluster re-interpretation | The dominant-digit analysis is **retracted to a coarse descriptive cut** (shift-mass on the discovery cluster's dominant digit class, distributions only); the 0.86 "cluster-proportional null" is withdrawn as unjustified (cluster purity ≠ predicted effect-mass distribution), and "factors spare their discovery cluster" language is dropped |
| 10 | Factor-heterogeneity program | **Adopted (exploratory):** per-factor behavioral profiles before aggregation — per-image a_f(x) = ½‖·‖₁ summarized over the 20 evaluation clusters × 10 true classes; held-out cohort effects (own cluster / same digit other clusters / other digits) with cohort sizes and per-image averages; association treated as a separate axis (strength in bits, gap, never argmax alone); cluster locality as annotation |
| 11 | Test-cluster assignment rule (Phase A, frozen) | PHATE refit deterministically (seed 5200) on discovery; fitted k-means labels discovery; `kmeans.predict` on the PHATE out-of-sample transform assigns test images; PHATE's built-in pre-fit-transform warning anticipated and answered by assignment-behavior checks run and reported before any effect is read |
| 12 | Profiles scope (Phase B) | q = 0.80, both input conditions, LC + matched-random profiles; grids stored per model seed; heatmaps seeded |
| 13 | Stratified readout (Phase D) | Selection on discovery-side association strength/gap ONLY (independent of intervention outcomes), exploratory label, controls on the same cohorts |
| 14 | q extension | **[0.95, 0.99] as a supplementary, post-hoc-documented extension** (no q = 1.0 — with nonnegative association scores the support degenerates to the whole layer, a layer wipe whose controls degenerate identically); at q = 0.99 report support density and cross-factor support overlap (Jaccard, post-hoc from stored supports) |
| 15 | Round-1 coarse in-cluster row | Kept as one descriptive paragraph (skewed r_dom distribution), never a finding |

Experiment 4 records: the pre-registered grid was run in full and
reviewed five times; the readings, the layer-level perturbation statement,
and the cross-experiment cross-references are recorded in this document
(§E4.12–§E4.16 **[rev3]**, replacing the nonexistent "E4.4–E4.4.4" range)
and the committed results directory.

**Round-3 adoptions (2026-11-03, third review; every item first verified
against the code and the committed tables, then adopted — decisions
#16–27):**

| # | Item | Resolution |
|---|---|---|
| 16 | Cancellation reconciliation | Quantified: maxima-ratio and weighted-average ratios defined; per-class and per-factor distributions computed; Finding 1 reconciled with §E4.14.1 (cancellation lives in small-effect classes and a minority of H1 factors) |
| 17 | Logit-standardization claim | Round-1 explanation withdrawn as mathematically incorrect (column scaling preserves Σ_c s_c X_c = 0); corrected dependence statement + model-based-scores note (§E4.5, estimator note 3) |
| 18 | Extension univariate signs | Confirmed transcription error; ΔH_input signs corrected to negative at H3 (§E4.14.2) |
| 19 | Jaccard reference | Same-size random reference computed from actual support sizes and valid pools: overlap ≈ chance (0.42 observed vs 0.38 reference within fits; cross-cluster at chance); "near-identical supports" and the overlap-explains-specificity mechanism withdrawn; self-J = size ratio noted (§E4.14.2) |
| 20 | TC–magnitude adjustment | Original calculation retained under its exact description (residualized on size); standard partial Spearman and size-band sensitivity added; "within a fixed support size" and "size-mediated" removed (§E4.14 F6) |
| 21 | Factor-level display | All-factor heterogeneity fractions (strong & selective / strong & broad / weak) added to the main analysis; association's separate links to magnitude and concentration reported (§E4.14, round-3 display; Finding 3 reinterpreted) |
| 22 | Cluster-locality wording | Narrowed to the average contrast with paired uncertainty; cohort masks defined explicitly; within-digit contrast for mixed clusters added; Phase-A assignment diagnostics reported (§E4.14.1) |
| 23 | Distributed-deletion claim | Narrowed to the two supported statements; head-vs-tail ablation identified as the test of the stronger claim, not run (§E4.14 F6) |
| 24 | Stored MI precision | Semantics documented: primary supports built at full precision (verified by deterministic refit of all 300 output-aware conditions, exact vector reproduction); extension reconstruction from the rounded vectors shifts 15/3,000 supports by 1–4 units; full-precision storage listed as follow-up (§E4.6) |
| 25 | Grid arithmetic | §E4.13 component list corrected (120,000 output-aware + 7,500 hidden-only random; total 141,000) |
| 26 | Head-vs-tail ablation | Not run this round; the narrowed wording (decision #23) is the adopted resolution — the focused ablation remains available as a supplementary benchmark if the stronger claim is wanted |
| 27 | Number corrections | Per-seed excess (0.170 for seed 5100), max single-cell shift (0.96), dose-response density ranges, univariate paired ranges (primary and extension), profile-cohort levels-vs-excess labels, association-prediction correlations recomputed, activity-ratio range, panel association values confirmed (§E4.13–§E4.14.2) |

Adopted tightening items (round 3): removal-magnitude ratios reported as
similar-median evidence only (not a rule-out of activation confounding);
concordance read against the measured matched-random reference
(0.10–0.13) instead of the uniform guess; "prospectively predicts"
replaced by the exploratory-design disclosure; "specificity collapses"
replaced by "specificity declines at q = 0.99, especially at H3";
output-aware-vs-hidden-only similarity explicitly scoped to aggregate
behavior, not factor equivalence; prevalence denominator convention
stated (§E4.10); qualitative-panel perturbation semantics specified and
its decoder caveat recorded (§E4.13); pre-registration boundary stated
(§E4.12).

**Round-4 adoptions (2026-11-04, fourth review; every item first verified
against the code and the committed tables, then adopted — decisions
#28–34):**

| # | Item | Resolution |
|---|---|---|
| 28 | Panel perturbation formula | Confirmed an implementation error: the coded direction σ⊙s_h/‖σ⊙s_h‖² achieves neither minimum-norm objective and overshoots the intended score change 2.0–3.9× on the plotted factors (measured). `decode_panel.py` now implements the minimum-Euclidean-norm raw-activation solution Δh = δ·(s_h⊘σ_l)/‖s_h⊘σ_l‖² with a runtime assertion (achieved change = δ, checked per plotted row); the panel is regenerated; the score perturbed is specified as the factor's hidden-coordinate component (§E4.13) |
| 29 | Extension authority + rounding closure | Authority reversed to the re-run store's exact-mask extension; full-precision hidden-MI vectors saved (`factor_hidden_mi_full_precision.npz`); regenerated primary supports match the stored ones exactly (checked, 0 mismatches); the 15 rounding-affected families re-evaluated with matched controls (330 rows, `extension_summary_fp` + comparison CSV); §E4.14.2 numbers updated; the quoted-precision claim now explicitly checked — one quoted value corrected (H2/q = 0.95 H_input 0.53 → 0.54) |
| 30 | Cluster-identity ambiguity | The two distinct checks separated and both run over ALL observations: deterministic refit reproduces the stored discovery labels at direct agreement 1.0000 (10,000/10,000 — no permutation exists or is applied; routing direct), and the fresh test interpolation reproduces the stored test assignments at agreement 1.0000 (§E4.14.1) |
| 31 | Mixed-cluster counts/wording | Cohort image counts reported for every highlighted comparison; LC-levels vs excesses labeled; per-seed paired contrasts quoted; "style-driven membership shapes…" replaced by "cohort-specific differences among secondary-digit images"; the item-1 headline narrowed to the average-contrast wording (§E4.14.1) |
| 32 | Cluster bootstrap | The unused nested resampler removed from the summarizer (dead code); the crossed-design caveat recorded; no cluster-sampling interval reported (§E4.10, §E4.11) |
| 33 | Wording overreach | "Not an artifact of support size" → modest, method-sensitive adjusted association (size-related explanations reduced, not excluded); q = 0.99: support-size caution only, factor-specificity untested; "whole-factor deletion neighborhood" → "broad-support deletion"; "operational sweet spot" → exploratory favorable range on this network (§E4.14.2, Finding 6) |
| 34 | Presentation | `fig07`'s top-5-mean scope clarified vs the per-factor heterogeneity display; per-layer, per-seed association correlations added (assoc/T_max +0.32–0.77, assoc/H_input −0.26–−0.65 across the 15 layer-seed cells — the pooled +0.60/−0.57 is not a layer-mixing artifact) |

**Round-5 adoptions (2026-11-04, fifth review; every item first verified
against the code and the committed tables, then adopted — decisions
#35–39):**

| # | Item | Resolution |
|---|---|---|
| 35 | Cohort-contrast consistency | The cluster 0/digit 9 H1 quantities verified mutually consistent (inside/outside excesses, their difference −0.005, and the mean of the per-seed contrasts −0.0047 all use the same statistic, factors, and weights); the doc's quoted per-seed RANGE was the sole error (−0.010 to +0.000, corrected from "−0.003 to +0.000"; the other quoted ranges re-verified) |
| 36 | Mixed-cluster interpretation | "Digit-general behavior for dominant digits" removed; the supported summary is **within-digit response heterogeneity across clusters and layers, involving both dominant and secondary digits** (comparable magnitudes, opposite signs); `out_n` added wherever `in_n` is quoted |
| 37 | Change-summary accuracy | The blanket "every other number unchanged" replaced by the explicit, baseline-identified list of quoted-precision changes (H2/q = 0.95 H_input 0.53 → 0.54; H3 univariate ΔT_max +0.049–0.070 → +0.047–0.069; mean q = 0.99 density 0.52 → 0.53, medians 0.38 → 0.59) (§E4.14.2) |
| 38 | Small edits | §E4.6 now states the COMPLETED full-precision storage and re-evaluation (the follow-up wording moved to execution history); §E4.14.2's ending no longer implies construction-based non-specificity ("broad-support deletion whose behavioral specificity is measured by the reported outcomes"); σ_Z pinned to the hidden-coordinate score the code perturbs (§E4.13) |
| 39 | Methods vs history | The account of record — the final procedure and results in one consolidated reader-facing statement — added to §E4.16, separating the current state from the round-by-round revision trail |
## E4.16 Experiment 4: what the pre-registered readings gave (the synthesis)

Readings: (1) matched-random control — strongly supported; (2) univariate
control — layer-dependent, supported only at the deepest layer; (3)
TC–distributedness–behavior — positive after the size adjustment,
method-sensitive in strength **[rev3]**; (4) layer/condition structure —
supported: layers separate sharply in class concentration, and the fair
condition comparison at q = 0.8 finds logits behaviorally small; (5)
dose-response — no specificity trade-off until q ≈ 0.95 **[rev3]**, with
specificity declining (not collapsing) at q = 0.99, especially at H3;
(6) concordance — descriptive, tracks layer, now read against the
measured random reference **[rev3]**. The per-image cohort summary
(§E4.14.1) is an exploratory extension, not one of the six readings
**[rev3]**. Word-of-record
is the influence formulation: the pipeline's factors are recovered,
supported, and their perturbation effects are class-structured — but the
instrument signs effects, never interaction type.

**[rev5] The account of record (final procedure and results, separated
from the review history).** One consolidated statement of what exp4
established, independent of the revision trail above:

*Procedure (final).* Canonical MNIST, seeds 5100–5104 (test accuracy
.9860–.9864, all above the 97% flag); 20 PHATE (seed 5200) / k-means
(seed 5201) discovery clusters from 10,000 held-out discovery images
(assignment verified: a deterministic refit reproduces the stored
discovery labels and the test-image assignments at agreement 1.0000 over
all 10,000 each); per (seed, cluster, layer) Linear CorEx fits on
standardized hidden activations (output-aware adds row-centered
standardized logits), top-5 positive-TC factors; cumulative-MI supports
at q ∈ {0.6, 0.7, 0.8, 0.9} (supplementary 0.95/0.99) built from the fit's
full-precision MI (stored full-precision and rounded vectors verified
mutually consistent; regenerated primary supports match the stored
exactly); matched-random (20×), univariate (OA), and hidden-only (q =
0.8) controls; the frozen evaluator (activation masking, trunk path,
eval mode) on all 10,000 test images; 141,000 + 66,000 intervention
evaluations, zero failures (store tables `mnist_*`;
`analysis_feedback_round3.py`, `reeval_extension_fullprecision.py`
reproduce every number below).

*Results (final, all [rev5]-verified from the authoritative store).*
(1) LC supports beat size-matched random deletion on every metric at
every layer and threshold (e.g. per-seed H2/q = 0.9 T_max excesses
0.170/0.184/0.162/0.168/0.147 vs random ≈ 0.006); matched-random
deletion of the same sizes is behaviorally inert. (2) The dose-response
through q = 0.95 raises T_max (0.113/0.171/0.155 at q = 0.9 → 0.215/
0.250/0.254) without a class-concentration penalty (mean H_input falls);
at q = 0.99 specificity declines, especially at H3 (prevalence
0.382 → 0.150) — large broad-support masks (≈208 units, mean density
0.53) whose chance overlap is already high (observed cross-factor
Jaccard ≈ its same-size random reference). (3) Layer structure: at q =
0.9 prevalence passes the conjunctive criterion mostly at H2/H3
(0.376/0.370 vs H1 ≤ 0.06) — but H1 is not inert: 34.8% of H1 factors
are strong-but-broad (T_max ≥ 0.10, H_input > 0.50) and per-image
effects are real; the layer signal is class concentration, not
magnitude. (4) LC beats the univariate control where the gap grows (H3:
ΔT_max +0.006 → +0.053, ΔH_input −0.02 → −0.12 across q); at H1/H2 the
differences are within ±0.021/±0.011, sign-mixed. (5) Including logits
in the fit changes neither excess meaningfully (≤ 0.006 in T_max
excess) and keeps the framing label-free. (6) TC and intervention
magnitude are associated after the size adjustment (partial Spearman
−0.01–0.32; size-band medians 0.12–0.42); association strength relates
separately to magnitude (+0.32–0.77 per layer-seed cell) and
concentration (−0.26–−0.65). (7) Directional concordance tracks layer
(0.27/0.75/0.55) against a matched-random reference of 0.10–0.13.
(8) [Exploratory] Per-image effects: same-digit inside ≈ outside the
discovery cluster on average (paired contrast ≤ 0.004, SE ≤ 0.002), with
within-digit response heterogeneity across clusters and layers
(contrasts +0.05 to +0.11 and −0.05 for dominant- and secondary-digit
cohorts at H1); discovery-side association predicts held-out excesses
(Spearman 0.32/0.65/0.67). All claims are influence language (§E4.1);
interaction type is never signed.

## E4.17 Grid summary and where everything lives

Pre-registration (E4.1-E4.11) — fixed k-means/PHATE clusters, the
per-cluster intervention grid, matched random/uniform controls, the
layer-wise reading — all applied as written; deviations are minimal and
listed where they occur. The record: this document's §E4.12–§E4.14
(readings, execution record, results; exploratory extensions in
§E4.14.1–§E4.14.2) plus the committed
`results/` tables **[rev3]** (cross-reference corrected from the
nonexistent "§E4.12-§E4.12.2"); the figures and the profile notebook are in the
exp4 package; the reference implementation (`tnnls_experiments/exp4_mnist/`
**[rev3]**, correcting `reference/exp4_mnist/`)
stays as the spec record of the run. The round-3 recomputations live in
`analysis_feedback_round3.py` in the same package **[rev3]**; the
round-4 precision closure lives in `reeval_extension_fullprecision.py`
**[rev4]**.
