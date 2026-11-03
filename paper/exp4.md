# Experiment 4: factor-ablation benchmark on MNIST — the behavioral-probe instrument on a real network

Companion to `simulation_methodology.md` (Experiments 1–2) and `exp3.md`
(Experiment 3). This document is the Experiment 4 pre-registration: motivation,
methodology, grid, and outcome-neutral readings — modeled on the Experiment 3
document. §E4.13 records implementation status; §E4.14 (results) is written
only after the executed run. Reference implementation
`tnnls_experiments/exp4_mnist/`; the human-pipeline package is the run target
and will implement any redesign agreed below.

**Feedback record:** an external review (2026-11-03, `exp4_feedback.md`)
endorsed the design and requested interpretive/analysis revisions. All were
adopted; they are marked **[rev]** at the places they changed, and the
decision log (§E4.15) records every adopted recommendation. The reviewer's
unverified claims (equivalence tests, convergence, runtimes) were verified
during this revision: 64 tests pass and the smoke run reproduces; see
§E4.13.

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
   post-hoc associations, and the univariate control alike. (Standardizing
   each centered column by its own spread afterwards mixes a small residual
   of the mode back in and is recorded here — no exact singularity results.)
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

## E4.6 Hidden-unit supports

Supports are constructed from **factor-to-hidden MI only**; logit columns
never enter a deletion support. Cumulative-MI thresholds q ∈ {0.60, 0.70,
0.80, 0.90} in the output-aware condition, q = 0.80 only in the hidden-only
condition. Supports are nested by construction, with MI-score ties broken by
the same seeded permutation as factor ranking **[rev]**, and stored with
requested and achieved fractions, size sⱼ(q), density sⱼ(q)/d_l (d_l ∈
{500, 400, 300}), ordered hidden-unit indices, and MI scores in bits
(`mnist_supports`). Degenerate factors (total MI ≤ 10⁻¹²) get empty,
flagged supports and define no interventions.

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
   failed-fits-are-data convention, §E3.7).

**Uncertainty targets, stated separately [rev]:** with five model seeds,
per-condition results are reported per seed in addition to across-seed
summaries with small-sample intervals; these intervals describe
**training-seed variation conditional on this dataset and the fixed
discovery partition**. The hierarchical bootstrap (resampling seeds, then
clusters within seeds) addresses the cluster-sampling target — a different
question — and is reported only where relevant, labeled as such. The 6,000
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
95% small-sample interval). For factor-level distributions, the
hierarchical bootstrap resamples seeds, then clusters within seeds (§E4.10,
uncertainty targets).

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
result.

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

## E4.14 Results

*(written after the executed run; nothing claimed here now)*

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