# Local CorEx: Learning Context-Dependent Structure in Data and Neural Representations

**Draft v0.4 (2026-11-05).** Internal draft for review and external
AI analysis. Sources: the committed experiment documents
(`paper/exp3.md`, `paper/exp4.md`, `paper/exp4_addendum_global_arm.md`,
`paper/exp5.md`), the claim audit (`paper/claim_evidence_table.md`),
and the committed results CSVs. Every quantitative display is
regenerated from committed result files by
`paper/figures/make_paper_figures.py`; analysis checks live in
`paper/figures/check_*.py`, with outputs under `outputs/`.
Review-round history is in the git log of branch
`worktree-paper-draft`.

---

## Abstract

Dependence structure in data — and in the representations of trained
neural networks — varies across the data manifold. A global factor model
must fit one factor budget to all of it, while a *local* variant fits
separate models to discovered regions. This paper establishes, through
controlled experiments, when localization improves the recovery of
context-dependent dependence groups, and uses neural interventions to
assess the behavioral relevance of the resulting groups. On Gaussian grids
where the grouped covariance varies across regions, localizing improves
top-d support recovery dramatically given an informative partition
(data-driven partition 0.980 vs. global 0.788 F1; +0.192, CI [0.177,
0.207]); it vanishes in the homogeneous control, and it is not a scoring
artifact: under coverage over the fixed
pooled ground-truth collection the local equal-candidate stack still
beats the global fit (1.000 vs. 0.65–0.91). A per-estimator analysis of
pairwise-invisible nonlinear generators shows the gain persists
(Bio: sine-sum +0.03…+0.13, parity +0.14…+0.30 F1; Linear: sine-sum
+0.02…+0.10, and parity only under the estimated partition). A
capability diagnostic restricts the default-EM parity failure to
optimization within the tested EM configuration, not expressiveness. A
matched-TC
study then draws the interpretive boundary: fitted TC-side summaries do
not support a reliable, general reading of interaction type, because they
also trace recoverability; the signed Ω diagnostic and the behavioral
probe answer different, complementary questions. On MNIST (5-seed MLP),
systematic ablations show local-CorEx groups are behaviorally consequential
for their size (larger, more class-concentrated shifts than size-matched
random deletion at every layer × threshold), with localization's aggregate
advantage over prediction-guided single-unit selection concentrating at
the deepest layer, and with local and global fitting identifying groups
with different layer-dependent behavioral profiles — a gap that persists
under an approximately support-volume-matched comparison. On ResNet-18/ImageNet
(282 (cluster, factor) cells, 94 clusters, exact pooled-feature deletion)
the protocol transfers unchanged: ΔTV +0.214 vs. matched random
(CI [0.192, 0.238]), and case panels show class signal identified on
discovery data collapsing the right class out of sample (owl −0.86 vs.
−0.05 for matched randoms). Throughout we keep separate how much
predictions change (a class-mean TV measurement over output coordinates,
maximized across input classes), how concentrated the changes are across
input cohorts, and which images experience the largest per-image shifts.

---

## 1. Introduction

Higher-order interactions — dependence among several variables that is
not reducible to pairwise relationships — are hard to detect, and harder
still when they *vary across contexts*. Total Correlation Explanation
(CorEx) learns latent factors that explain the total correlation (TC) of
a dataset, and its variants Linear CorEx (modular, Gaussian) and Bio CorEx
(continuous/missing-data, discrete latent) identify groups of dependent
variables whose structure is recovered through each factor's mutual
information with the original features. The prior conference work (Kerby,
White & Moon, MLSP 2024) established Local CorEx — partitioning
observations with PHATE + k-means and fitting CorEx per cluster — and
showed on Gaussian recovery data and selected MNIST examples that local
fits outperform global fits when interactions vary across the manifold.
The dissertation chapter expanded the
applications (tabular communities
data, flattened MNIST pixels, classifier layers, ResNet-18/ImageNet) and
added qualitative visualizations (decoder reconstructions, DeepDream);
its results are cited to the dissertation throughout. The ImageNet case
study below extends the dissertation's single-cluster demonstration with
a control-matched systematic survey. Material already in the MLSP paper
is treated as prior published work that this article substantively
extends.

Those results alone do not establish *when* localization helps, or
*whether* the resulting groups are behaviorally meaningful under
controls. The journal contribution is precisely that step: **Local CorEx
discovers groups of dependent variables whose structure varies across
contexts. Controlled experiments establish when localization improves
recovery, and neural interventions assess the behavioral relevance of the
resulting groups.** The advance here is the layer of controlled
comparison: matched partitions and controls, coverage-checked
equal-budget comparisons, held-out intervention protocols, and stated
uncertainty — none of which the earlier applications carried.

The named contributions, each carried by one experiment:

1. **Operating conditions (E1, Fig. 2).** A 27-condition × 16-replicate
   Gaussian grid separates the factors behind the average: the
   strongest general pattern is the covariance regime — the
   localization benefit vanishes in the homogeneous control, is small
   and sample-size-dependent in the shared-structure (nondisjoint)
   regime, and is large regardless of size or separation when each
   region's covariance structure is incompatible (disjoint). It also
   requires an *informative* partition (random partitions do not beat
   global), and at equal candidate-factor counts the local stack still
   beats the global fit under fixed-collection coverage (Fig. 2c).
2. **Boundary and estimator responsibility (E2, Fig. 3).** The same
   question under pairwise-invisible nonlinear structure. Every headline
   number is labeled with its CorEx estimator; estimated partitions are
   shown alongside the true-region reference partition; two reconciliation
   conventions are stated (§4.2) and the parity-inversion pattern is
   reported directly.
3. **Interpretive boundary (E3, Fig. 4).** What information about
   dependence structure survives fitting and support selection.
   Population TC does not uniquely determine redundancy vs. synergy
   dominance; *fitted* TC additionally traces recoverability; and neither
   statistical dependence nor recovery quality determines a group's
   behavioral role. The larger diagnostic grid moves to supplement.
4. **Systematic neural evidence, with its two separated questions (E4,
   Figs. 5–7).** Discovered MNIST groups are behaviorally consequential
   relative to *arbitrary* (size-matched random) groups of the same size;
   whether the multivariate factor model adds anything beyond a simpler
   prediction-guided single-unit selector is a separate question with a
   layer-dependent answer. A direct local/global comparison on the same
   measurements is added, and qualitative examples from the dissertation
   restore an intuitive reading of what the factors capture.
5. **Protocol transfer (E5, Fig. 8).** The instrument, protocol-frozen and
   unmodified, on a large pretrained network with an explicit
   discovery/evaluation split and cluster-level uncertainty.

**Scope.** The Gaussian grids validate recovery of *multivariate
dependence groups* whose structure is specified through covariance; they
do not establish recovery of interactions irreducible to pairwise
relationships — that boundary is E2's subject. The communities, dropout,
and pixel-group applications of the dissertation chapter are left outside
this manuscript (each introduces its own new scientific questions; see
§9, and the dropout study as a companion-paper direction).

**Explicit non-claim.** No claim that a recovered factor's statistics
reveal interaction *type* (redundancy vs. synergy), nor that TC magnitude
predicts functional properties beyond what the intervention instrument
directly measures. Recovery, interaction-type characterization
(computable only on tractable joints), and intervention are measurement
vocabularies: O-information characterizes statistical dependence;
deletion measures behavioral contribution. Deleting a group does not
establish whether its underlying dependence is redundancy- or
synergy-dominated.

## 2. Background and setup

### 2.1 Total correlation and CorEx

Let X = (X₁,…,X_p) be observed variables and Z = (Z₁,…,Z_m) latent
factors. The total correlation of X is
TC(X) = KL(p_X ∥ ∏ᵢ p_{Xᵢ}) — the generalization of mutual information
to many variables. CorEx maximizes the explained total correlation
TC(X;Z) = TC(X) − TC(X|Z): *Linear CorEx* assumes X is jointly Gaussian,
learns Z = WX + ε with independent modular factors, and supports parallel
factor learning; *Bio CorEx* is distribution-free in X with discrete
factors, suited to missing/undersampled data but more expensive. Each
factor's importance is its explained TC; the predicted interaction group
("HOI") of a factor is the set of variables with high mutual information
with it, ranked per factor. Throughout, ranking rules never select on
estimated TC across arms being compared; supports use a fixed
cumulative-MI threshold rule per factor.

### 2.2 Partitions: what is clustered and what is grouped

Two distinct operations must not be conflated:

- **PHATE + k-means partitions *observations*** (rows): the data
  manifold is embedded (10-d by default) and clustered into regions;
- **CorEx groups *variables*** (columns): within each region, a fit on
  the **original variables** (never the plotting coordinates) yields
  factor–variable MI scores that define interaction groups.

Fitted per region, the local *Linear* CorEx configuration is
piecewise-linear, which makes it fast (parallel Linear CorEx per region)
and empirically capable on mildly nonlinear data (the experiments below);
this scoping applies to the Linear configuration only and is not a claim
that partitioning makes every CorEx estimator robust to nonlinear
dependence. Hyperparameters (embedding
dimension, k, factors per fit, cumulative-MI threshold q) are fixed per
experiment below.

### 2.3 Three measurement vocabularies

- **Recovery** — does the fit rediscover the generating groups?
  (matched/top-d precision, recall, F1, cosine overlap).
- **Type characterization** — redundancies vs. synergies via signed
  O-information Ω, computable only where the joint is tractable.
- **Intervention** — behavioral effect of zeroing a factor's support on
  images the selection stage never saw.

## 3. Shared design conventions

1. **Paired design.** One dataset per condition, shared by every method
   compared on that condition; method differences are read like-for-like.
2. **Derived seeds.** Seeds are hashes of (root, purpose, condition);
   re-running a condition reproduces its data bit-for-bit.
3. **Discovery/evaluation separation.** Everything that selects
   (clustering, fitting, support thresholds, ranking) sees only discovery
   data; outcomes are computed on evaluation data the selection stage
   never saw.
4. **Restartability and failure-as-data.** Conditions write atomically
   and skip on rerun; failed fits are recorded with status, never
   silently dropped. (E1/E2/E3 grids: zero failed fits across all
   conditions.)
5. **Uncertainty discipline, labeled per figure.** Pairing/interval units
   are stated per figure: (synthetic grids) paired t-based 95% CIs across
   the 16 replicates of each condition, df = 15; (ImageNet, Fig. 8)
   cluster-level bootstrap, B = 2000; (MNIST) error bars are ±2·SE of
   the stored pooled aggregates over (seed × cluster × factor) cells —
   scale indicators, since the replication unit is the pseudo-replicate
   pool rather than the 5 network seeds. Where the E4 store allows
   per-seed summaries (the local arm's absolute panels, Fig. 6a–b), the
   figures show per-seed ±2·SE; the E4 global arm's per-factor store is
   not preserved, so its side of the comparison reports pooled means
   with the interval flagged (§6.5, §A).
6. **Matching conventions are stated per experiment and never mixed
   silently.** E1's primary score is *matched* top-d F1 under a
   one-to-one (Hungarian) alignment between learned factors and true
   groups — a recall-or-precision view of the whole factor collection.
   E2's primary convention is *per-true-group best top-d match* (each
   true group takes its best candidate factor) — a recall-flavored view
   that is *favored* by larger candidate pools (see the chance-reference
   note below). E2's
   sensitivity arm keeps a pooled all-rows convention. Chance references
   use the same candidate-factor pool and matching procedure as the
   scored arm (§4.3); because the per-true-group-best rule rewards larger
   candidate pools, random-partition rows in E2 are read only for
   qualitative sanity, not as calibrated baselines.

| Table 1 — benchmark conditions |
|---|
| **E1 Gaussian** | 3 covariance regimes × 3 region sizes × 3 mean separations; 16 paired reps; Linear + Bio estimator; partitions {true-region reference, PHATE, random}; equal-candidate capacity arm (global M ∈ {5,10} vs. the 10-candidate local stack, scored by coverage and matched-pair precision) |
| **E2 nonlinear** | sine-sum (continuous, pairwise-invisible) + parity (discrete, pairwise-invisible) + product (supplementary, *not* pairwise-invisible); 2 regions with incompatible group structure; n ∈ {500, 5000} × α × noise; Linear + Bio |
| **E2 parity stress** | parity block alone; default-EM vs. ground-truth atom-seeded init at equal state cardinality; 8 reps/cell |
| **E3 type boundary** | redundant vs. parity generators calibrated to exactly TC\* = 0.500 bits; d ∈ {3,4,5}; n ∈ {500,5000}; 32 paired reps; 768 conditions |
| **E4 MNIST** | 5 network seeds; 3 layers (500/400/300 units); 20 discovery clusters (PHATE + k-means on 10k discovery images); per (seed, cluster, layer) Linear CorEx, top-5 positive-TC factors; q ∈ {0.6,0.7,0.8,0.9} (+0.95/0.99); global-CorEx arm (M ∈ {10,100}); 141k + 66k + 66k intervention evaluations, zero failures |
| **E5 ImageNet** | ResNet-18 on full ImageNet val (50k images); discovery = 25/class, evaluation = 25/class; k = 100 clusters; M = 10, top-3 retained; q = 0.80; 282 (cluster, factor) cells in 94 clusters; 5 matched-random + univariate per cell |

## 4. Controlled recovery

![Figure 1 — pipeline overview](figures/figure_1.png)

**Figure 1 — the shared pipeline.** PHATE + k-means partitions
*observations* (rows); per-region local CorEx groups *variables*
(columns) into factors (with a global fit for comparison); per-factor
cumulative-MI supports over the original variables; exact masked
interventions; held-out behavioral readout (T_max class-mean TV over
output coordinates, maximized across input classes; input-class
concentration H_input; localization contrast L_j) against
size-matched random and univariate controls.

### 4.1 Gaussian grids: the operating conditions (E1)

Two equally sized Gaussian populations over 25 signal variables in five
disjoint groups of five (equicorrelation ρ = 0.7 within groups). Three
covariance regimes fix how *incompatible* the regions' group structures
are: *homogeneous* (identical), *nondisjoint* (regions share most groups,
partially overlapping), *disjoint* (no shared groups). Mean separation α ∈
{0, 0.5, 1.0} controls partition difficulty; region sizes {50, 500, 5000};
16 paired replicates per condition; true-region reference, PHATE +
k-means, and size-preserving random partitions; matched top-d F1 under
one-to-one alignment.

![Figure 2 — operating conditions and capacity](figures/figure_2.png)

**Figure 2 — the conditions behind the averages.**
(a) Data-driven (PHATE) − global paired ΔF1 per condition: exactly zero
across the homogeneous control; the disjoint regime is large at every
condition (+0.28…+0.47, roughly flat in α); *within* the nondisjoint
regime the gain **increases with mean separation** (e.g. +0.02 → +0.20 →
+0.25 at n = 50) and with region size only at α = 0 (+0.02 → +0.14);
at α ≥ 0.5 the nondisjoint gain is roughly stable around +0.20. Size
and separation effects are therefore conditional on the regime, and the
regime is the strongest general pattern. (b) True-region reference −
global: ≈ +0.20 nearly everywhere the regions exist (up to +0.47),
including small-n disjoint where the PHATE partition fragments. The
true-region partition is called a *reference*, not an upper bound: §4.2
shows an estimated partition can outperform it for a particular
estimator. (c) The equal-candidate capacity comparison, with the two
scoring views separated: **coverage** (F1 summed over the fixed 10-group
pooled collection; unmatched groups contribute zero) and **matched-pair
precision** (mean over matched pairs only, the statistic plotted in
earlier drafts). With only M = 5 global factors, coverage is necessarily
≤ 0.5 against 10 pooled groups (0.440 disjoint / 0.500 nondisjoint) even
though matched-pair precision is 0.880–1.000; at M = 10 the assignment
becomes square and coverage rises to 0.909 / 0.649. The budget-matched
local stack (two M = 5 fits = 10 candidates) reaches 1.000 coverage in
both regimes. The earlier "nondisjoint M = 5 tie" and "M = 5 → M = 10
decline" are therefore **denominator artifacts of averaging over matched
pairs only**: the global fit never loses coverage when its budget grows.

*Conditions that fail, kept visible.* In the homogeneous regime the
localization gain is zero (global = reference ≈ 1.000): localization
needs heterogeneity. Random partitions do not beat global under the
matched convention (Δ +0.021, CI [−0.001, +0.042] crosses zero): the
partition must be informative. TC is not a recovery proxy: random
partitions explain *more* TC than the reference (2.287 vs. 1.439 bits
per fit), so explained TC alone cannot rank partitions.

*The nondisjoint M = 10 fit, decomposed.* At the square 10 × 10
assignment, the global nondisjoint fits reach coverage 0.649 (vs. 1.000
for the equal-candidate local stack). Two measured components explain
the gap: (i) ≈0.9 of the 10 fitted global factors per replicate are
nonnegligible-but-degenerate (empty top-k support → scored 0; degenerate
fraction 0.09–0.20 across replicates, maximum 2/10); (ii) the remaining
≈9 matched factors recover ≈ **0.715** of their group's members on
average (matched non-degenerate F1 range 0.64–0.825; mean matched cosine
similarity 0.73) — mixture supports that straddle the overlapping group
boundaries. Disjoint M = 10 global fits show zero degenerates (matched
cosine 0.83, coverage 0.91), so the effect is regime-specific. We report
this as an observed global-fit, regime-specific behavior and do not
carry a broad capacity argument on it alone (the neural analogue — added
budget leaving the retained top-5 factors' explained TC unchanged — is
§6.5's M = 100 check, phrased narrowly there).

*Validity note.* The Gaussian design specifies structure through
covariance: it validates recovery of grouped *multivariate dependence*,
whose interactions are pairwise-visible in the Gaussian case. Higher-order
pairwise-invisible structure is E2's subject.

*Estimator generality.* On the single check condition (n = 500, α = 0.5)
under Bio CorEx, the gain reproduces (true-region reference 0.992 /
PHATE 0.988 vs. global 0.659). Bio's cost profile on the neural scale is
addressed
separately (§6, pilot verdict; the estimator-generality claim carried by
the synthetic grids, not by a neural Bio arm, which was declined on
feasibility).

### 4.2 Pairwise-invisible nonlinear structure (E2)

Groups carry dependence invisible to any pairwise scan: **sine-sum**
(continuous; Y = sin(ΣΘᵣ) + ε, marginally independent of each angle) is
the primary generator; **parity** (discrete; the d-th variable XORs the
others with flip noise) is the stress case; **product** is a supplementary
comparison that is nonlinear but *not* pairwise-invisible and is never
used as such. Two regions carry deliberately incompatible group
structures; two routing variables encode region identity (excluded from
scoring); nuisance columns grow dimensionality without structure. Each
headline says which estimator produced it.

![Figure 3 — nonlinear recovery](figures/figure_3.png)

**Figure 3 — one reproducible primary calculation, one clearly defined
sensitivity analysis, and the capability diagnostic.** Both quantitative
panels are computed from the committed combined match tables by the
committed figure script, filtered to **one input variant per generator**
before matching (the primary `binned/with_routing` variant; pooling the
two continuous variants would let each group score against the better
variant's factors, which would report best-of-preprocessing rather than
the performance of one configured method).
(a) *Sensitivity analysis, per condition:* paired ΔF1 vs. global under
the per-true-group best-match convention, computed separately for each
(α, noise, estimator) condition and paired within it across the 16
seeds (95% t CIs, df = 15; labels show wins/16; n = 5000, noise 0.05
shown; over all 9 conditions per generator, signs are unchanged except
one small negative Linear sine-sum cell at the noisiest setting).
For the Bio estimator the true-region and **estimated PHATE
partitions** beat global on sine-sum (oracle +0.054…+0.130, PHATE
+0.003…+0.068) and clearly on parity (oracle +0.142…+0.193, PHATE
+0.199…+0.295, 14–16/16). Under Linear the sine-sum gains are small
(oracle +0.018…+0.036; PHATE +0.016…+0.098) and the parity pattern is
inverted: the true-region partition does *not* beat global
(−0.012…−0.023, consistent with this record's own linear-parity entry,
ΔF1 −0.052, CI [−0.093, −0.010]) while **PHATE does** (+0.177…+0.207,
15–16/16 seeds). (b) *Primary convention, computed and labeled:* the
primary analysis convention — per-(seed, α, noise, region, group) best
match over **all candidate factors** of the Bio fits on the primary
input variant, per-condition means weighted equally within each seed
across all 3 α × 3 noise levels, seed-level pairing with a t(15) 95% CI
— sine-sum n = 5000, F1 **0.337 vs. 0.268** (Δ **+0.069**,
CI [0.043, 0.094], 15/16 seeds), cosine **0.383 vs. 0.319** (Δ +0.064,
CI [0.043, 0.085]); parity n = 5000, F1 0.425 vs. 0.263 (Δ +0.161,
CI [0.123, 0.200]), cosine 0.438 vs. 0.282 (Δ +0.156, CI [0.118,
0.194]). (c) *Parity capability diagnostic* (8 reps/cell; EM-fitted
variant): the model family *represents* the parity structure at the
tested cardinality — a ground-truth atom-seeded init reaches ≈ the
analytic 0.500-bit TC at every cell — while the default-EM init
collapses to 0.0000 bits in 8/8 replicates at n = 5000 (d = 3, 4) — an
optimization failure from the default initialization, not an
expressiveness failure (consistent-with, not a proof). The diagnostic is
a statement about the EM-fitted variant's basin structure; it does not
explain the Linear CorEx parity result in panel (a).

**Reconciliation of the two conventions (why (a) and (b) differ).** With
the single-variant filter, the two computations share the
candidate-factor pool, the best-match rule, the nuisance settings (the
core grids are single-nuisance) and the region handling (both keys
include the region identifier; each true group belongs to exactly one
region). The remaining differences are (i) the estimator rows shown —
(b) plots the Bio fits with the linear reference annotated as absolute
scores, while (a) plots each estimator separately; (ii)
weighting/pairing — (b) averages per-condition values within each seed
across all 3 α × 3 noise levels and pairs at the 16 seeds, whereas (a)
pairs within each condition; and (iii) interval construction (t(15)
both). In fact, with the filter, (b)'s point estimate equals (a)'s
all-condition average exactly (e.g. the sine-sum +0.069 is the mean of
(a)'s nine Bio true-region deltas) — the two calculations are the same
estimator read at two aggregation levels, and only the CIs differ.

**The linear-parity pattern, reported directly.** For the *Linear*
estimator on parity data at n = 5000, PHATE partitions help while the
true-region partition does not. The cause is not established by the
current evidence. Two candidate explanations are recorded without
commitment: PHATE conditioning may create locally pairwise-visible
structure within otherwise pairwise-invisible populations (the sort of
conditional structure Parity's XOR destroys globally), or it may be an
artifact of fit selection/scoring across partition sizes. The EM
capability diagnostic in (c) applies to the EM-fitted parity fits of E3,
not to Linear CorEx's fitting procedure, and is not offered as the
explanation of the linear-parity inversion.

*Preprocessing variants as labeled sensitivity.* The continuous grid's
Bio fits exist on two input variants; the primary tables use
`binned_with_routing` throughout. The alternative (`gaussian_with_
routing`) is reported only as a variant: its pooled true-region
sine-sum scores are cosine 0.426 / F1 0.336 (slightly higher on cosine
than the primary variant's 0.383 / 0.337), and it never enters panels
(a)–(b) or any headline.

The honest composite: **nonlinear success depends on both partitioning
and the estimator; with the appropriate estimator and the practical
partitioned configuration, localization improves group recovery;
support-level precision stays moderate (cosine 0.383 at n = 5000 under
the primary convention).**

### 4.3 Matching conventions, stated and separated

E1's matched top-d F1 (one-to-one Hungarian over the full factor
collection) and E2's per-true-group best match answer different recovery
questions, and neither is silently traded for the other. Within each
experiment the conclusion is checked against a comparable convention:
E1's equal-budget arm reproduces the E1 conclusion under per-factor
best-match scoring as well; E2's pooled all-rows convention is retained as
a labeled sensitivity reading (same signs, smaller margins). Chance
references are computed with the same candidate-factor pool and matching
procedure as the scored arm — under per-true-group best-match rules,
larger candidate pools reward more factors, so random-partition rows in E2
(a 10× larger pool) are not calibrated baselines and are read only
qualitatively.

## 5. Interpretive boundary: what survives fitting and support selection (E3)

Three claims are distinct and must not be fused:

1. **Population TC does not uniquely determine redundancy vs. synergy
   dominance** — two generators with identical TC can carry opposite
   higher-order structure.
2. **Fitted (explained) TC additionally depends on the estimator and
   optimization** — a fit's TC-side summaries trace *whether the
   optimization found the structure*, not only the type.
3. **Statistical dependence does not determine a group's behavioral role**
   — even a perfect characterization of the dependence is a different
   measurement than what removing the group does to a network. (The
   intervention experiments answer that third question; they do not sign
   interaction type.)

E3 addresses the first two by construction: a redundancy-dominated and a
synergy-dominated (parity) generator are calibrated to **exactly the same
population TC** (TC\* = 0.500 bits, residual error < 1e-6); population Ω
is +0.133…+0.166 (redundant) vs. exactly −(d−2)·TC = −0.500…−1.500
(parity), while pairwise MI is 0.074–0.211 vs. exactly 0.

![Figure 4 — what fitted summaries keep](figures/figure_4.png)

**Figure 4 — the boundary in two panels.**
(a) At matched population TC = 0.500 (dotted line), fitted top-factor TC
differs sharply by generator (redundant 0.42–0.50 bits vs. parity
0.02–0.11; main arm n = 5000, 11/12 cells outside the ±0.05 equivalence
margin). The one inside the margin (d = 3, n = 500) is exactly the cell
where both structures recover perfectly (top-d F1 = 1.0 both sides).
Read not as "TC is blind to type" but as: **fitted TC summaries do not
provide a reliable, general interpretation of interaction type — they
also encode recoverability** (mechanism-capacity matches and EM basins; §4.2).
(b) The signed instrument, read as *the effect of support-recovery error
on subsequent characterization*: Ω on the *discovered support* equals
population Ω wherever the support matches the true group (all redundant
n = 5000 cells; parity d = 3) and — when recovery fails — returns a value
near zero (parity n = 5000: −0.03…−0.12 on the support vs.
−0.50…−1.50 on the true group, markers) **without any warning that
important variables were missed** — this silent failure, not a dramatic
one, is precisely the limitation the experiment demonstrates. Recovery
quality is not a safe proxy for "the structure understood", and on the
correct columns Ω's **estimation error** (absolute deviation from the
exact population value) is ≈0.01 bits at n = 5000 (≤0.04 bits at n = 500).
The full 768-condition grid (32 paired replicates, 5 EM restarts, zero
failed fits, ≈64 CPU-h) moves to supplement.

## 6. Neural representation analysis I: MNIST

### 6.1 Shared neural setup (read before both neural experiments)

The table below fixes, in one place, the design elements common to E4 and
E5. "Held out" always means *held out from factor discovery and support
selection*: the essential separation is between the selection stage and
the intervention evaluation stage, which is what the analysis requires.

| Design element | Definition |
|---|---|
| Discovery pool (E4) | 10,000 images partitioned out of the 60,000-image training pool (split manifest: 45,000 classifier-training + 5,000 validation + 10,000 discovery, per-class counts accounted; verification in `paper/figures/check_capacity_metrics.py`). The essential separation for the analysis is between factor selection and intervention evaluation — the discovery pool is disjoint from classifier training by the recorded stratified split, and the evaluation stage uses only the untouched test images |
| Evaluation pool (E4) | the untouched 10,000-image **canonical test split** — used only for intervention outcomes |
| E5 pools | full ImageNet val: first 25 images per class = discovery (HF row order), remaining 25 = evaluation; evaluation images routed to clusters by the frozen fallback (nearest discovery neighbor) after standardization |
| Partition construction | discovery **images** 512-d pooled (E5) or flattened (E4) activations → PHATE → k-means; one partition shared by all layers/factors; deterministic refit reproduces assignments exactly (E4: agreement 1.0000 over 10,000) |
| Factor inputs | *output-aware*: standardized hidden activations + row-centered standardized logits (logits aid interpretability); *hidden-only*: activations alone |
| Support construction | per factor, rank per-variable MI and take the smallest prefix whose cumulative share reaches q; the denominator counts activation coordinates (logit columns excluded); supports are precomputed on discovery data |
| Interventions | E4: activation masking, deterministic, dropout off (eval mode), trunk-equivalent to full masked forward (unit-tested); E5: exact pooled-feature update Δz = −W[:,S]h_S, demonstrated against re-forwards (max logit absdiff 3.6e-5 ≤ 1e-4 frozen tolerance) |
| Matched random controls | E4: 20 (output-aware) / 5 (hidden-only) uniform draws of the same size within the layer; E5: 5 draws per support |
| Univariate control | the top-s units by |correlation| with the factor's most-associated logit **within the same discovery cluster**, on centered logits; E4: output-aware arm only; E5: per cell, same cohort |
| Behavioral metrics | per **input class** y (cohort): class-mean prob vector Δ_{y,c} = E[pᶜ_c(X) − p_c(X) \| Y=y] over **output coordinates** c; T_y = ½Σ_c\|Δ_{y,c}\| — i.e. T_y is the **total-variation distance between the class-mean prediction vectors**, and T_max = max_y T_y (a maximum over input classes, not over output coordinates, not measured in bits); r_y = T_y/Σ_u T_u and **H_input** = normalized entropy of r (concentration **over input classes**; a related output-coordinate concentration H_output is stored); **L_j** (E5) = (tv_source − tv_off)/(tv_source + tv_off), tv = aggregated probability-model distance on the discovery- vs. off-cluster cohorts; complementary **per-image** summaries collected in the same evaluations: U_y = mean ½\|\|p_iᴬ − p_i⁰\|\|₁ (cancellation-free) and prediction-flip rates |
| Conservation & scale | each Δ_{y,·} sums to zero over c (class *means*, exact), so single-class probability drops imply gains elsewhere; probability and logit scales are distinct (logit shifts are not conserved) |
| Uncertainty | E4: 5 network seeds — primary MNIST comparisons are reported as per-seed paired differences where the store allows; where a committed table stores only a pooled (5-seed × factor-cell) aggregate, its ±2·SE bars are labeled as scale indicators (the replication units differ: seeds vs. pseudo-replicate factor rows); E5: cluster-level bootstrap (B = 2000), conditional on the analyzed network and split |

### 6.2 Instrument and grid (E4)

Spec-exact MLP (d ≈ 500/400/300 hidden units per layer); 5 independently
initialized models (test accuracy .9860–.9864); discovery pool and test
pool disjoint as above; per (seed, cluster, layer) Linear CorEx fits,
top-5 positive-TC factors; q ∈ {0.6,0.7,0.8,0.9} (supplementary 0.95/0.99);
141,000 output-aware + 66,000 hidden-only + 66,000 re-evaluation
intervention rows, zero failures.

### 6.3 Qualitative example: what a factor looks like

![Figure 7 — worked example](figures/figure_7.png)

**Figure 7 — the retained worked example, from the *earlier* analysis
(dissertation Ch. 2; the panel reproduces the dissertation's Fig. 2.12a).
Provenance differs from the benchmark:** it comes from an earlier
2-hidden-layer MNIST network with **200-unit layers**, whose hidden
representations were clustered on the **test data** with selection by
top mutual information — not the 500/400/300-unit benchmark model, its
discovery/evaluation split, or its matched controls. It illustrates what
the method finds under an earlier configuration and inherits none of the
benchmark's selection-discipline statements. For a 4/9-heavy cluster,
factor 1 of the H1 fit carries TC ≈ 9.2 bits (14.8% of the layer's):
perturbing its nodes in proportion to their MI bends the digit's
top-left stroke toward a 9, and the factor's MI profile concentrates on
one logit (4). Factor 2 (5.1 bits) shifts the middle bar and base. None
of the paper's quantitative claims rests on this panel. (The remaining
historical panels — cluster-deletion accuracy bars, the ResNet-18
DeepDream textures, and the African-grey logit shifts — are kept as
supplement assets at `paper/figures/assets/diss_*.png`; the ResNet-18
case study they come from is described in §7.)

### 6.4 Systematic benchmark results

![Figure 5 — MNIST benchmark](figures/figure_5.png)

**Figure 5 — MNIST benchmark** (error bars: ±2·SE of the pooled
(5-seed × cluster × factor) aggregates stored in the experiment record —
read as scale indicators; the replication unit is the pooled aggregate,
not the 5 network seeds, which is why the paired mean differences are
interpreted against those records' SEs rather than a seed-level CI).
(a) Local-CorEx supports produce larger class-mean TV shifts than
size-matched random deletion at **every** layer × threshold × seed
stratum (H2, q = 0.9: paired ΔT_max **+0.166**, SE 0.006; per-seed
excesses 0.147–0.184), while random deletion of the same sizes leaves
this aggregate essentially at zero (random T_max ≈ 0.005–0.006). (b)
Layers separate in **input-class concentration**, not only magnitude: at
q = 0.9 conjunctive prevalence (T_max ≥ 0.10 & H_input ≤ 0.50) is
**0.376 (H2)** / **0.370 (H3)** vs. **≤ 0.06 (H1)**; mean T_max is
0.113/0.171/0.155 across H1/H2/H3 — H1 carries substantial influence that
is often broad (34.8% of H1 factors strong-but-broad), so its low
prevalence reflects concentration, not inactivity. (c) The univariate
control is a fair competitor on magnitude except at the deepest layer:
ΔT_max (LC − univariate) is +0.006 → **+0.053** growing in q at H3 but
−0.019/−0.021 at H1/H2, q = 0.9 (H3 ΔH_input −0.116); at H1/H2 the
differences are within ±0.021, sign-mixed. (d) Per-image complement:
because class-*mean* shifts allow cancellation, the collected per-image
metrics close the loop — relative to matched randoms, LC deletion raises
the prediction-flip rate (the fraction of the 10,000 test images whose
argmax prediction changes, averaged over all input classes) by
**0.6–1.6 percentage points at q = 0.9** (H3 0.56 pp to H2 1.59 pp;
smaller at lower thresholds), i.e. individual predictions change far
more under selected
deletion than under random deletion even where the class-mean TV
difference is small.

*Two distinct questions, answered separately.* (i) *Consequential
selection:* selected groups are behaviorally consequential relative to
arbitrary groups of the same size — unambiguous, everywhere. (ii)
*Beyond simpler selection:* the multivariate factor model's aggregate
advantage over prediction-guided single-unit selection is **layer-scoped**
(H3 only, growing in q) — factor organization and behavioral impact
provide complementary descriptions of a representation, and neither is
strictly dominated by the other on these aggregates. Dose–response:
q = 0.95 raises T_max (0.113/0.171/0.155 → 0.215/0.250/0.254) without a
concentration penalty; q = 0.99 declines specificity (H3 prevalence
0.382 → 0.150) as broad-support masks' chance overlap approaches that of
same-size randoms. TC ↔ behavior is modest after size adjustment
(partial Spearman −0.01…+0.32; residualized 0.29–0.48, method-sensitive);
directional concordance tracks layer (0.27/0.75/0.55) against a measured
random reference of 0.10–0.13. [Exploratory] Per-image cohort analysis:
same-digit images inside vs. outside the discovery cluster are
indistinguishable on average (≤ 0.004, SE ≤ 0.002), with within-digit
heterogeneity across clusters/layers and discovery-side association
predicting held-out excesses (Spearman 0.32/0.65/0.67). All claims are
influence language; interaction type is never signed.

### 6.5 The direct local/global comparison

The global-CorEx arm (one global Linear CorEx per (seed, layer,
condition); M = 10 per-fit parity, M = 100 budget-matched) makes the
local/global comparison on the *same measurements*. Same-threshold
comparisons are **not size-controlled**, so this section reports support
sizes explicitly (panel c) and adds an approximately
**support-volume-matched** comparison (panel d): pairs of thresholds
(global q = 0.7 with local q = 0.8; global q = 0.8 with local q = 0.9)
whose pooled support fractions agree within 0.6–2.3 percentage points of
the layer. Pooled absolute means are used on both sides (the global
arm's committed tables store only pooled means; its per-factor behavioral
rows are not preserved in the current workspace, so a per-factor/seed
re-derivation and a fully controlled common-cardinality design remain
export tasks, §A).

![Figure 6 — local vs global, same measurements](figures/figure_6.png)

**Figure 6 — local and global fits identify groups with different
layer-dependent behavioral profiles (output-aware; global M = 10).**
Local bars in (a)–(b) carry per-seed ±2·SE error bars (5 network seeds,
per-seed means over the 20-cluster pool); the committed global-arm
tables store only pooled means, so a matching global-side interval is
not derivable from them (flagged §A).
(a) Magnitude at q = 0.9 (T_max, class-mean TV): global factors produce
**larger** effects than local factors at H1 (0.60 vs. 0.11) and H2
(0.77 vs. 0.17), but at H3 exceed their size-matched random reference by
only ≈+0.03 (0.04 vs. 0.01), while local factors retain substantially
larger H3 effects (0.16). The random-reference levels themselves differ
by arm and support size (global H1 randoms 0.04 > local randoms
≈ 0.006): magnitude ranks depend on the reference and the intervention
size, which panel (c) makes explicit. (b) Input-class concentration:
global H2 factors are *more* class-concentrated than local (0.18 vs.
0.56), with H3 again inverted (global 0.83 ≈ the random reference 0.92
vs. local 0.59). (c) Same q, different removal volume: global supports
reach 38%/33%/40% of H1/H2/H3 units at q = 0.9 vs. local 25%/20%/28%.
(d) Matched support-fraction pairs (volume differing by ≤2.3 pp of the
layer within each pair): the profile persists at matched volume —
global leads at H1 (0.18 vs. 0.05 and 0.38 vs. 0.11) and H2 (0.10 vs.
0.08 and 0.37 vs. 0.17), while local factors are **larger at H3 under
both matched pairs** (0.01 vs. 0.07 and 0.02 vs. 0.16, global vs.
local). **Summary: local
and global fitting identify groups with different layer-dependent
behavioral profiles at the same q *and* at approximately the same
support volume — the global fit is potent and class-concentrated at
H1/H2, the H3 advantage is local, and the layer's concentration findings
(Fig. 5b) and the beyond-univariate-selection effect (Fig. 5c) are
properties of the localized fits.** Relative to their own matched
randoms, global groups exceed randoms in effect and concentration nearly
everywhere (ΔT_max vs. randoms: +0.56 (H1) / +0.76 (H2) / +0.03 (H3)),
so the statement is about *where structure lives*, not about global
factors being behaviorally negligible.

*Factor-budget wording, made precise.* Distinguish factors *fitted*
(M per fit), *retained* (top-5 by explained TC), and *evaluated* (those
supports intervened on; the global arm evaluates its retained top-5).
At M = 100 the global top-5 factor TC is essentially unchanged (≈16–35
bits, top-5, output-aware, H1–H3). **That establishes, and is stated as
no more than, unchanged top-five explained TC**; with no neural
ground-truth groups, it supports nothing about the unretained factors'
usefulness, and we draw no claim from them.

## 7. Neural representation analysis II: ImageNet (E5)

The dissertation's ResNet-18/ImageNet case study (supplement assets,
§6.3) demonstrated the readout on a single cluster; this section extends
it with a protocol-frozen, control-matched systematic study. Settings
per §6.1
(pools, routing, Δz = −W[:,S]h_S with the demonstrated cache); protocol
frozen (F1–F7) before outcomes were computed, with the
two mid-life freeze changes logged before the affected results were read.
300 fits across 100 clusters, zero failures; per-cluster size
distributions written before any systematic table (median 161, min 17,
max 1669; 14 clusters flagged low_sample; 17 factor rows excluded below
the 30-observation floor); survey covers 282 (cluster, factor) cells in
94 clusters; cluster-level bootstrap (B = 2000). Pooled-feature deletion
is exact arithmetic: all readings are about **which statistically
selected feature groups contribute to which predictions of the final
linear layer** — no synergy or mechanism language anywhere.

![Figure 8 — ImageNet survey and panels](figures/figure_8.png)

**Figure 8 — held-out, systematic, and case-level evidence.**
The plotted paired contrasts use the primary-threshold pairing: controls
are matched to each factor at q = 0.80 (the committed summarizer
implements this; a check against the pre-fix convention — controls
averaged over all thresholds — changes no reported value beyond the
fourth decimal; `paper/figures/check_e5_pairing_fix.py`, §A).
(a) Paired contrasts (94 clusters, cluster bootstrap CIs): local-CorEx
supports vs. their matched randoms, Δ total-variation on the routed
evaluation cohort **+0.214** (CI [0.192, 0.238]) and Δ localization
contrast **+0.263** (CI [0.232, 0.296]). The sign convention is shown
(LC − control): the **univariate control yields larger raw TV
(−0.019, CI [−0.028, −0.012])** and slightly *higher*, not lower,
localization contrast (−0.017, CI [−0.026, −0.009]) — on these two
aggregate measurements the prediction-guided single-unit selector
matches or exceeds LC on ImageNet, so the argument for factor structure
rests on the consequential-selection result vs. randoms and on the
complementary-accounts reading of §6.4, not on aggregate superiority
over univariate selection. (b) Localization contrast distributions: LC
median **0.263** (97.5% of factors > 0) vs. random median **0.005**
(52% > 0). (c) Case-study panels (values recorded in the experiment
record; §A): **primary — great grey owl (cluster 21, 64.4% purity)**,
q = 0.80
ablation collapses the owl class by **−0.86 / −0.85** (two factors) vs.
**−0.05 / −0.09** for matched randoms; the owl class dominates the
per-class shift, and by exact conservation the freed probability mass
must return across the other classes (the record describes the
redistribution as diffuse; its detailed distribution awaits verification
against the stored probability vectors, §A);
L_j 0.58/0.56 vs. randoms ≈ −0.40; the univariate control matches the
owl collapse (−0.85 / −0.70). **Secondary, illustrative — African grey
(cluster 85, 100% pure, flagged low_sample)**: grey collapses −0.28 /
−0.35 vs. −0.08 (random), with redistribution to nearby bird classes as
recorded; no univariate control is recorded for this panel.

Dominant-class concordance is modest (0.308, CI [0.250, 0.366]) —
factor identity (by MI) and per-cell dominant-class behavior agree only
moderately, an honest dissociation also visible in the case panels (the
owl factors' MI-dominant logits are *not* the owl class). Factor-TC ↔ magnitude association is modest (Spearman ≈ 0.29–0.37). The case-study
selection rule was re-frozen class-agnostically (2026-11-05, before the
panels' results were read) after the original ≥ 100-discovery-image
African-grey rule proved impossible under the 25 + 25 split (max 50 per
class in the set); the degenerate first pick is superseded.

## 8. Discussion

**When localization helps.** The strongest general pattern is the
covariance regime (Fig. 2a): the localization benefit vanishes in the
homogeneous control, grows with mean separation and (at matched
separation, weakly) with region size within the shared-structure
(nondisjoint) regime, and dominates under incompatible (disjoint)
structures at every size. It also survives scoring: at equal candidate
counts, local fits beat the global fit by a wide margin under
fixed-collection coverage (Fig. 2c), and the E4 global arm's added
budget leaves the retained factors' explained TC unchanged (stated
narrowly). Under
nonlinearity, success depends on partitioning *and* on the estimator
(Fig. 3a): with the appropriate estimator the practical partitioned
configuration wins, support-level precision stays moderate (Fig. 3b),
and the Linear-parity inversion (PHATE helps where the true-region
partition does not) remains a reported pattern with an unestablished
cause. The parity failure under default EM init is optimization-mediated
(Fig. 3c) — capability ≠ successful fitting, read as consistent-with
rather than proof, and scoped to the EM-fitted variant.

**What the matched-TC experiment does and does not say.** Population TC
does not identify interaction type; *fitted* TC adds a recoverability
trace; and a group's behavioral role is a third, separate property. The
instrument for type is a signed Ω diagnostic on a tractable joint; the
intervention instrument answers behavioral questions. Neither substitutes
for the other, and deletion cannot sign type.

**What the intervention evidence is, precisely.** E4/E5 establish
structured behavioral effects of removing statistically selected groups:
consequential relative to arbitrary same-size groups (everywhere);
layer-scoped in the beyond-univariate comparison; and *different
layer-dependent profiles* for local vs. global fitting that persist
under an approximately volume-matched comparison — not a blanket
"localized is better than global on behavior". Neither broad MI supports
nor aggregated deletion establish that behavioral effects *require*
spread (head-vs-tail/singleton ablation not run — "distributed deletion"
is not claimed). In E5 the readout is pooled features before a linear
head: claims are about which feature groups contribute to which
predictions, not synergistic downstream computation; semantic
concentration of redistribution patterns remains qualitative.

**Baselines, prioritized.** If one further neural baseline is added, we
would prioritize **a simple alternative grouping method using the same
discovery data and intervention budget** (the value-of-localization
ladder's remaining rung), which is cheap and directly answers "does
CorEx add anything beyond simpler grouping?"; a Bio CorEx neural
expansion was declined on feasibility grounds (pilot: every global fit
and all H1 fits hit the compute cap; the estimator-generality claim is
carried by the synthetic grids).

## 9. Future work (short)

The dropout/TC-redundancy question is a companion direction, not paper
content: paired design (10 cloned-init seed pairs, dropout the only
difference), four separable sub-hypotheses, and interpretation rules that
license statistical-dependence language from TC and functional-redundancy
language only from deletion analysis. Next computations, in priority
order: **the fully controlled per-factor matched-cardinality design**
(re-derive the E4 global-arm per-factor behavioral rows from cache and
re-run the interventions at matched support sizes with seed-level
pairing; §6.5), the simple alternative-grouping baseline (§8), the E5
DeepDream semantic-concentration panels (deferred), and a subsampled Bio
CorEx arm (≈ 100-unit activations) if estimator generality is to be
claimed on neural data.

## References

1. T. Kerby, T. White, K.R. Moon, "Learning Local Higher-Order
   Interactions with Total Correlation," *IEEE MLSP*, 2024 (the
   conference version this paper journalizes).
2. G. Ver Steeg, H. Harutyunyan, D. Moyer, A. Galstyan, "Fast structure
   learning with modular regularization," *NeurIPS*, 2019.
3. G. Ver Steeg, A. Galstyan, "Discovering structure in high-dimensional
   data through Correlation Explanation," *NeurIPS*, 2014.
4. S. Pepke, G. Ver Steeg, "Comprehensive discovery of subsample gene
   expression components by information explanation," *BMC Medical
   Genomics*, 2017.
5. K.R. Moon, D. van Dijk, et al., "Visualizing structure and transitions
   in high-dimensional biological data," *Nature Biotechnology*, 2019
   (PHATE).
6. J.M. Rosas, P.A.M. Mediano, M. Gastpar, H.J. Jensen, "Quantifying
   high-order interdependencies via multivariate extensions of the
   mutual information," *Phys. Rev. E*, 2019 (O-information).
7. R.R. Coifman, S. Lafon, "Diffusion maps," *Appl. Comput. Harmon.
   Anal.*, 2006.
8. S. Casper, M. Nadeau, D. Hadfield-Menell, G. Kreiman, "Robust
   feature-level adversaries are interpretability tools," *NeurIPS*,
   2022.
9. [Dissertation chapter: "Learning Local Higher-Order Interactions,"
   Ch. 2 — source of the qualitative panels in Fig. 7; internal document.]

## Appendix A. Provenance, commitments, and open flags

| Item | Source |
|---|---|
| E1 grids & contrasts | `tnnls_human_experiments/gaussian/ablation_data/summaries/*.csv` |
| E1 capacity (coverage/precision) | `tnnls_human_experiments/gaussian/capacity_data/global_capacity/global_capacity_combined.csv`; decomposition in `paper/figures/check_capacity_metrics.py` + committed tables `outputs/capacity_check/` |
| E2 core grids | `.../exp2_nonlinear/nonlinear_results/{continuous_core,discrete_core}/*_combined.csv`; the primary convention is *defined* in `analysis_core_grids.ipynb` (cell 15) and is **recomputed by the committed figure script** (`_e2_frozen` in `make_paper_figures.py`, used for Fig. 3b — reproduced to all printed digits); the per-condition sensitivity uses the same candidate pool with per-condition pairing, filtered to one input variant (Fig. 3a); reconciliation in §4.2 |
| E2 parity capability | `.../exp2_nonlinear/nonlinear_results/parity_capability/parity_capability_summary.csv` |
| E3 | `.../exp3_redundancy/redundancy_results/` (`core/core_info_summary.csv`); detailed record `paper/exp3.md` §E3.12 |
| E4 | `paper/exp4.md` §E4.16; `tnnls_human_experiments/exp4_mnist_benchmark/results/` (absolute local arm: `mnist_tc_density_magnitude.csv`; contrasts: `mnist_lc_vs_random.csv`, `mnist_lc_vs_univariate.csv`; prevalence: `mnist_prevalence.csv`; global arm pooled: `global_arm_summary.csv`, `global_arm_contrasts.csv`, `global_arm_coverage.csv`; split manifest copy: `results/mnist_splits_manifest.json`); `paper/exp4_addendum_global_arm.md`; matched-volume comparison in `paper/figures/check_capacity_metrics.py` + `outputs/capacity_check/mnist_matched_volume.csv` |
| E5 | `paper/exp5.md`; protocol freeze `paper/exp5_protocol_freeze.md`; committed tables `outputs/summaries/imagenet_*.csv`; **corrected pairing tables used by Fig. 8**: `outputs/summaries/imagenet_systematic_lc_vs_*_q080.csv` (written by `paper/figures/check_e5_pairing_fix.py`, which also verifies control coverage: all 282 factor rows have 5 random / 1 univariate controls at q = 0.80); case-study panels: `paper/exp5.md` §E5.6 values (commit 45cd0bb produced them; the two-panel CSVs are not registered in `outputs/summaries/` — regenerate via `tnnls_human_experiments/exp5_imagenet/case_study_two.py` if the raw store is available) |
| Qualitative Fig. 7 | dissertation Ch. 2 Fig. 2.12(a) (retained panel); the historical montage (Figs. 2.13, 2.17, 2.20-top) is kept as supplement assets at `paper/figures/assets/diss_*.png`, extracted from `paper/Local_CorEx_Dissertation.pdf` |
| Claim audit | `paper/claim_evidence_table.md` |
| Figure regeneration | `paper/figures/make_paper_figures.py` (single committed script); analysis checks: `check_e5_pairing_fix.py`, `check_capacity_metrics.py` |

**Open flags (tracked, none silently patched in the figures):**

1. *Owl per-class redistribution (open).* The experiment record states
   "all other per-class shifts ≤ 1e-6", which conflicts with exact
   conservation of the class-mean probability vector (a −0.86 drop
   requires ≈ +8.6e-4 of mean gain elsewhere), and the "diffuse
   redistribution" description could not be verified from the
   committed data (the case-two probability store is not present in the
   workspace). The manuscript states only "owl dominates; conservation
   places the mass across the remaining classes"; re-read the
   `case_two_top_shifts` store before submission.
2. *E4 per-factor behavioral store (open, data preservation).* The E4
   intervention store (per-factor rows for the local random/univariate
   controls and the global arm) is **not preserved in the current
   workspace**; only the committed summary CSVs survive. Consequences
   flagged in text: the global side of Fig. 6 has no per-seed interval;
   the matched-volume comparison uses pooled means on the global side;
   seed-level paired intervals for the MNIST contrasts require
   re-deriving from the checkpoints' cached activations (no retraining
   needed, but the intervention stage must be re-executed).
3. *Fully controlled common-cardinality design (open).* Fig. 6d's
   volume matching is approximate (pooled fractions within 0.6–2.3 pp);
   the per-factor matched design at exactly equal support sizes remains
   §9's first computation.
4. ~~*Summarizer pairing (E5)*~~ **RESOLVED.** Controls are matched at
   q = 0.80 in the committed `summarize.paired_lc_vs_control`; the
   corrected calculation is the source of Fig. 8's panels, and the
   pre-fix comparison changes no reported value beyond the fourth
   decimal (`check_e5_pairing_fix.py`; convention tables committed at
   `outputs/e5_pairing_check/`).
5. ~~*E2 conventions*~~ **RESOLVED.** The primary convention is computed
   (Fig. 3b, `_e2_frozen`; reproduced to all printed digits of the
   analysis notebook), both E2 calculations are filtered to a single
   input variant, and after that filter their point estimates agree
   exactly (§4.2).
6. ~~*E4 discovery-vs-training overlap*~~ **RESOLVED.** The committed
   split manifest accounts for the pool exactly (45,000 + 5,000 + 10,000
   with per-class profiles; `check_capacity_metrics.py`), so the
   selection/evaluation separation, not classifier-training overlap, is
   the load-bearing statement (verified in `paper/figures/`
   `check_capacity_metrics.py`).
7. *Bio CorEx neural arm:* declined (feasibility pilot in
   `paper/exp4_addendum_global_arm.md` §A.5); the alternative-grouping
   baseline (rung 4 of the value-of-localization ladder) remains unrun.
