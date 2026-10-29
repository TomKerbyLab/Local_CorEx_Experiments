# Experiment 3: TC-matched redundancy versus synergy — a type-blindness diagnostic

Companion to `simulation_methodology.md` (which covers Experiments 1 and 2).
This document records the motivation and methodology for Experiment 3 —
reference implementation `tnnls_experiments/exp3_redundancy/`, human-pipeline
implementation `tnnls_human_experiments/exp3_redundancy/` (the latter is the
intended run target and implements the redesign described in §E3.4 and
§E3.8). Where a statement depends on results of runs that have not yet been
executed, it is marked as such; analytic population values of the
data-generating processes are exact mathematics and are reported as design
facts, not as results.

## E3.1 Motivation and role in the program

Experiments 1 and 2 are performance experiments: they ask whether localized
CorEx fits recover region-specific groups better than global fits, first under
linear (covariance-structured) dependence and then under nonlinear,
pairwise-independent dependence. Their ground truths sit at opposite ends of
an axis the two experiments themselves never manipulate or measure. An
equicorrelation block (Experiment 1) is a shared-cause, redundant structure;
an η = 0 parity group (Experiment 2) is a synergistic structure in which the
dependence is invisible to every pairwise scan. Neither experiment controls
that axis, and neither computes a signed measure of it.

Experiment 3 poses the interpretational question directly on controlled data:

> **Question (E3).** When the pipeline recovers a variable group, what does
> that establish about the *type* of interaction holding the group together —
> redundancy versus synergy?

The design constructs the two structures with **equal total correlation**,
fits the identical CorEx pipeline to both, and computes an external signed
diagnostic, O-information Ω = TC − DTC, on the group. The three-level
vocabulary this supports, used consistently through the paper:

- **Group recovery** identifies a statistical dependency and its member
  columns.
- **O-information** characterizes whether that dependency is redundancy- or
  synergy-dominated — where it is computable (synthetic regimes with a
  tractable joint).
- **Interventions** characterize a learned factor's behavioral influence in
  a network — the tractable stand-in where the joint is not.

The reason this question earns space in the paper is the premise the
methodology already operates on in the neural-network experiments
(Experiments 4–6): *Local CorEx identifies the nodes; a second,
structure-appropriate instrument characterizes what they are doing*. §E3.9
relates Experiment 3 to that architecture — conditionally on what the run
shows (§E3.8), since the two-stage design is justified by what TC cannot do,
and that limit is a property of the statistic, while what a particular run
demonstrates is empirical.

## E3.2 Information quantities

All quantities in bits. For a discrete vector X = (X₁, …, X_d):

\[
TC(X)=\sum_{i=1}^{d}H(X_i)-H(X),
\qquad
DTC(X)=H(X)-\sum_{i=1}^{d}H(X_i\mid X_{-i}),
\qquad
\Omega(X)=TC(X)-DTC(X).
\]

Interpretation:

- **TC** is the total amount of dependence in the group, however structured.
  It is by construction type-blind: a d-variable group can carry the same TC
  through a shared cause or through a distributed constraint.
- **DTC** is the dual view: each variable measured by how strongly the rest
  of the group pins it down. In redundant data, one copy strongly constrains
  every other variable; in synergistic (parity) data, any d − 1 members
  determine the remaining member *exactly*.
- **Ω** is their difference: positive indicates redundancy-dominated
  dependence, negative synergy-dominated, near zero inconclusive. Ω is a
  summary of the redundancy/synergy *balance*, not a decomposition of the
  dependence into signed parts, and the paper's vocabulary follows that —
  regimes are described as redundancy- or synergy-dominated, never as "the
  synergy" or "the redundancy" of a group.

Entropies are estimated with the empirical plug-in estimator; every marginal,
reduced-joint, and full-joint entropy also receives the Miller-Madow
correction (K − 1)/(2n ln 2) from *its own* occupied-cell count, TC-MM and
DTC-MM combining them with the signs implied by the identities
TC = Σᵢ H(Xᵢ) − H(X) and DTC = Σᵢ H(X₋ᵢ) − (d − 1) H(X). One accounting note
that must not be stated loosely: at *noiseless* d = 3 parity the plug-in TC
bias and the Miller-Madow correction cancel exactly (marginal supports
K_i = 2, joint support K_J = 4). At any positive flip noise the joint
occupies all 2^d cells (K_J = 8 at d = 3), so the cancellation no longer holds
and MM-TC sits below plug-in TC by (K_J − 1 − Σᵢ(K_i − 1))/(2n ln 2) — small
(≈ 0.006 bits at n = 500, d = 3) but exactly the kind of accounting the
paper should not get wrong. Experiment 3's operating points all have
positive noise, so the qualified statement is the one that applies.

## E3.3 Data-generating processes

One dependent group of size d ∈ {3, 4, 5} plus p_n = 5d mutually independent
Bernoulli(1/2) nuisance variables, appended after the group. The
independence of the nuisances is guaranteed by their construction (independent
coin draws in the generator) and is checked at implementation level; no
pairwise statistical test can certify *mutual* independence — parity is
precisely the counterexample — so checks are implementation sanity checks,
not certificates. All group variables are Bernoulli(1/2) marginally under
both generators; only the dependence anatomy differs.

**Redundant common cause.** Z ~ Bern(1/2) hidden; E_i ~ Bern(η_R)
independent; X_i = Z ⊕ E_i. Every pair carries visible dependence through the
shared cause.

**Synergistic parity.** X₁ … X_{d−1} iid Bern(1/2); X_d = X₁ ⊕ ⋯ ⊕ X_{d−1} ⊕ E,
E ~ Bern(η_S). Any proper subset is mutually independent (at η_S = 0); the
dependence lives only in the full group.

Both generators admit exact population distributions over 2^d outcomes, from
which TC, DTC, Ω, and pairwise MI all follow in closed or exactly-enumerable
form. Two analytic facts organize everything below (both verified exactly):

1. **Parity TC is capped at 1 bit regardless of d:** TC_S = 1 − h₂(η_S),
   where h₂ is the binary entropy function. This is why TC-matching by
   coarse noise grids fails for large d (redundancy can carry up to d − 1
   bits; its overlap with parity's ≤ 1-bit range is a sliver) — and why a
   fixed target below 1 bit is the well-posed design.
2. **For parity, DTC_S = (d − 1) · TC_S exactly, hence Ω_S = −(d − 2) · TC_S:**
   every (d − 1)-subset that includes X_d is independent (the parity
   constraint is reversible: X_i = ⊕_{j≠i,d} X_j ⊕ X_d), so H(X₋ᵢ) = d − 1
   identically in η. Consequence: at a fixed TC target, parity's |Ω| *grows*
   linearly in d while redundancy's stays small — the matched contrast
   sharpens with group size.

## E3.4 Equal-TC design: target matching by root finding (frozen)

**Design rule.** A single population-TC target TC\* = 0.5 bits is fixed; for
each d, each generator's flip rate is solved exactly from its analytic
population TC by numerical root finding (bisection; both TC(η) are monotone
decreasing). The calibration output records the target, roots, residuals
(< 10⁻⁶ bits), and exact population quantities, and is frozen before any
replicate is generated or fitted. Target matching replaces the coarse
{0.01, 0.025, 0.05, 0.10, 0.20}² grid-matching of the reference
implementation, whose best pairs at d = 4, 5 missed the 0.05-bit tolerance by
wide margins (0.110 and 0.087 bits) — unavoidable on a finite grid, given
parity's 1-bit cap (§E3.3, fact 1). The target must satisfy TC\* < 1 bit for
parity to be solvable at positive noise; TC\* = 0.5 gives parity moderate
noise and keeps both generators well inside their ranges.

Exact population quantities at the frozen operating points (analytic;
TC residuals < 10⁻⁶ bits by construction):

| generator | d | η | TC | DTC | Ω | mean pairwise MI |
|---|---|---|---|---|---|---|
| redundant | 3 | 0.136932 | 0.500 | 0.366817 | **+0.133183** | 0.211 |
| redundant | 4 | 0.185584 | 0.500 | 0.349593 | **+0.150407** | 0.116 |
| redundant | 5 | 0.218686 | 0.500 | 0.333617 | **+0.166383** | 0.074 |
| synergistic | 3 | 0.110028 | 0.500 | 1.000 | **−0.500** | 0.000 |
| synergistic | 4 | 0.110028 | 0.500 | 1.500 | **−1.000** | 0.000 |
| synergistic | 5 | 0.110028 | 0.500 | 2.000 | **−1.500** | 0.000 |

(The parity rows follow from the closed forms in §E3.3, fact 2; the
redundant rows are the exact pmf values.) Two setup facts are now sharper
than under grid matching:

1. The two generators carry **exactly equal TC at every d** and equal TC
   *across* d — the design holds dependence strength constant over the whole
   grid, leaving interaction type as the only manipulated axis.
2. The **pairwise certificate persists at the operating points**: the
   redundant regime is pairwise-visible (mean pairwise MI 0.211 / 0.116 /
   0.074 bits by d) while the parity regime is exactly pairwise-invisible.
   Any CorEx behavior common to both regimes is common to dependencies a
   pairwise scan classifies with opposite verdicts. The Ω contrast also
   sharpens with d (§E3.3, fact 2): +0.13 → −0.50 at d = 3, +0.15 → −1.00 at
   d = 4, +0.17 → −1.50 at d = 5.

## E3.5 Latent capacity (a condition, not a constant)

The two generators make different demands on Bio CorEx's discrete factors:

- The redundant generator has a binary common cause; each group's full
  dependence is rendered conditionally independent by the binary latent Z.
  **dim_hidden = 2 is exactly sufficient.**
- A parity group is rendered conditionally independent by the latent
  (X₁, …, X_{d−1}) — **2^{d−1} states**, i.e. 4 / 8 / 16 at d = 3 / 4 / 5. A
  binary latent is structurally insufficient for parity (it can carry only a
  fraction of the group's constraint).

Configuring a single small cardinality for both generators therefore
confounds interaction type with representation capacity: any observed
generator difference could reflect capacity, not structure. The experiment
answers this with two arms, mirroring the capacity treatment Experiment 2
already received (its capability diagnostic and capacity grids):

- **Main arm (interpretation experiment):** configured capacity
  dim_hidden = 2^{d−1} for *both* generators — at least sufficient for
  parity, and overspecified (harmlessly) for redundancy. Under this arm,
  differences between generators cannot be attributed to capacity; any
  capacity-remnant question is closed in advance.
- **Sensitivity arm (practical estimator diagnostic):** dim_hidden = 2, the
  cardinality matching redundancy's mechanism. This arm characterizes what
  the *standard small-configured* estimator recovers of each structure —
  explicitly expected to disadvantage parity, and reported as such, never
  pooled with the main arm.

Optimization difficulty is a third confound, addressed the same way the rest
of the program addresses it: Bio CorEx runs with n_repeat = 5 EM restarts
(keeping the highest-TC solution, the pipeline's standard), and a **favorable
pilot precedes any full-grid commitment** (the program's standard gate; see
§E3.11). If ordinary EM still fails on parity at sufficient capacity, the
contingency is the Experiment 2 pattern: a ground-truth-informed
initialization (`uses_ground_truth_init = True`, flagged, permanently
excluded from recovery summaries) as a capability statement only.

## E3.6 Experimental grid

For each arm (main / sensitivity), generator ∈ {redundant, synergistic},
d ∈ {3, 4, 5}, n ∈ {500, 5000}:

- 32 independent replicate datasets, root seeds 4100–4131; component seeds
  (dataset generation, CorEx initialization, characterization sample —
  §E3.7) derived from the root seed by the program's serialized-hash
  convention;
- dataset = dependent group (d columns) + 5d nuisance columns (p = 6d ≤ 30);
- one Bio CorEx fit per replicate at 3 requested factors — more than one, so
  the fit must *choose* which combinations earn factors.

2arms × 2gens × 3d × 2n × 32 = 768 fits at full scale. No partitioning,
routing, or global-versus-local comparison is involved: this is the one
experiment in the program that diagnoses interpretation rather than
localization, and the absence of a partition step is deliberate.

## E3.7 What is measured (per replicate)

**Ground-truth side (before and independent of any fit):**
TC, DTC, Ω (plug-in and Miller-Madow) on the true group columns of the fit
sample; the exact population TC, DTC, Ω, and mean pairwise MI (§E3.4). The
population values turn the estimation quantities into error measures read
against truth — the same "analytic reference" asset Experiment 2's
diagnostics already exploit.

**Fit side, factor table (all fitted factors, preserving the fit's full
output):** per factor — explained TC, AUPRC, top-d precision/recall/F1/Jaccard
(d = group size), 80% cumulative-MI support size/density/F1, nuisance
false-positive proportions for the top-d set and the support. Factor
selection for the *primary* analysis is by highest explained TC only — never
by recovery ground truth. Score ties in top-d and support construction are
broken by the program's **neutral tie rule** (seeded column permutation
before the descending stable sort), not by variable layout: the group always
occupies the first d columns here, so a lower-index tie-break would silently
favor the true group.

**Two-stage demonstration (the discovered-support measurement).** To
evaluate whether recovery actually provides a useful input to
characterization — not merely whether the estimator works on known groups —
each replicate additionally runs the pipeline's stated two-stage workflow
end-to-end on synthetic data:

1. Fit on the fit sample; select the highest-TC factor; freeze its top-d
   discovered support.
2. Generate a **fresh characterization sample** (independent seed, n_char =
   10⁵ — independent of the fitting data, per the program's data-separation
   convention).
3. Estimate Ω on exactly the discovered-support columns of the fresh sample
   and record it alongside the exact population Ω of the true group.

Support errors are informative, not noise to be tolerated: for parity,
omitting one necessary member destroys the group's population dependence, so
a nearly-correct support yields Ω ≈ 0 rather than Ω < 0 — the end-to-end
procedure's quality, including its failures, becomes measurable. Ω on the
true group's columns of the same fresh sample is recorded as an estimator
sanity reference.

**Fit-diagnostic side:** convergence, runtime, negative-MI count, valid
factor count, fit status. **Failed fits are data**: a replicate whose fit
raises still stores its ground-truth-side rows with
`fit_status = failed_fit` and the exception note, so a failed fit costs none
of the replicate's information content.

**Required tables:** `redundancy_calibration` (target, roots, residuals, and
exact population quantities per (d, generator) — the audit trail for
§E3.4); `redundancy_info` (per condition × replicate × factor-table-level
row: ground-truth quantities, fit diagnostics, discovered-support result);
`redundancy_factors` (per condition × replicate × factor row).

## E3.8 Pre-registered readings (outcome-neutral)

Three patterns are distinguished by the run; the paper takes whichever the
data support. Nothing in this section is a result.

| Outcome | Supported interpretation |
|---|---|
| Both groups recovered well, with similar summaries | Those summaries do not distinguish the two regimes under the tested settings; the paper describes factors as localized multivariate dependencies and restricts synergy claims to ground-truth-informed analyses (Experiment 2's parity structure, where the pairwise-independence proof sits in the DGP) or to external signed measures such as Ω |
| Recovery or summaries differ systematically | The differing quantity is a candidate type signal; characterize it — including capacity and optimization explanations via the two arms — before generalizing |
| One or both generators recovered poorly | The experiment primarily identifies a discovery limitation; report which regime and which arm |

Three pre-commitments on inference:

1. **Nonsignificant is not equivalent.** Where practical similarity matters,
   differences are read against pre-specified *equivalence margins* (mean
   generator differences in factor TC with CI held inside ±0.05 bits — the
   same order as the TC-matching residual — as the default margin), not
   against zero.
2. **Scope the claim to what is reported.** A demonstration of similarity
   among the summaries the pipeline records (factor TC, AUPRC, top-d and
   support recovery) does not establish that *all* observable outputs of a
   CorEx fit are type-blind; the learned representation contains more than
   the reported quantities. Claims travel with the summaries named.
3. **The replicate is the unit of analysis**; like-for-like reads are paired
   on the same data; plug-in/MM estimates are read against the exact
   population values rather than against a null (§E3.7); the sensitivity arm
   never pools with the main arm.

## E3.9 Relation to the neural-network experiments

Experiments 4–6 implement the two-stage workflow: Local CorEx identifies node
groups; interventions (masks/ablations) characterize their role
(`exp4_mnist/interventions.py`). Experiment 3 supplies the synthetic
justification *for the architecture*, in advance of and independently of any
particular outcome: the three-instrument decomposition of §E3.1 (recovery /
Ω / interventions) is what the program's NN experiments operationalize.
Which of the §E3.8 outcomes the run supports determines how strongly that
justification can be stated — under the first outcome, "the second stage is
necessitated by what TC measures" becomes a controlled demonstration; under
the others, it remains a design argument with the empirical caveat reported.

Two boundary conditions, held regardless of outcome:

1. **The two characterization stages use parallel instruments, not
   equivalents.** Ω is a distributional statistic on a known joint; it
   certifies dominance type but is computable only where the joint is
   tractable. Perturbation is a causal behavioral probe: tractable on a
   learned network, establishes influence and specificity, and does not sign
   redundancy versus synergy among the nodes themselves. Both exist because
   TC cannot characterize type; neither recovers what the other measures.
2. **The NN experiment descriptions comply with the same restraint and cite
   this reason:** their metrics (behavioral shift magnitude, class
   concentration, specificity against random controls) characterize
   influence, never "synergistic role." That restraint is principled —
   mandated by the type-limit of total correlation — rather than cautious
   phrasing.

## E3.10 Display

Supplementary limitations panel (built by `summarize_redundancy.py`, neutral
titles — outcome language stays out of figure titles until results exist):

1. Top-factor explained TC by generator and d, **one panel per sample size**
   (n = 500 and n = 5000 are separate regimes at these dims; collapsing them
   into one trendline conflates estimation precision with structure), with
   main and sensitivity arms distinguished.
2. Ω by generator and d — plug-in and Miller-Madow side by side against the
   exact population line (§E3.4 table), sample sizes separated.
3. Support-recovery F1 by generator and d, sample sizes separated.
4. The two-stage panel: Ω on the discovered support (fresh characterization
   sample) against the exact population Ω, showing end-to-end
   characterization quality including support failures.

Aggregate tables: mean/SE/95% CI over the 32 replicates per condition, for
the info and factor tables; equivalence-margin CI readout for the generator
differences (§E3.8, commitment 1).

## E3.11 Implementation status and run plan

- Reference implementation: `tnnls_experiments/exp3_redundancy/` — kept as
  the spec record; its coarse-grid calibration is superseded by §E3.4.
- Human-pipeline implementation (run target):
  `tnnls_human_experiments/exp3_redundancy/` — self-contained modules in the
  program's house style (derived seeds, restartable per-condition CSVs,
  sharded parallel runner, neutral ties ported from Experiment 2, failed-fit
  rows), implementing §E3.4 (target matching), §E3.5 (capacity arms),
  §E3.7 (population references, per-factor preservation, fresh-characterization
  two-stage measurement).
- Planned gates before any full-grid commitment:
  1. smoke grid (single d=3 configuration, small n, one to two replicates)
     exercising every table and the sharded runner;
  2. favorable pilot per §E3.5 with a timing probe per (d, capacity) cell —
     this re-budgets the grid, since runtime grows with dim_hidden
     (Experiment 2's timing data: parity at n = 5000 is on the order of
     hours per condition even at modest cardinality);
  3. full-grid launch sized from the pilot, sharded, results-saving.
- Ω permutation nulls are not part of the design: column permutation
  characterizes independence and cannot separate redundancy from synergy.
  With exact population Ω as the reference, estimation bias and uncertainty
  are assessed by direct comparison against truth (§E3.7, §E3.8, commitment
  3) — a permutation null is at best an appendix sensitivity row, not a
  discriminator.