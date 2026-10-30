# Experiment 3: TC-matched redundancy versus synergy — a type-blindness diagnostic

Companion to `simulation_methodology.md` (which covers Experiments 1 and 2).
This document records the motivation, methodology, and results of
Experiment 3 — reference implementation `tnnls_experiments/exp3_redundancy/`,
human-pipeline implementation `tnnls_human_experiments/exp3_redundancy/`
(the latter is the run target and implements the redesign described in
§E3.4 and §E3.8). Analytic population values of the data-generating
processes are exact mathematics and are reported as design facts, not as
results; everything else that is empirical is reported from the executed
run in §E3.12.

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
The executed run supports the second §E3.8 outcome (§E3.12): discovery
quality is structure-dependent at equal total correlation, so the
second-stage instrument — an external signed measure or a behavioral probe,
according to what is tractable — is an empirical necessity for type
characterization, not merely a design argument.

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

Supplementary limitations panel (built by `summarize_redundancy.py`, one
panel per experiment — main and sensitivity arms are separate panels and are
never pooled):

1. Top-factor explained TC by generator and d, with the exact population TC
   line and the sample sizes kept distinct (marks n = 500 and n = 5000).
2. Ω by generator and d (plug-in) against the exact population line
   (§E3.4 table) and the zero line.
3. Support-recovery F1 of the top-TC factors, by generator and d.
4. The two-stage panel: Ω on the discovered support against the exact
   population Ω — end-to-end characterization quality, support failures
   included.

After the executed run (§E3.12), the panel title carries the findings
language ("CorEx discovers only the redundancy-dominated structure;
Ω separates the types on recovered supports"); the figure was neutral-titled
before results existed (the §E3.12 outcome selection is a findings
decision, made once, in that section — not per-panel).

Aggregate tables: mean/SE/95% CI over the 32 replicates per condition, for
the info and factor tables; equivalence-margin CI readout for the generator
differences (§E3.8, commitment 1).

## E3.11 Implementation status and run record

- Reference implementation: `tnnls_experiments/exp3_redundancy/` — kept as
  the spec record; its coarse-grid calibration is superseded by §E3.4.
- Human-pipeline implementation (run target):
  `tnnls_human_experiments/exp3_redundancy/` — self-contained modules in the
  program's house style (derived seeds, restartable per-condition CSVs,
  sharded parallel runner, neutral ties ported from Experiment 2, failed-fit
  rows), implementing §E3.4 (target matching), §E3.5 (capacity arms),
  §E3.7 (population references, per-factor preservation, fresh-characterization
  two-stage measurement), and the §E3.5 contingency
  (`parity_capability.py`, the Experiment 2 capability-diagnostic pattern).
- Executed, in order (all on the human pipeline): the smoke grid
  (every table and the sharded runner exercised); the favorable pilot
  (2 replicates per cell, both arms) whose per-(d, capacity) timing probes
  budgeted the full grid; the full grid — 768 conditions, both arms,
  32 replicates per cell, zero failed fits, ~64 CPU-h (8 wall-hours at 8
  workers alongside Experiment 2's active grid); the §E3.5 contingency run
  (§E3.12, finding 7). Results and summaries are committed
  (`redundancy_results/`).
- Run-record correction: the first full-grid pass recorded fit-side TCs in
  local_corex's native nats while every population quantity, Ω value, and
  the equivalence margin are in bits. The defect was fixed at the fit
  boundary (factor TCs and MIs reported in bits from `fit_bio_corex`: fits
  themselves are unchanged) and the grid was re-run end-to-end; recovery
  metrics and all order-based statistics were unaffected (positive
  rescaling), only the TC-scale columns changed. §E3.12 quotes the
  corrected run.
- Ω permutation nulls are not part of the design: column permutation
  characterizes independence and cannot separate redundancy from synergy.
  With exact population Ω as the reference, estimation bias and uncertainty
  are assessed by direct comparison against truth (§E3.7, §E3.8, commitment
  3) — a permutation null is at best an appendix sensitivity row, not a
  discriminator.
## E3.12 Results

All numbers below are from the executed run (§E3.11): 768 conditions —
both arms, both generators, d ∈ {3, 4, 5}, n ∈ {500, 5000}, 32 paired
replicates per cell, 5 EM restarts per fit — with **zero failed fits**, at
~64 CPU-h. All fit-side TCs are reported in bits. Population values from
the §E3.4 table are exact.

**Outcome selection (§E3.8).** The data support the **second reading**:
recovery and the reported summaries differ systematically between the
generators in most of the grid — but the differences trace to
*discoverability* (the optimization's access to the structure), not to any
quantity that reads off the interaction type. Where both structures are
recovered equally well, the summaries are equivalent, exactly as the
first reading would predict for those cells. The third reading's condition
(synergistic recovery poor) is reached at n = 5000 in both arms. The
instrument-level findings follow.

**Finding 1 — equal-TC regimes are not behaviorally equivalent, and the
trace is discoverability.** At n = 5000 (main arm; population TC 0.500
bits for both generators), the redundant generator's top-TC factor
captures 0.42–0.50 bits (per d) while the synergistic generator's captures
0.02–0.11 bits — from fits identical in procedure, capacity, and target.
The paired generator differences (§E3.8, commitment 1) are outside the
±0.05-bit equivalence margin in **11 of 12 main-arm cells**; the one
inside (d = 3, n = 500: difference 0.011 bits, 95% CI [−0.021, +0.042]) is
exactly the cell where both structures are recovered perfectly (top-d F1
= 1.0 on both sides): **where the fit succeeds equally, its summaries are
equivalent; where they differ, it is performance, not type reading.** In
the sensitivity arm the differences are 0.43–0.50 bits at every cell —
none inside the margin (redundancy's mechanism-matched latent succeeds
everywhere; parity's is structurally excluded by capacity, §E3.5).

**Finding 2 — the sensitivity arm isolates the mechanism-capacity match.**
With dim_hidden = 2 the redundant top factor captures ≈ 0.50 bits (the
entire population TC) and recovers the group exactly at every cell
(top-d F1 = 1.0, zero nuisance false positives, support membership exact,
Ω-on-discovered-support equal to the population value everywhere), while
the binary latent carries 0.005–0.09 bits of the parity structure. This is
the arm's designed statement (§E3.5): a mechanism-matched latent recovers
the redundant structure exactly at both sample sizes.

**Finding 3 — at n = 500 with high capacity, fitted factors consume
finite-sample nuisance dependence (a capacity-remnant result).** With 25–30
nuisance columns, sample-level fluctuations form a large spurious
dependence pool at n = 500, and a sufficiently large latent consumes it:
main-arm top-factor TC rises to 0.71 / 1.45 bits (redundant, d = 4 / 5)
and 0.54 / 1.06 bits (parity, d = 4 / 5) — *above* the group's 0.5-bit
population TC — with 14–18-column supports, nuisance false-positive
proportions of 0.22–0.7, and degraded top-d F1 (0.82 / 0.69 redundant;
0.80 / 0.16 parity). Every n = 5000 cell of the same arm is clean
(redundant top-d F1 = 1.0, 1.0, 0.94; supports match the group). The
interpretive reading therefore rests on the n = 5000 cells; the n = 500
main-arm rows report the fit's robustness to spurious structure at small
sample with large capacity, a separate capacity-remnant finding.

**Finding 4 — the external signed measure is well estimated from samples
everywhere.** Ground-truth-side Ω (plug-in, and Miller-Madow) on the true
group agrees with the exact population values at every cell in both arms:
within ≈ 0.01 bits at n = 5000 and ≤ 0.04 bits at n = 500 (largest plug-in
gap: redundant d = 5, n = 500, 0.126 vs population 0.166), including the
exact −(d − 2) · TC parity values. The estimator is not the bottleneck:
the signed diagnostic is available whenever the group is known.

**Finding 5 — the two-stage Ω instrument works exactly where recovery
works, and fails loudly exactly where it does not (the designed
end-to-end signal).** Where the top factor's discovered support matches
the true group — every main-arm and sensitivity-arm redundant cell at
n = 5000, all sensitivity-arm redundant cells, parity d = 3 n = 500 —
Ω measured on exactly those columns of the fresh characterization sample
**equals the population Ω** (e.g. +0.133 vs +0.133; −0.501 vs −0.500).
Where recovery fails — parity at n = 5000 (support-match rates 0.09–0.25)
main arm and 0.00–0.09 in the sensitivity arm — Ω-on-discovered-support
**collapses to ≈ 0** (−0.03 to −0.12) while Ω-on-true-group remains at the
population value (−0.50 to −1.50). The end-to-end procedure's quality,
including its failures, is exactly the measurable quantity §E3.7 set out.

**Finding 6 — the parity failure is optimization-mediated.** At n = 5000
the parity fits in both arms converge EARLY into trivial solutions
(top-factor TC ≈ 0 at converged EM; converged fractions 0.34–0.94 across
d) — the trivial fixed point is an attractor precisely where the
structure is hardest to find; redundant fits at the same settings do not
collapse (top TC 0.42–0.50 bits). At n = 500 the runs exhaust the 200 EM
iterations without convergence and the group is found at d = 3 (both
generators) and d = 4 (F1 0.81–0.82), degrading at d = 5.

**Finding 7 — §E3.5 contingency (capability diagnostic): the parity
failure is optimization, not expressiveness.** On the parity group block
alone, one latent factor, both initializations at the same 2^d-state
cardinality (8 replicates per cell; the Experiment 2 pattern of §E3.5,
flagged `uses_ground_truth_init = True` and excluded from recovery
summaries): the **atom-seeded init reaches the analytic population TC at
every cell** — 0.498–0.522 bits per replicate against the exactly-0.500
analytic value, SE ≤ 0.017 — so the model family *represents* the parity
structure at this cardinality. The **default EM init at the same
cardinality** reaches the (global) optimum at n = 500 — 0.45–0.60 bits,
converging to the seeded fit's value (agreement to 4–6 decimals where the
optimum is sharp) — but **collapses at n = 5000**: 0.0000 bits in every
replicate at d = 3 and d = 4, one replicate of eight finding the
structure at d = 5 (mean 0.066 ± 0.188). This localizes the grid's parity
failure (Findings 1 and 6) to EM's basin structure from the default
initialization at larger n — an optimization failure, not an
expressiveness failure — and independently reproduces the n-regime
contrast (the group is found at n = 500, missed at n = 5000) in a minimal
setting with no nuisance columns at all.

**Scoped conclusions (§E3.8, commitment 2).** Within the summaries this
pipeline reports (explained factor TC, fit-side recovery metrics,
discovered-support Ω):

- The fit's practical behavior on equal-TC redundancy-dominated and
  synergy-dominated regimes is **not equivalent** — differences in
  top-factor TC are far outside the pre-specified margin in 17 of 18
  cells across the two arms — but the difference is a *discoverability*
  trace, not a type reading: it tracks whether the optimization finds the
  structure (mechanism-capacity match, §E3.5 arms; EM basins, Finding 6),
  not a feature of fitted TC that could be interpreted as type.
- Consequently, in operation, a fitted run **cannot distinguish "little
  dependence" from "synergy-like dependence missed"** from its TC-side
  summaries alone: both present as low top-factor TC at equal population
  dependence (Finding 1; the informative failure of Finding 5).
- Type characterization therefore requires the second or third instrument
  of §E3.1 — an external signed measure where the joint is tractable (Ω:
  accurate at every sample size, Finding 4) or a behavioral probe where it
  is not — and recovery quality is not a safe proxy for "the structure
  understood": recovered supports carry Ω forward *only when found*
  (Finding 5). This is the empirical support for the identify-then-
  intervene architecture of §E3.9, in the form the second §E3.8 outcome
  licenses.
