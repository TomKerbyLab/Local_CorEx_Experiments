# Experiment 5: localized factors in a trained network held out of sample (ImageNet)

Companion to `exp5_protocol_freeze.md` (the binding protocol, items F1–F7)
and the `exp5-port` branch README (implementation notes). This document
records what E5 is for, what was run, and what it found. Reference
implementation `tnnls_experiments/exp5_imagenet/`; run target
`tnnls_human_experiments/exp5_imagenet/` (the port). All E5 protocol
decisions (F1–F7) were frozen before outcomes were computed; the two change
that occurred mid-life are logged in the freeze document's change log
(2026-11-05: class-agnostic two-panel case study; DeepDream deferred) and
were written **before** the affected results were read.

## E5.1 Role in the program

E4 (MNIST) established that Local CorEx groups on a small trained network
behave better than their controls under ablation. E5 tests that on a
much larger, deeper, and more diverse setting — ResNet-18 on full ImageNet
validation (50,000 images, 1000 classes) — with the discovery/evaluation
split made explicit: everything that selects (clustering, fitting, support
thresholds) sees only the discovery half; the evaluation half is routed to
clusters afterwards and carries all outcome statistics. E5 is therefore the
program's held-out generalization test for the identify-then-intervene
architecture: Local CorEx identifies feature groups; the exact sparse
ablation Δz = −W[:,S]h_S on cached pooled features measures what each
group contributes to predictions.

## E5.2 Pipeline (frozen, F1–F2)

ResNet-18 (`IMAGENET1K_V1`), representation = 512-d pooled pre-FC
activations, logits = Wh + b. Features cached once and **demonstrated**:
fresh forward passes on 2,000 stratified decoded images agree with cached
logits to max absolute difference 3.6e-5 (frozen tolerance 1e-4). Discovery
half = first 25 images per class (HF row order); evaluation half = the
remaining 25. Standardization statistics, PHATE embedding (10-d, seed 7000),
k-means (k=100, n_init=50, seed 7001), all CorEx fits (M=10 per cluster,
top-3 factors retained), and q=0.80 cumulative-MI supports are computed on
discovery data only. Evaluation images are routed to clusters by the frozen
fallback (nearest discovery neighbor in standardized 512-d space); each
cluster's evaluation cohort = all routed images (in-cluster, `source`) plus
an equally sized stratified off-cluster control sample, and is shared by
every factor and every control (F3).

## E5.3 Interventions and controls (F3)

For each (cluster, factor) cell: ablate the factor's support at threshold q
(exact sparse update, unit-tested against a full final-layer forward);
matched random supports of equal size (5 draws, seeds derived from the
factor key); and a univariate logit-association control support of equal
size. All interventions run on the identical cohort. The localization
contrast L_j = (tv_source − tv_off)/(tv_source + tv_off) is reported
**control-relatively**: LC sits beside the L_j distribution of its
size-matched randoms and the univariate support's L_j, paired per cluster
with cluster-level bootstrap CIs (F4).

## E5.4 Executed run

Branch `exp5-port` (summary commit 3b1313c; case-study commit 45cd0bb).
300 Local CorEx fits across 100 clusters, **zero failures**; 300
main-stage q=0.80 ablation cells, zero failures; 26/26 unit tests.
Per-cluster size distributions written before any systematic table (F2):
discovery median 161, min 17, max 1669; 14 clusters below 50 evaluation
images flagged `low_sample`; 17 factor rows excluded below the 30-observation
floor and reported in `imagenet_coverage.csv`. The systematic survey covers
282 (cluster, factor) pairs in 94 clusters.

## E5.5 Systematic results

All numbers from the executed run (`outputs/summaries/imagenet_*` on
`exp5-port`); paired stats resample at the cluster level (F4).

1. **Selected supports localize, and random supports do not.** Paired
   LC − random Δ tv_source **+0.214** (95% CI [0.192, 0.238]); paired ΔL_j
   **+0.263** (CI [0.232, 0.296]). L_j distribution: LC median **0.263**
   with 97.5% of factors positive, versus random median 0.005 with 52%
   positive. Locally-selected units move the factor's discovery region far
   out of proportion to what deleting the same number of arbitrary units
   does.
2. **Univariate supports are bigger, not better localized.** The univariate
   control yields larger raw TV (Δ tv_source −0.019, CI [−0.028, −0.012])
   and slightly *less* localized shifts (ΔL_j −0.017). LC trades raw
   magnitude for localization; the paper's value-of-selection claim uses
   the localized/useful framing, never "univariate hurts less" or "LC
   damages most".
   **[2026-11-05 dated correction].** The data define the pairs as
   diff = LC − control: with ΔL_j = −0.017 (CI [−0.026, −0.009]), the
   univariate control's localization contrast is slightly *higher* (more
   localized) than LC's, and its raw TV is larger — the opposite of this
   paragraph's original wording. The corrected reading ("the prediction-
   guided single-unit selector matches or exceeds LC on both aggregates")
   is adopted in `paper/manuscript.md` §7; this section's wording should
   be regenerated against the store when the two-panel CSVs are next
   written.
3. **Dominant-class concordance is modest:** the factor's dominant class
   appears among the five most-decreased outputs for the most-affected true
   class in 0.308 of cells (CI [0.250, 0.366]). Factor identity (by MI) and
   per-cell dominant-class behavior agree only moderately — an honest
   dissociation, echoed by the case study (E5.6).
4. **TC ↔ effect magnitude is modest:** Spearman ρ ≈ 0.29–0.37 between
   factor TC and tv_source — larger TC weakly predicts larger behavioral
   effect, as in E4.

## E5.6 Case study: the 2026-11-05 re-freeze and the two panels

Under the frozen ≥100-discovery-image African-grey rule the selection was
degenerate: cluster-81 has only 12/248 African greys. The 25+25 per-class
split makes a pure ≥60-image cluster impossible for *any* class (max 50
images per class in the whole 50k set). The freeze change log (written
before any of the following results were read) re-froze the case study
class-agnostically with two pre-designated panels, both reported
regardless of outcome; DeepDream panels were deferred (user decision:
subjective, exploratory, not needed for the claims). The panels:

- **Primary — cluster-21 (great grey owl, 47/73 = 64.4% purity).** Two
  retained factors (MI with the owl logit 0.69 and 0.59 bits). At q=0.80
  the LC support ablation collapses the owl class probability on the
  routed evaluation cohort (39 in-cluster + 39 stratified off) by
  **−0.86 / −0.85** versus **−0.05 / −0.09** for the matched random
  supports — and the classwise shift is entirely the owl class
  (all other per-class shifts ≤ 1e-6, with mass returning diffusely
  across classes); top-1 accuracy on the cohort falls 0.27–0.42. The
  localization contrast is 0.58 / 0.56 against randoms' ≈ −0.40 and the
  univariate control matches the owl collapse (−0.85 / −0.70).
  **[2026-11-05 flag — recording left as written.]** The "≤ 1e-6" clause
  conflicts with exact conservation of the class-mean probability vector
  (a −0.86 drop requires ≈ +8.6e-4 per mean elsewhere); either the
  quantifier refers to a different aggregation than the class-mean shift
  Δ_{y,c}, or it is a record error. It must be reconciled against the
  `case_two_top_shifts` store before submission; the manuscript states
  only "the owl class dominates; conservation places the freed mass
  across the remaining classes". Two
  dissociations worth a sentence in prose: the factors' own MI-dominant
  logits (206, 171) are *not* the owl class; and the per-class
  most-affected label lands on a second bird class present in the cohort
  (the per-class TV aggregates all of a class's redistribution, while the
  owl delta is concentrated on the owl coordinate).
- **Secondary — cluster-85 (African grey, 46/46 = 100% pure, `low_sample`:
  evaluation cohort 24 < the 30-image floor; reported as illustrative).**
  Factors 2 and 0 (MI with the grey logit 0.41 and 0.07 bits). LC
  ablation collapses the grey probability **−0.28 / −0.35** (q=0.80)
  versus **−0.08** for randoms; the most-affected true class is the grey
  class itself, and the freed probability mass moves to nearby bird
  classes (hummingbird, kite, gallinule, eagle…). Even factor 0 —
  nearly grey-orthogonal by MI — still collapses the grey class
  (overlapping supports), consistent with the class's dependence being
  distributed across the cluster's factors.

**Reading.** The two panels agree: class signal identified on discovery
data survives out of sample on routed evaluation images, is far larger
than its matched random control, and (for the pure cluster) redistributes
semantically. The F3 caveat stands everywhere: the univariate support does
the same job on that axis. This is the quantitative form of the preprint's
qualitative story, with its controls.

## E5.7 Limitations and hygiene

- Pooled-feature deletion is exact arithmetic on cached features: all
  claims are about which statistically-selected feature groups contribute
  to which predictions of the final linear layer (F6 discipline; no
  mechanism/synergy language).
- Case-study panels are single-cluster illustrations; the systematic
  survey (E5.5), not the panels, carries the paper's aggregate claims.
- Run hygiene: the case-study's first cell batch was accidentally written
  to the main `ablation` shard family; it was quarantined by exact
  condition hash and re-run under a dedicated stage (`case_two_ablation`),
  verified not to enter the main-stage tables on regeneration. One latent
  quirk flagged for a future pass: `summarize.paired_lc_vs_control`
  averages control rows across thresholds rather than matching q=0.80;
  the committed systematic tables are unaffected except marginally for
  the three cluster-81 case factors.
