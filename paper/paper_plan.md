# Paper plan — Local CorEx journal extension — REVISED after external feedback

This document is the drafting plan for the journal paper, revised to adopt
the external feedback on the 2026-11-04 draft plan (recorded verbatim in
`paper_plan_feedback.md`). The experiment inventory and source-doc mapping
are the same as in the first revision; what changed is the claims language,
the relative prominence of Experiment 3, the section consolidation, the
figure economy, and the framing of Experiment 6. Structural changes from
the first version are marked **[fb]** so they can be audited against the
feedback.

Companion document: `claim_evidence_table.md` — the reviewer's requested
claim-to-evidence table (claim / actual result / comparator / effect size /
uncertainty / independent units / strongest qualification), built from the
committed results rather than from the plan's intentions.

---

## 1. Purpose, claims, and scope boundary

**Framing.** **[fb #7]** This is a journal extension of a public preprint
(arXiv 2402.04440) that already presented local CorEx, synthetic recovery
comparisons, and neural group-deletion examples. The paper's contribution
statement must be built around what readers can *now* conclude with
substantially greater confidence: (i) partition quality, capacity, and
recovery are separated more carefully; (ii) nonlinear dependence is tested
and estimator limits are documented; (iii) selected neural examples are
replaced with systematic, held-out intervention evidence; (iv) variation
across layers, factors, and cohorts is established; (v) the evaluation
extends to a large pretrained convolutional network.

Two claims, each carried by named experiments and by nothing else:

> **Claim 1.** Localizing a CorEx model improves recovery when multivariate
> dependence varies across regions of the data manifold, given an
> informative partition — with the qualification **[fb #6]** that an
> informative partition is not the only operative condition: local sample
> size, estimator suitability, optimization, and model capacity all bear on
> outcomes, so capacity controls share main-text billing and the pilot
> history that selected sine-sum as the primary nonlinear test is preserved
> in the reporting. Parity keeps a visible account of what failed and under
> which conditions.

- **Experiment 1 — Gaussian grouped-covariance recovery.** Grids run and
  analyzed; results committed.
- **Experiment 2 — Nonlinear recovery** (sine-sum primary, parity stress
  case, two capability/optimization diagnostics). Convention frozen
  (2026-11-04): primary reading = per-true-group best top-d match per
  condition; sine-sum oracle−global F1 +0.069 (CI [0.043, 0.094],
  15/16 seeds) at n=5000 with support cosine 0.383 — **gains real but
  modest, and the strict cosine condition of the original verdict rule
  fails**; the parity case shows a large localization gain
  (ΔF1 +0.161, 16/16 seeds) with mislocalized supports and Linear CorEx
  outright failing — the honest composite is "localization helps group
  recovery under nonlinearity; support-level precision lags". The pooled
  all-rows convention is kept as a labeled sensitivity reading.
- **Global-capacity controls** (localization does not win by hidden factor
  budget alone): presented in the main argument, not the appendix,
  **[fb #6]**.

> **Claim 2 — revised wording [fb #1].** Local CorEx factors learned from
> neural representations identify groups of hidden units whose removal
> produces **structured behavioral effects**, whose **magnitude**,
> **concentration across output classes**, and **dependence on the
> evaluated cohort** vary systematically across layers and factors — with
> class-concentrated effects at deeper layers an important observed
> pattern. The claim deliberately does not say "coherent, class-specific"
> (it overpackages what E4 found: relatively diffuse H1 effects; stronger
> class concentration in H2/H3; meaningful factor-to-factor and
> cluster-to-cluster variation, including within the same digit), and it
> does not say "distributed" as a functional property: a broad MI support
> establishes statistical association across units and group deletion
> establishes the aggregate effect of removing that support; neither alone
> establishes that the effect *requires* spread. Throughout the results,
> three quantities are kept separate **[fb #1]**: how much predictions
> change (magnitude), how concentrated the changes are across output
> classes (concentration), and which input cohorts experience them
> (cohort).

- **Experiment 4 — MNIST factor-ablation benchmark** (principal
  quantitative neural evidence; complete through six review rounds; account
  of record §E4.16; numbers reproducible from committed shards).
- **Experiment 5 — ResNet-18/ImageNet validation** (systematic survey on a
  pretrained network + African-grey case study; to be run; see §5 and
  §7-E).

**Explicit non-claim (unchanged from v1).** No claim that a recovered
factor's statistics reveal interaction *type* (redundancy vs synergy), nor
that TC magnitude predicts functional properties beyond what the
intervention instrument directly measures — per E3. The three-level
vocabulary stands, with the feedback's sharpening **[fb #3]**: these are
*complementary* measurements — recovery, interaction-type characterization
(computable only on tractable joints), intervention (behavioral) — and the
difficulty of estimating a neural joint is **not** the fundamental reason
interventions are needed (they answer a different question than
O-information even when the joint were tractable).

## 2. Narrative spine [fb #3, #8]

1. **Problem.** Dependence structure varies across regions of the data; a
   global CorEx model allocates one factor budget across all of it.
2. **When localization helps recovery (E1, E2).** Oracle / data-driven /
   random partitions under linear and nonlinear-pairwise-invisible ground
   truth, with capacity controls and the parity stress case.
3. **What recovery does not tell you (E3, concise).** At matched TC,
   redundancy- and synergy-dominated groups look alike; TC magnitude alone
   does not identify interaction type. This is an interpretive safeguard:
   matching TC rules out TC magnitude only — not that all statistical
   properties are indistinguishable, and not the behavioral questions.
4. **What systematic interventions reveal (E4, E5).** The behavioral probe
   — support deletion vs size-matched random and univariate controls on
   held-out cohorts — establishes structured effects, layer-wise and
   factor-wise heterogeneity, and specificity where it appears; E5
   transplants the instrument, unmodified, to a large pretrained network.
5. **What the instrument does not tell you.** Intervention results
   describe behavior, not interaction type; E5's pooled-feature readout is
   exact arithmetic, so it evidences *which feature groups contribute to
   which predictions*, not synergistic downstream computation; "semantic
   concentration" needs a stated assessment to become a quantitative
   conclusion **[fb #4]**.

## 3. Section outline (consolidated) [fb #7]

The five experiment sections consolidate into two macro-sections:
**Controlled recovery** (E1–E2) and **Neural representation analysis**
(E4–E5), each preceded by its motivation beat and followed by the other.
E3 sits between them as a concise subsection (its full experimental detail
moves to supplement — see §4). Operational material the preprint's methods
already don't need (tidy schemas, derived-seed recital, run ledgers,
script mappings) moves to supplementary; the shared *scientific* methods
(paired design, discovery/evaluation separation, one-to-one matching,
resampling discipline) stay in the main text, written once.

| Paper section | Content | Source material |
|---|---|---|
| 1 Introduction | Heterogeneity problem; extension framing; contribution statement (the five advances, §1) | New prose |
| 2 Background & setup | CorEx/TC, factor supports, cumulative-MI supports, the three complementary measurements | Methodology excerpts |
| 3 Shared design conventions | Paired design, data separation, matching, replication discipline, failure handling | Methodology §2–4 (condensed) |
| 4 Controlled recovery | **E1** Gaussian grids + capacity controls; **E2** sine-sum (primary) + parity (stress) + diagnostics; qualifications kept visible (§1 wording) | `simulation_methodology.md` + committed E1/E2 results |
| 5 The interpretive gap (concise E3) | Matched-TC design, O-information contrast, the two explicit distinctions; full experiment → supplement | `exp3.md` (trimmed of the internal review arc) |
| 6 Neural representation analysis: MNIST | **E4** design + account-of-record findings; the value-of-localization ladder (below); layer/cohort heterogeneity foregrounded | `exp4.md` (worktree-exp4-prep), §E4.16 spine |
| 7 Neural representation analysis: ImageNet | **E5** frozen-protocol run; systematic survey + case study | `agent_instructions/05` + E5 freeze doc (§7-E) |
| 8 Discussion | Consequences; limitations with the feedback's three distinctions (nonlinear ≠ pairwise-invisible; capability ≠ successful fitting; consistent-with ↔ proof for optimization limits) | New prose |
| 9 Future work (short) | Dropout/TC-redundancy question — one discussion paragraph + supplement pointer **[fb #7]** | See §6 |
| Appendix | Parameter tables, metric formulas, output schemas, ledgers, E6 pre-registration pointer | Existing appendices |

**The value-of-localization ladder (E4 presentation layer) [fb #2].** Four
named comparisons with what each tests, stated in the main text:

| Comparison | Tests whether |
|---|---|
| Size-matched random supports | selected supports have unusual behavioral effects *for their size* |
| Univariate logit-association supports | single-unit selection gives a comparable specificity result |
| Global CorEx supports | *localization itself* adds usefulness within the same factor-learning framework |
| Simple alternative grouping | CorEx adds anything beyond a simpler grouping method |

Status: the first two ran in E4 (matched-random and univariate arms, e.g.
6,000 univariate supports); the third is deliberately *new* — the E4 doc
records the exclusion of a global arm under the then-scope (decision 8),
which this paper now reverses and says so openly (§7-C). The fourth is a
backup priority (feedback: prefer it over Bio if resources are capped).

## 4. Figure and table economy [fb #8]

Main text (revised target ≤ 8 display items, reviewer-prioritized):

1. **Fig. 1 — Pipeline overview** (accessible): partition → local/global
   CorEx → supports → exact activation/logit interventions → held-out
   behavioral readout. Reviewer-ranked priority #1 for accessibility.
2. **Fig. 2 — E1/E2 headline**: localized vs global under
   oracle/data-driven/random partitions, linear and sine-sum, with
   capacity-control panel.
3. **Fig. 3 — E4 layer/cohort findings**: effect magnitude, class
   concentration, and cohort variation across H1/H2/H3 (reviewer-ranked
   priority over an E3 main-text figure).
4. **Fig. 4 — E4/E5 specificity contrasts**: local-CorEx vs matched-random
   and univariate (and, once run, global) supports at matched deletion
   sizes, common cohorts.
5. **Fig. 5 — E5 African-grey case-study panel** (within the spec's
   ≤3-item E5 display rule; the systematic comparison and secondary panel
   go to supplement **[fb #8]** — a second E5 main-text display was
   dropped).
6. **Table 1 — Benchmark conditions** (E1/E2 dimensions).
7. **Table 2 — Claim-to-evidence summary** (from `claim_evidence_table.md`,
   compact form).
8. Reserved (space-permitting): global/local × Linear/Bio crossed-design
   compact panel from the E4 additions.

Supplementary: E3 full experiment (design, tables, limitations panel) — per
**[fb #3]** unless its empirical findings re-earn main-text space; E1/E2
supplementary grids; E4 full distributions and per-cluster tables; E5
systematic comparison, q-sensitivity, remaining distributions, DeepDream
config; E6 pre-registration pointer.

## 5. E5 protocol freeze (before any outcome evaluation) [fb #5]

E5 runs only after this checklist is confirmed as a written freeze doc
(`paper/exp5_protocol_freeze.md`), so it is auditable:

- **Discovery and routing:** data used for preprocessing, PHATE, clustering,
  CorEx, support selection; evaluation-half cluster assignment rules.
- **Local sample sizes:** the actual per-cluster size distribution at k=100
  (mean ≈ 500 pre-split → ≈250 evaluation-half) with a flagging rule for
  fragile small clusters.
- **Localization controls:** source/off-cluster cohorts can differ in class
  composition, confidence, activation magnitude; the factor's contrast is
  compared against **the same contrast computed for each matched control**.
- **Aggregation and uncertainty:** specify the reporting unit (cluster ×
  factor × support) and that three factors from each of 100 clusters in one
  checkpoint are **not** 300 independent network replications.
- **Case-study provenance:** distinguish a discovery-selected example from
  a historically-revisited one; the African-grey selection rule runs on
  discovery data only.
- **Demonstrated implementation check:** cached-logit algebra vs actual
  masked forward passes on a real decoded sample (per the spec's test
  list; "verified by construction" is replaced by a demonstrated check).

Language governing E5 readout (from feedback #4): interventions act on
pooled features before a linear head (Δz = −W:,S h_S); conclusions are
about which statistically selected feature groups contribute to which
predictions; pooled-feature deletion does not establish synergistic
downstream computation; output-class concentration and semantic
concentration are distinct claims.

## 6. Future work — dropout/TC-redundancy (compressed) [fb #7]

The treatment shrinks to one discussion paragraph + a supplementary
pre-registration pointer. Substance to keep intact: the paired design (10
seed pairs, cloned inits, dropout the only difference), the four separable
sub-hypotheses, and the interpretation rules (higher TC alone licenses
only statistical-dependence language; "functional redundancy" language is
licensed only by the deletion analysis). The dose–response mapping (dropout
sweep + other levers), the type–magnitude dissociation against E3, and
instrument generality are recorded at pointer depth, not as a program. No
expected direction is stated anywhere, and the paper makes **no**
dropout/TC-redundancy claim. Its execution is deliberately sequenced
outside this paper's drafting path (a first run on the existing reference
implementation is underway as program-level context, not as paper content).

## 7. Status ledger and pre-drafting tasks [REVISED]

- **A. Plan revision + claim-to-evidence table** — this document and
  `claim_evidence_table.md` (E1–E4; E5 rows fill in after its run).
- **B. Global Linear CorEx arm for E4 [fb #2]** — new analysis on cached
  activations: per-layer global fits on the 10k discovery set, top-factor
  supports, cumulative-MI thresholds, interventions via the existing trunk
  machinery, with size-matched random controls per global support and
  evaluation on the same held-out cohorts as the local arms; matched
  deletion sizes across arms. The E4 doc's exclusion note (decision 8) is
  explicitly updated.
- **C. Bio CorEx feasibility pilot [fb #7-feedback]** — discovery-data
  only; all three layers plus one global fit; a predetermined range of
  cluster sizes; measure runtime, peak memory, convergence,
  degenerate-factor frequency. Expansion to a global/local Bio arm is a
  **feasibility** decision, never an outcome-driven one. Match deletion
  sizes or pair against size-matched controls (cumulative thresholds may
  differ between estimators); rank factors within each estimator; never
  select Linear vs Bio by reported TC. Outcome-interpretation table from
  the feedback adopted as-is (both-estimators / similar-patterns /
  changed-patterns / narrower-qualification).
- **D. Simple alternative-grouping baseline** — backup priority above Bio
  if compute is constrained.
- **E. E5 protocol freeze + run** — §5 checklist first.
- **F. exp6 seed run** — running (program-level context; not paper content).

Order: A → B (cheap) and E5 freeze in parallel → E5 run concurrent with
drafting E1–E4 sections → Bio pilot decision from C → last: final claim
wording checks (E5 section strength follows its results).

Drafting order once approved: shared §3 → E1 → E2 → E3 (concise) → E4 →
E5 → Introduction/Background → Discussion → Future work (short) →
appendices.
