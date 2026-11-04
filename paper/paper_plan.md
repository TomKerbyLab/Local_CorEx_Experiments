# Paper plan — Local CorEx journal extension

This document is the drafting plan for the journal paper (TNNLS target) that
consolidates the experiment program recorded in `simulation_methodology.md`
(Experiments 1–2), `exp3.md` (Experiment 3), and `exp4.md` on the
`worktree-exp4-prep` branch (Experiment 4), plus Experiment 5 (ResNet/ImageNet
validation; reference implementation `tnnls_experiments/exp5_imagenet/`,
not yet ported or run). It states the purpose and claims, the narrative
spine, a section-by-section outline mapping existing source material to paper
sections, the figure/table economy, and the future-work position on
Experiment 6.

It is intended as the brief a human or agent can review *before* prose is
written: structural feedback here is cheap; structural changes after sections
are drafted is not.

---

## 1. Purpose, claims, and scope boundary

The paper advances two claims, each carried by named experiments and by
nothing else:

> **Claim 1.** Localizing a CorEx model improves recovery when multivariate
> dependence varies across regions of the data manifold, provided that the
> partition is informative.

Carried entirely by the synthetic experiments:

- **Experiment 1 — Gaussian grouped-covariance recovery** (linear ground
  truth; oracle / data-driven / random partitions; global-vs-localized
  fits). Status: four grids run and analyzed (commit 2affb68; results
  under the Gaussian pipeline's outputs + summary tables).

- **Experiment 2 — Nonlinear higher-order recovery** (parity and sine-sum
  generators; pairwise-invisible dependence). Status: both core grids
  complete and verified (32 shard batch discrete; 64 worker batch
  continuous); analysis notebook committed. Within E2 the **sine-sum
  generator is the primary test** of Claim 1 under nonlinearity — a role it
  earned via the favorable pilot (oracle BioDense parity-group recovery and
  paired global-fit comparison), with parity retained as the stress case
  characterizing the limits of generic CorEx EM. The two diagnostics
  (capability vs. optimization) separate what CorEx can *represent* from
  what EM *finds*.

> **Claim 2.** Local CorEx factors learned from neural representations
> identify distributed groups of hidden units whose removal produces
> coherent, class-specific changes in model behavior.

Carried by:

- **Experiment 4 — MNIST factor-ablation benchmark** (the principal
  quantitative neural evidence). Status: complete through six external
  review rounds; account of record in `exp4.md` §E4.16; every quoted number
  reproducible from committed shards via
  `analysis_feedback_round3.py` + `reeval_extension_fullprecision.py`.

- **Experiment 5 — ResNet-18/ImageNet validation** (systematic 100-cluster
  × top-3-factor survey on a pretrained network, plus the African-grey
  case study). Status: reference implementation exists
  (`tnnls_experiments/exp5_imagenet/`); **not yet ported to the human
  pipeline or run**; ResNet-18 cached-feature pipeline (50k × 512 pooled +
  50k × 1000 logits) designed and verified-by-construction in
  `experiments/paper_neural_networks/resnet_experiment.ipynb`. E5 is scope,
  and its execution is an open pre-drafting task (see §7).

**Explicit non-claim (load-bearing for the vocabulary throughout).** The
paper does **not** claim that a recovered CorEx factor's statistical
structure reveals the *type* of interaction holding the group together
(redundancy vs. synergy), nor that total-correlation magnitude predicts any
functional property beyond what the intervention instrument directly
measures. This follows from Experiment 3 and is why the paper's neural arm
operates as *identify-then-intervene*: Local CorEx identifies node groups; a
separate, structure-appropriate instrument (the behavioral probe) says what
they are doing. The three-level framework from `exp3.md` is used verbatim
across all sections: **group recovery** (what the pipeline does), **O-
information** (type characterization, computable only on tractable
synthetic joints), **interventions** (behavioral characterization, used
wherever the joint is not tractable).

What the paper deliberately does not cover is recorded as future work in
§6 (the dropout/TC-redundancy question, Experiments' program's Experiment
6).

## 2. Narrative spine

The argument, in five beats:

1. **Problem.** Real data-collections of multivariate dependence are
   regionally heterogeneous: different subsets of variables interact in
   different regions of the manifold. A single global CorEx model must
   allocate its factor budget across all of that variation at once.
2. **Claim 1 (E1, E2).** With a partition (oracle as upper bound;
   data-driven as the realistic case; random as negative control),
   localized fits recover region-specific groups better than the global
   fit — under clean covariance structure (E1) and under nonlinear,
   pairwise-invisible structure (E2 sine-sum), with parity as the honest
   stress case.
3. **The interpretive gap (E3).** Recovery is a statement about *that* a
   group exists and which columns it contains — not what kind of structure
   binds it. With two synthetic generators at **matched total correlation**
   (one redundancy-dominated, one synergy-dominated), the identical
   pipeline recovers both, and TC cannot tell them apart; O-information
   can, where its joint is tractable. Type-blindness is a property of the
   statistic.
4. **The instrument (E4, E5).** On learned networks the joint is not
   tractable, so the second measurement is behavioral: perturb the learned
   factor's support (exact activation masking), compare against
   size-matched random controls, and score classwise prediction shifts.
   E4 establishes this on a trained MLP as a full benchmark
   (clusters × layers × factors × support sizes, discovery/evaluation
   separation); E5 shows the same instrument runs unmodified on a large
   *pretrained* convolutional network — no fine-tuning, supports defined on
   pooled units, ablation by final-layer algebra — and that effects remain
   localized and semantically concentrated. E5's case study is the paper's
   interpretability payoff; its systematic sweep is its robustness payoff.
5. **Consequences.** What the two claims imply for using Local CorEx as an
   analysis instrument on neural networks — and what the pipeline
   deliberately does not tell you (E3's lesson carries into the
   interpretation rules for interventions), leading to §6.

## 3. Section outline and source mapping

| § | Paper section | Content | Source material |
|---|---|---|---|
| 1 | Introduction | Multivariate-dependence heterogeneity; CorEx/TC background; the two claims; contributions list | New prose; pull motivation framing from `00_README_experiment_program.md` §claims |
| 2 | Background and problem setup | CorEx objective, TC, factor supports ($a_{ji} = I(Z_j;X_i)$), cumulative-MI supports, local modeling | Methodology §1–2 excerpts; notation shared |
| 3 | Shared methodology | Paired design; derived seeds; discovery/evaluation data separation; tidy outputs; failure-as-data; one-to-one matching; resampling discipline (replicate = primary unit; cluster/factors are not independent replicates) | `simulation_methodology.md` shared conventions + `00_README` non-negotiables; write once here, referenced by every experiment section |
| 4 | E1 — Gaussian recovery | Design, condition grid, headline results (localized vs global under oracle/data-driven/random partitions), recovery metrics | `simulation_methodology.md` §Gaussian (Methods-usable as written) + results appendix |
| 5 | E2 — Nonlinear recovery | Sine-sum primary result; parity stress case; the two capability/optimization diagnostics | `simulation_methodology.md` §nonlinear + analysis notebook outputs → new results prose |
| 6 | E3 — Type-blindness diagnostic | Matched-TC redundant/synergistic generators; O-information signed contrast; what recovery does/doesn't establish | `exp3.md` (largely assembled; trim internal review-arc material for paper voice) |
| 7 | E4 — MNIST benchmark | Benchmark design; discovery/evaluation separation; ablation mechanism + unit tests; matched random controls; headline effects; classwise specificity; account of record | `exp4.md` on branch `worktree-exp4-prep`; consolidated account §E4.16 is the spine; pre-registration material compresses into Design |
| 8 | E5 — ImageNet validation | Cached-feature pipeline; 100-cluster survey (supports, matched random + univariate controls, source/off-cluster localization contrast); African-grey case study + DeepDream panel | `agent_instructions/05` is the design doc; §5 here is written *after* the run (see §7). Case-study selection rule is deterministic on discovery data only |
| 9 | Discussion | Consequences of Claims 1–2; the identify-then-intervene architecture as the reusable pattern; limitations (E2 parity outcome honestly characterized as an optimization limit; E3 type-blindness; E5 is a validation not replication; TC is type-blind so intervention results describe behavior, not interaction type) | New prose; guard rails from each experiment's interpretation rules |
| 10 | Future work | **The dropout/TC-redundancy question (Program experiment 6) — see §6 of this plan** | This plan §6 |
| A+ | Appendices | Full parameter tables; metric formulas; supplementary grids (capacity arms, sensitivity arms); run ledgers; failure tables; regenerability map | Existing `.md` appendices + summarizer outputs |

## 4. Figure and table economy

Main text (target ≤ 8 display items):

1. **Fig. 1 — E1/E2 headline.** Localized-vs-global recovery, oracle /
   data-driven / random partitions, both generators (one coherent panel;
   sine-sum emphasized per the pilot).
2. **Fig. 2 — Type-blindness (E3).** Matched-TC redundant vs synergistic
   groups; TC vs O-information contrast. This is the pivot figure of the
   paper.
3. **Fig. 3 — E4 intervention overview.** Effect sizes vs matched random
   controls, classwise concentration; the panel figure as regenerated in
   the exp4 round-4 fix.
4. **Fig. 4 — E5 systematic comparison.** Local CorEx vs matched-random vs
   univariate supports: TV, localization contrast.
5. **Fig. 5 — E5 African-grey case study.** DeepDream factor panels +
   top output-probability shifts (the spec's ≤-3-item display rule for E5
   respected: systematic comparison, case-study panel, optional secondary
   panel → supplementary if space).
6. **Table 1 — Benchmark conditions** across E1/E2 (dimensions: d,
   regions, dependence structure, partition method, estimator variants).
7. **Table 2 — E5 ablation summary** (factor keys, supports, TV source/off,
   L_j, dominant-output fraction).
8. Optional: **E4 support-size × layer** table if it earns its space after
   the main figure is laid out.

Supplementary: capacity-arm grids and supplementary thresholds (E1/E2),
E3 sensitivity arms and limitations panel, E4 full distributions +
per-cluster/per-layer tables and the full account-of-record tables, E5
remaining distributions, support-threshold sensitivity (q ∈ {0.6, 0.7,
0.9} on the case-study cluster), DeepDream configuration.

Selection rule carried over from the program spec: no display item may be
selected using held-out evaluation performance; all figures must
regenerate from committed summaries (`outputs/summaries/` + per-shard raw
tables).

## 5. Methods-preamble decision

The single biggest consolidation win: one shared §3 written once, then each
experiment section cites it instead of restating paired design, seeding,
separation, failure handling, output schema. Each experiment section keeps
only what is experiment-specific (its generators/architecture, its metrics,
its controls, its interpretation rules). This is how the current
methodology/experiment docs already factor — §3 is extracted, not authored
from scratch.

## 6. Future work — the dropout/TC-redundancy question (Program E6)

**The question.** Does training with dropout (a manipulation that plausibly
increases multivariate dependence) increase TC in later-layer
representations, and does that statistical increase correspond to greater
functional robustness under factor-guided deletion?

**Why it is future work, not part of this paper.** Its design is a
hypothesis test in a controlled paired setting (10 seed pairs, cloned
initializations, identical minibatch order, dropout as the only
difference), testing four distinguishable sub-hypotheses (H2/H3 Gaussian
TC ↑; H1 unchanged; support density ↑; excess factor-guided deletion
damage ↓ vs matched random controls). Its outcome does not bear on either
claim: Claim 1's recovery comparisons and Claim 2's intervention effects
are unchanged by any possible result, and the paper's vocabulary never
attributes interaction "type" or redundancy status to factors (E3's
lesson). Including it would not change any result in §4–8; it would only
change how the *instrument's statistics* are discussed — which this paper
already handles via the three-level framework. The paper therefore makes
**no dropout/TC-redundancy claim anywhere**, and §6 in the paper is where
the question and its design are laid out for follow-on work.

**What we hope to do with it (the follow-on direction).** E6's seed run is
the first point on a deliberate mapping between TC magnitude and function:

1. **Dose–response.** Sweep the dependence-inducing lever (dropout rate
   0 → 0.9) and other levers (weight decay, label noise, width, training
   duration), and regress TC against excess-deletion damage. E6 is one
   point; the paper is the curve.
2. **The type–magnitude dissociation.** E3 shows TC is type-blind
   (redundant and synergistic structure at matched TC are
   indistinguishable). The follow-on asks whether TC *magnitude*
   nonetheless predicts functional consequence — a cleaner formulation of
   "what does TC measure" than either E3 or E6 alone.
3. **Instrument generality.** Whether the TC↔function relation requires
   CorEx-style grouping at all, or holds under alternative grouping
   schemes.

This material is presented in the paper as *the next question*, not as an
in-progress claim: no expected direction is stated as an expected result,
and the design (including its four separable hypotheses and its
interpretation rules — higher TC alone supports only a statement of
statistical dependence; "functional redundancy" language is licensed only
by the deletion analysis) is given at pre-registration depth in the
appendix pointer.

## 7. Status ledger and pre-drafting tasks

Open items before prose drafting, in dependency order:

1. **E5 execution** — port the reference implementation to the human
   pipeline; locate/regenerate the ResNet-18 cached features
   (50k×512 + 50k×1000; regeneration ≈ minutes-class backbone pass);
   confirm validation-set decode access for the DeepDream case study.
   Then run: PHATE + k-means (100 clusters), 100 × CorEx(M=10) fits,
   top-3 × 80% supports + matched random + univariate controls,
   source/off-cluster evaluation. This is the only blocking run for §8.
2. **E2 results → section prose.** The analysis notebook exists (commit
   5de976b); section 5 of the paper converts its findings to results
   prose, following the same pattern as `exp3.md` §E3.12 (reference
   executed runs, design facts vs empirical results kept distinct).
3. **E4 doc harvest.** `exp4.md` lives on `worktree-exp4-prep` (currently
   at 300f878; nothing pending per the ledger). Pull its §E4.16 account of
   record into paper §7 once the branch is merged to the drafting branch.
4. **Figure regeneration pipeline.** Every display item must map to a
   committed summarizer/figure script reading committed outputs; build the
   figure list from §4 above as a checklist.

Drafting order once the plan is approved: shared §3 → E1 → E2 → E3 → E4 →
E5 → Introduction/Background → Discussion → Future work (§6 of this plan
becomes paper §10) → appendices.
