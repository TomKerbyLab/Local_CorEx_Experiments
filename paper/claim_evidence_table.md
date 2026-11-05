# Claim-to-evidence table (pre-drafting audit)

Requested by the plan reviewer (`paper_plan_feedback.md`, closing item):
for each proposed claim, the **actual result, comparator, effect size,
uncertainty, independent sample count, and strongest qualification** —
built from committed results, not intentions. Every number below cites its
committed source; MISSING items are listed at the end. Values in bold are
the primary numbers; anything computed in this audit (rather than read from
a committed table) is marked **[recomputed]** and reproducible from the
committed raw CSVs.

Paired contrasts use t-based 95% CIs (df = seeds−1 = 15 unless stated).
This audit's read-only recompute scripts live in the job's tmp directory;
any number adopted in the paper must be re-derived by a committed script
before it enters prose.

---

## Claim 1 — Localizing improves recovery (given an informative partition)

### R1.1 Linear regime (E1, Gaussian grids)

| Item | Value |
|---|---|
| Result | overall top-d F1: oracle **0.999**, data-driven (PHATE) **0.980** vs global **0.788**; random **0.809** |
| Effect size | paired PHATE − global Δ **+0.192**, SE 0.007, CI **[0.177, 0.207]**; paired oracle − global decisive |
| Comparator / units | global CorEx fit on the same data; 16 replicates/condition × 27 primary conditions (3 setups × cluster size {50,500,5000} × α {0,0.5,1.0}) |
| Qualifications | homogeneous regime: no localization gain (global = oracle ≈ 1.000); random partitions do not beat global (Δ +0.021, CI [−0.001,+0.042] crosses 0) — the gain needs an informative partition |
| Source | `tnnls_human_experiments/gaussian/ablation_data/summaries/{gaussian_summary.csv, gaussian_contrasts.csv}` |
| Estimator generality (bonus) | **bio_check grid**: oracle 0.992 / PHATE 0.988 vs global 0.659 under the alternate CorEx estimator — same gain |

### R1.2 Nonlinear, pairwise-invisible (E2, sine-sum primary)

| Item | Value |
|---|---|
| Primary convention — **RESOLVED [recomputed, frozen]** (per-(true-group, condition) best top-d match; criterion at n=5000; notebook re-executed on branch `worktree-e2-conventions`) | sine-sum n=5000: oracle F1 **0.337** vs global **0.268** vs linear-oracle 0.256 — ΔF1 **+0.069** CI [0.043, 0.094]; cosine oracle **0.383** vs global 0.319 — Δ **+0.064** CI [0.043, 0.085]; n=500: cosine Δ +0.019 CI [0.008, 0.030], F1 Δ −0.006 ns. Note: a best-match taken jointly over groups ∪ factors (an earlier audit draft, 0.617/0.668) is selection-inflated and **not adopted** |
| Verdict | **F1 gains real and significant at n=5000 (15/16 seeds); cosine below the 0.5 bar (0.383) → localization improves recovery but support-level precision stays moderate; the criterion's strict cosine condition fails** |
| Sensitivity: committed pooled convention | oracle cosine 0.363 / F1 0.295 vs global 0.321 / 0.264; notebook's own criterion → MIXED/REJECTED (superseded in prominence by the primary convention above) |
| Additional qualifications | TC is not a recovery proxy (random partitions explain **more** TC than oracle: 2.287 vs 1.439 per fit); nuisance false-positive fraction ≈ 0.4–0.55 across schemes; pilot numbers superseded by core-grid numbers |
| Source | `tnnls_human_experiments/AI_implemented/exp2_nonlinear/analysis_core_grids.ipynb` (cells 2–6), combined CSVs under `nonlinear_results/{continuous_core,discrete_core}/` |

### R1.3 Parity (E2, stress case)

| Item | Value |
|---|---|
| Result | **Partial recovery, mislocalized supports**: Bio oracle cosine 0.487, F1 0.485 (grid means); paired oracle − global **[recomputed]**: F1 Δ +0.161 CI [0.123, 0.200] at n=5000 (16/16 seeds), +0.179 CI [0.162, 0.196] at n=500 |
| Estimator limit | Linear CorEx fails: n=5000 oracle F1 0.412, **below** global 0.464 (Δ −0.052, CI [−0.093, −0.010]) |
| Capability diagnostic | default-init EM reaches optimum at n=500 (TC 0.45–0.60) but collapses at n=5000 (0.0000 bits in 8/8 replicates at d=3,4); atom-seeded init reaches analytic TC (0.498–0.522, SE ≤ 0.017) → **optimization, not expressiveness** (consistent-with, not proof — the three-distinction wording) |
| Source | same notebook cells 3, 6; `exp3.md` Finding 7 |

### R1.4 Global-capacity controls — RUN (fills the reviewer's factor-budget objection)

| Item | Value |
|---|---|
| Result | Apples-to-apples at equal **total factor budget** (global single fit vs the budget-matched stack of the two local fits, identical support sets and Hungarian/cosine matching): disjoint — global M=5 **0.880** vs local-pooled **1.000** (paired Δ −0.120, SE 0.013, n=16); M=10 global **0.909** vs local **0.996** (Δ −0.088, SE 0.013). Nondisjoint — M=5 both **1.000** (tie); M=10 global **0.649** vs local **0.992** (Δ −0.344, SE 0.022) |
| Interpretation | Localized fits beat global at the same factor budget, so the localization gain is not explained by the local arm's larger budget; in the homogeneous case localization adds nothing (consistent with the informative-partition condition); the nondisjoint non-monotonicity (global degrades from M=5 to M=10) is reported as an observed capacity/estimator behavior, not smoothed over |
| Units | 32 conditions × 16 paired replicates, budget ∈ {5,10}, both covariance setups, linear estimator |
| Source | `tnnls_human_experiments/gaussian/capacity_data/global_capacity/global_capacity_combined.csv` (generated from committed code, commit 5e653b0; summarise via `summarize_gaussian.py` conventions) |

## The interpretive gap (E3) — TC is type-blind

| Item | Value |
|---|---|
| Design fact (population, exact) | Calibration froze both generators at TC\* = 0.500 bits (residuals < 1e-6); population Ω: redundant +0.133 / +0.150 / +0.166 vs parity −0.500 / −1.000 / −1.500 (d = 3/4/5); parity pairwise MI 0.000 vs redundancy 0.074–0.211 |
| Result 1 — detectability, not type | At matched **population** TC, fitted top-factor TC differs by generator: redundant 0.42–0.50 vs parity 0.02–0.11 (11/12 main-arm cells outside the ±0.05 equivalence margin; the one inside is d=3,n=500 where both recover at F1 = 1.0). Read as: the fit traces *discoverability* (pairwise-visible components), not type |
| Result 2 — end-to-end type reading | Where supports match the true group (all n=5000 main-arm redundant + all sensitivity-arm redundant cells + parity d=3,n=500), Ω on the discovered support **equals population** (+0.133 vs +0.133; −0.501 vs −0.500); where recovery fails (parity n=5000, support-match 0.09–0.25), Ω on the support collapses (−0.03 to −0.12) while Ω on the true group stays −0.50 to −1.50 |
| Estimator accuracy | Ω within ≈0.01 bits at n=5000, ≤0.04 at n=500 |
| Units | 768 conditions × 32 paired replicates, 5 EM restarts, 0 failed fits (≈64 CPU-h) |
| Qualification | n=500 main-arm cells carry capacity remnants (14–18-col supports) — main reading rests on n=5000 cells; fitted TC cannot distinguish "little dependence" from "missed synergy-like dependence" |
| Source | `paper/exp3.md` §E3.12 (lines 398–526, worktree paper-plan); raw under `tnnls_human_experiments/exp3_redundancy/redundancy_results/` |

## Claim 2 — Neural factors identify groups with structured behavioral effects

### R2.1 E4 (MNIST, principal evidence; §E4.16 account of record)

| Item | Value |
|---|---|
| Local CorEx vs matched random | T_max paired Δ at q=0.9: H1 **+0.0929** (SE 0.0044), H2 **+0.1661** (0.0060), H3 **+0.1473** (0.0031); H_input Δ −0.168 / −0.375 / −0.335; favorable at every (metric × layer × threshold × seed) stratum; random T_max ≈ 0.005–0.006 | 
| Comparator / units | 141k + 66k + 66k intervention rows: 6,000 LC output-aware + 1,500 LC hidden-only + 120,000 matched-random + 6,000 univariate + 7,500 + 66,000 reeval; **5 model seeds × 20 clusters × 3 layers × top-5 factors — not independent network replicates** |
| Class concentration (layer pattern) | Conjunctive prevalence (T_max ≥ 0.10 & H_input ≤ 0.50): H2 **0.376**, H3 **0.370**, H1 **≤ 0.06**; but H1 is not inert (34.8% of H1 factors strong-and-broad, 5.8% strong-and-selective at q=0.9); factor split at q=0.8 (n=1,500): 15.3% strong & selective / 2.5% strong & broad / 82.3% weak |
| Local CorEx vs univariate | significant, H3-scoped, grows with q: ΔT_max +0.006 (q=0.6) → **+0.053** (q=0.9; SEs 0.0007–0.0015); ΔH_input **−0.116** (SE 0.0048) at q=0.9; H1/H2 within ±0.021 / sign-mixed |
| Dose–response | stronger supports (q=0.95): T_max 0.215 / 0.250 / 0.254 without concentration penalty; q=0.99: specificity declines (H3 prevalence 0.382 → 0.150) |
| TC ↔ behavior | partial Spearman −0.01..+0.32 (positive 11/12 strata); residualized 0.29–0.48 — modest association, not a mechanism claim |
| Concordance | 0.27 / 0.75 / 0.55 (H1/H2/H3) vs measured random reference 0.10–0.13 (influence-language, descriptive) |
| Cohorts (exploratory) | same-digit inside-cluster ≈ outside-cluster (≤ 0.004, SE ≤ 0.002); within-digit heterogeneity ±0.05–0.11; discovery-side association predicts held-out excesses (Spearman 0.32 / 0.65 / 0.67) |
| Qualifications | "distributed deletion" not established (few heavy units vs spread not tested — head-vs-tail/singleton ablation recorded as not run); univariate contrast H3-scoped; concordance descriptive; cohort analysis exploratory |
| Source | `.../exp4-prep/paper/exp4.md` §E4.16 + `results/` CSVs |

### R2.2 The value-of-localization ladder (E4 completeness audit)

| Rung | Status |
|---|---|
| Size-matched random supports | ✅ ran (120,000 + 7,500 + 66,000 rows) — LC favorable everywhere |
| Univariate logit-association supports | ✅ ran (6,000 rows) — favorable H3 only |
| **Global CorEx supports** | ✅ **RUN** (branch `exp4-global-arm`, 5 seeds × {M=10 per-fit-parity, M=100 budget-matched} × 3 layers × both input conditions; paired vs 20 size-matched randoms; per-cluster TV recorded). First reading: global-selected groups substantially exceed their size-matched randoms in effect size and class concentration nearly everywhere (e.g. output-aware L2 M=10 q=0.9: ΔT_max +0.762, CI [0.675, 0.849]), but the *H3* contrast vs randoms is small (+0.03 vs the local arm's H3-specificity finding) — localization's payoff concentrates exactly where the E4 account said it does. Cross-arm (global vs local) size-matched comparison is the E4-doc addendum's remaining step |
| Simple alternative grouping | not run (priority below global-CorEx, above Bio per feedback) |

### R2.3 E5 (ImageNet validation) — RUN (branch `exp5-port`, commit 3b1313c)

| Item | Value |
|---|---|
| Implementation check | cache verified against 2,000 stratified decoded re-forwards (max logit absdiff **3.6e-5** ≤ the frozen 1e-4); 26/26 tests; 0 fit failures; zero ablation cells failed |
| Local CorEx vs matched random (282 pairs in 94 clusters; cluster-bootstrap CI) | paired Δ tv_source **+0.214** (CI [0.192, 0.238]); localization contrast **L_j +0.263** (CI [0.232, 0.296]); L_j distribution: LC median **0.263** (97.5% of factors > 0) vs random 0.0048 (52%) — effects concentrate near the factor's discovery region |
| Local CorEx vs univariate | univariate supports give **larger** raw TV (Δ −0.019, CI [−0.028, −0.012]) and less localized shifts (ΔL_j −0.017) — LC trades raw magnitude for localization; the paper's value-of-selection argument must use the review's framing (useful/structural/lcalized selection, not greatest damage) |
| Dominant-class concordance | c*_j among the five most-decreased classes for the most-affected true class: **0.308** (CI [0.250, 0.366]) — modest |
| TC ↔ magnitude | Spearman 0.29–0.37 (factor TC, unit-MI total) — modest, echoing E4's finding 6 |
| Sample-size discipline | per-cluster size distribution written **before** systematic tables (discovery median 161, min 17, max 1669; 14 clusters < 50 eval images flagged low_sample; 17 factor rows excluded < 30 source obs per the freeze) |
| Case study | ✅ **RESOLVED (re-frozen 2026-11-05, before results)**: class-agnostic two-panel rule — primary cluster-21 (great grey owl, 64.4% purity): LC ablation collapses the owl class −0.86/−0.85 vs random −0.05/−0.09 (univariate −0.85/−0.70), classwise shift is the owl class alone, L_j 0.58/0.56 vs random ≈−0.40; secondary cluster-85 (pure African grey, low_sample, illustrative): grey −0.28/−0.35 vs random −0.08, most-affected class = 87, mass redistributes to nearby birds. Panels agree on the preprint story (LC vs random), F3/univariate caveat in both; DeepDream deferred | 
| Readout rules | pooled-feature deletion is exact arithmetic (Δz = −W:,S h_S): conclusions are about which statistically-selected feature groups contribute to which predictions; no synergy/mechanism language; semantic-concentration statements remain qualitative unless a stated assessment is added |
| Source | branch `exp5-port`: outputs/summaries/imagenet_* (split, clusters, factors 300 rows, supports 46.9k rows, ablation_summary 2,100 cells, systematic LC-vs-random/univariate, localization_contrasts, coverage, case_study) |

## MISSING / unresolved (blocks specific table cells, not the drafting of E1–E4)

1. ~~**E1/E2 global-capacity control grids unrun**~~ — **E1 grid RUN** (results above; commit pending to the Gaussian branch); **E2 capacity grid still pending** if the nonlinear capacity objection is to be answered with data rather than the E1 analogue.
2. ~~**E2 verdict convention**~~ — **RESOLVED**: primary = per-true-group best match (frozen; criterion at n=5000 → REJECTED with significant F1 gain, cosine 0.383); pooled convention kept as the labeled sensitivity. See R1.2.
3. ~~**E4 global CorEx arm**~~ — **RUN** (branch `exp4-global-arm` @ 5edf2f5; addendum `paper/exp4_addendum_global_arm.md`). Remaining for prose: the cross-arm size-matched global-vs-local table and seed-level aggregation of headline deltas.
4. E3 per-cell CI tables — prose ranges + committed CSVs suffice for the table; formal per-cell table needs one pass over `core_info_summary.csv` if wanted.
5. ~~E5~~ — **CLOSED** (run + case study resolved; `paper/exp5.md` is the account of record; case-study commit 45cd0bb, freeze change 9cbef53). DeepDream panels deferred (only relevant if a semantic-concentration figure is later wanted).
6. Bio expansion — **DECLINED** with pilot evidence (addendum §A.5); alternative-grouping baseline (ladder rung 4) remains unrun, priority per feedback only if the global arm's H3 reading needs a further comparator.
7. ~~Case-study freeze change~~ — **DONE**: change logged 2026-11-05 (class-agnostic two-panel rule + DeepDream deferral) before any of the panels' results were read; cluster-81 (the degenerate frozen pick) is superseded.
