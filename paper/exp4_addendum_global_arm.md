# Experiment 4 addendum — the global CorEx comparison arm (and Bio verdict)

Addendum to `exp4.md` (on the `worktree-exp4-prep` branch). It records the
analysis the paper-plan review (fb#2) added *after* E4's account of record
was frozen: the global-CorEx rung of the value-of-localization ladder,
which the original E4 document deliberately excluded ("no claim that
locality itself improves discovery is in scope" — the §E4.2 note following
decision 8). The paper now makes a scoped version of that claim, so the
exclusion is reversed as a documented change: the design below was frozen
on 2026-11-04, before any of its outcomes were computed; §A.4 records the
first readings (with the recomputations the paper table will need), and
§A.5 records the Bio expansion verdict.

## A.1 What this arm tests

Comparison 3 of the value-of-localization ladder: whether the
*localization* step adds usefulness within the same factor-learning
framework. Rungs 1 (size-matched random supports) and 2 (univariate
supports) already ran inside E4; this rung swaps the **partition** — one
global fit on all discovery data instead of 20 local fits — while holding
everything else fixed: estimator (Linear CorEx), ranking rule (top-5 by
explained TC with the seeded tie-break), supports (cumulative-MI at the
same thresholds with the same tie rule), intervention machinery
(identical activation masking via the same TrunkEvaluator), the identical
held-out test cohorts, and matched-random control generation.

## A.2 Frozen design

- Fits: one global Linear CorEx per (model seed 5100–5104 × layer H1/H2/H3
  × input condition {output_aware, hidden_only} × budget M ∈ {10, 100}).
- M=10 is per-fit parity with the local fits (the workflow-parity
  reading); M=100 is factor-budget-matched (20 clusters × 10 factors) and
  doubles as the neural capacity control, complementing the Gaussian
  capacity grid.
- Supports at the primary thresholds {0.6, 0.7, 0.8, 0.9}; retained top-5
  factors per fit; 20 size-matched random supports per support (the local
  output-aware arm's control count).
- Evaluation: identical test trunk as every E4 arm; per-cluster
  image-level TV recorded per intervention (fixed test-set cluster
  assignment) so spatial concentration is comparable across arms.
- Data discipline: discovery-only fitting; supports/ranks never see
  evaluation data; failure rows recorded; factors within a seed are
  pseudo-replicates — the 25 factor units (5 seeds × top-5) must be
  labeled as such in the paper, or aggregated to seed level (n=10) for
  its primary contrasts.

## A.3 Fit facts

- All 60 fits converged; standard near-singular CorEx warnings logged,
  nothing else irregular.
- Runtime: M=10 ≈ 5 s; M=100 ≈ 5 min per fit. The marginal factors of the
  larger budget were near-zero-TC: the global top-5 factor TC is
  essentially unchanged between M=10 and M=100 (≈16–35 bits top-5,
  output-aware, H1–H3) — the additional budget buys no additional
  recovered structure on this 10k-sample representation; the neural
  analogue of the Gaussian capacity grid's non-monotone global behavior.

## A.4 First readings (paired global-LC minus its 20 matched randoms;
unit = 25 factor cells, per (layer, condition, budget, threshold) —
`results/global_arm_contrasts.csv` on branch `exp4-global-arm`)

- **Potency**: global supports ΔT_max > 0 with CI excluding zero in
  nearly every cell (the single exception: output-aware L1/M=100/q=0.6);
  e.g. output-aware L2, M=10, q=0.9: **+0.762** (CI [0.675, 0.849]);
  L1 q=0.9: +0.561. Larger supports drive larger effects (H1 q=0.9
  supports ≈ 190/500 units).
- **Concentration**: normalized-entropy deltas negative and
  dominant-cluster-fraction deltas positive vs randoms — global factors
  produce class-concentrated changes for their size.
- **The layer-specific negative**: at H3 the contrast vs randoms is small
  (q=0.9: +0.03 T_max) — an order below the H1/H2 contrasts and below the
  local arm's H3-specificity findings (LC−univariate +0.053 at q=0.9;
  H3 class-concentration prevalence 0.370). Together: **global fits are
  potent but blunt; localization's behavioral payoff concentrates at H3**
  — the layer pattern E4's account of record established independently.

Scoped claim for paper §6 (wording to survive the full review): under
fixed estimator, ranking, support, and intervention machinery, replacing
the 20-cluster local fits with one global fit shifts which groups support
the readings — global groups match random effects at H3 and gain power
primarily by raw support size at H1/H2, while local groups carry the
layer's class-concentration findings. Remaining tabulation before prose:
the cross-arm (global vs local) size-matched comparison from the raw
shards; seed-level aggregation of the headline deltas.

## A.5 Bio CorEx feasibility verdict (fb#2 follow-up)

Pilot (`bio_pilot_e4.py`, branch `exp4-global-arm`; discovery-only,
subprocess-isolated, 5-min per-fit cap; frozen before outcomes; full row
table `results/bio_pilot_e4.csv`): feasibility is **size-dependent, in
exactly the way the linear-vs-Bio comparison can state honestly** —

- Converged within the cap: H2 (≈410 vars) and H3 (≈310 vars) fits at the
  two smaller discovery clusters (n=290: 240 s/178 s; n=445: 141 s/130 s),
  with healthy recovered structure (top-5 factor TC 39–69 bits, zero
  degenerate factors).
- Timed out: **every** H1 fit (≈510 vars), the larger cluster (n=1113) at
  all layers, and **all four global fits** (n=10,000).

⇒ A **local-only** Bio sensitivity at H2/H3 on the smaller clusters would
run (single seed; ~2–4 min/fit), but the reviewer's crossed global/local
design is **infeasible on its global side** (10k-sample Dense-mode Bio),
and a two-cluster × two-layer × one-seed Bio comparison is too thin to
carry the estimator-generality claim it would be added for. **Verdict:
the Bio expansion is declined for this paper**, with the size-dependence
itself recorded as a fact (Bio's Dense-mode cost scales with sampled
observations × variables, and the linear pipeline's advantage is not
merely statistical). Estimator generality remains evidenced by the
synthetic grids: E1's bio_check grid (oracle 0.992 / data-driven 0.988 vs
global 0.659 under Bio — the localization gain reproduces under the
alternate estimator) and E2's full Bio-vs-Linear grids. A bounded revisit
(subsampled ~100-unit activations) belongs to the companion direction,
not this paper.
