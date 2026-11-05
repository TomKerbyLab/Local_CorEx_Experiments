# Experiment 5 — ImageNet / ResNet-18 (human pipeline port)

Human-pipeline port of `tnnls_experiments/exp5_imagenet` (program spec
`agent_instructions/05_resnet_imagenet_validation.md`, protocol freeze
`paper-plan/paper/exp5_protocol_freeze.md`, items F1-F6).

## Layout

- `data.py` — HF imagenet-1k loading (local cache only), fresh ResNet-18
  IMAGENET1K_V1 feature cache (512-d pooled pre-FC features + 1000-d logits +
  labels), F1 demonstrated cache verification (>=1000 stratified freshly
  decoded images, atol 1e-4 on logits), 25/25 split, integrity stop.
- `clustering.py` — discovery-only standardization, PHATE 10-d (seed 7000),
  k-means k=100 n_init=50 (seed 7001), evaluation routing (nearest discovery
  neighbor in standardized 512-d space, the frozen fallback), per-cluster
  evaluation cohorts (identical for a factor and all its controls, F3) and
  the F2 per-cluster size-distribution report.
- `fits.py` — Linear CorEx on [H_k | L_k] per discovery cluster, M=10,
  cluster-derived seed, top-3 factors by explained TC; MI recorded in nats
  and bits; q=0.80 cumulative-MI supports on the 512 pooled units.
- `ablations.py` — exact sparse-update ablation logits
  (`l^A = l - W[:,S] h_S`), per-cluster cohorts, matched random controls,
  univariate logit-association controls, per-cell metrics (image TV, L_j,
  classwise T_y, H_input, dominant fractions, accuracy changes), compressed
  classwise-shift npz artifacts, tidy top-20 shift rows.
- `case_study.py` — deterministic discovery-only African-grey selection
  (pure `african_grey_rule`), two factors with largest MI with the
  African-grey logit, supports at q in {0.6, 0.7, 0.9} (0.8 from the main
  stage), matched controls, secondary-cluster report.
- `summarize.py` — tidy CSVs in `outputs/summaries`, cluster-level bootstrap
  (F4), coverage table (F2), F3 control-relative localization tables.
- `tests.py` — the spec's 10 validation tests plus extra guards.
- `runner.py` — stage orchestration: `cache,verify,splits,clusters,fits,
  ablations,case_study,summarize`.

## Decisions recorded at port time (before any outcome was computed)

1. The old `resnet_100_feature_*.pkl` exploration cache is NOT reused; the
   cache is regenerated fresh and then demonstrated (F1). The old cache's
   shapes were only used to check the expected format.
2. Preprocessing standardization uses discovery-half statistics only (freeze
   F1 wording). This is a (deviation-from-reference) tightening: the
   reference runner standardized with full-set statistics.
3. Evaluation routing uses the frozen fallback (nearest discovery neighbor in
   standardized 512-d space, spec equation) rather than a PHATE
   out-of-sample transform; the method is recorded in every assignment row.
4. Evaluation cohorts are built once per cluster, independent of the factor,
   so the Local CorEx support and every control use strictly identical
   observations (stronger than per-factor sampling; satisfies F3).
5. Univariate controls are evaluated for every factor, so the spec's
   LC-vs-univariate summary covers the whole survey and the case-study
   cluster also carries univariate controls as required by F3.
6. Full 1000-class classwise mean-shift vectors and per-image summary
   vectors are preserved per cell in compressed npz files
   (`outputs/raw/imagenet/classwise_shifts/`); the tidy top-shift table
   keeps the 20 most decreased/increased classes with class names and wnids.
7. DeepDream is EXCLUDED (job order); test 10 is ported as a source guard:
   no DeepDream module exists and factor-ranking shards are only written by
   the fitting stage.
8. Aggregate statistics use the discovery cluster as the resampling unit
   (cluster-level bootstrap, F4); top-3 factors within a cluster are
   averaged before resampling. Factors from clusters with fewer than 30
   in-cluster evaluation observations are excluded from the systematic
   tables and appear in `imagenet_coverage.csv` (F2).
9. Low-sample clusters (<50 evaluation images) keep all their evaluation
   images and carry the `low_sample` flag in every summary row (F2).
10. MI units: Linear CorEx reports TC/MI in nats; every stored MI value is
    recorded in nats AND bits.
