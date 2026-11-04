# Experiment 5 protocol freeze — decisions locked before outcome evaluation

Companion to `paper_plan.md` §5. This document freezes every analysis
decision that the reviewer required to be settled *before* E5's outcomes
are computed or read (`paper_plan_feedback.md` item 5). The freeze is
written against the reference implementation
(`tnnls_experiments/exp5_imagenet/{data,runner}.py`) and the program spec
`agent_instructions/05_resnet_imagenet_validation.md`; deviations from the
reference implementation are recorded explicitly. Once the E5 run starts,
none of these may change; if one must change, the change and its reason go
into the change log below **before** any affected result is examined.

## F1. Discovery, routing, and support selection data

- Model: torchvision ResNet-18, `IMAGENET1K_V1` checkpoint, eval mode;
  representation = 512-d pooled pre-FC activations, logits = W h + b; exact
  package versions, checkpoint hash, preprocessing transform, and class
  index mapping recorded with the run.
- Features are precomputed once (cached pooled features + logits), and the
  cache is **demonstrated**, not assumed: fresh forward passes on a decoded
  sample of at least 1,000 images (stratified across classes) must agree
  with cached logits to atol 1e-4 before anything downstream runs; the
  agreement statistics are written into the run log (this is the
  demonstrated implementation check). Unit tests additionally verify
  sparse-update ablated logits equal a full final-layer forward pass and
  that an empty support reproduces baseline.
- Discovery half = first 25 images per class of the 50k validation set
  (HF row order within class; agreed design deviation from canonical
  relative paths, because the HF set exposes no paths); evaluation half =
  remaining 25. The dataset fingerprint is recorded; per-class 25/25
  integrity is asserted, never re-balanced.
- Preprocessing standardization, PHATE embedding (10-d, seed 7000),
  k-means (k=100, n_init=50, seed 7001) and all CorEx fits, retained
  factors, and support thresholds use **discovery-half data only**.
- Evaluation-half routing: record the actual method used (PHATE
  out-of-sample transform if stable; otherwise the spec's nearest-neighbor
  transfer in standardized 512-d space is frozen as the fallback) and do
  not mix methods within a run.

## F2. Local sample sizes

- With 50,000 images over 100 clusters, mean evaluation-half size per
  cluster is ~250 images before class stratification; the run must write
  the **actual** per-cluster size distribution (min, deciles, max) into
  the run report before the systematic table is computed.
- Clusters with < 50 evaluation images are flagged `low_sample` per spec
  (all images evaluated, flag retained in every summary row);
  factors from clusters with fewer than 30 evaluation observations are
  excluded from the systematic table and reported in a coverage table —
  never silently dropped.
- Within-cluster CorEx/support estimation uses each cluster's discovery
  observations only, with the count recorded per factor row.

## F3. Localization controls (contrast must be control-relative)

- The primary systematic localization statistic is NOT the raw factor
  contrast L_j alone; for every factor, the matched random controls and
  (for the case-study cluster) the univariate supports get the **same**
  source/off-cluster contrast computed on the same observations. The
  table reports L_j for Local CorEx side by side with the L distribution
  of its size-matched random controls (Δ and paired 95% CI with the
  cluster as the resampling unit), so source/off composition differences
  (class mix, confidence, activation magnitude) cannot masquerade as
  factor-level localization.
- Source/off-cluster cohorts are identical for Local CorEx and its
  controls on a given factor (spec requirement, unchanged).

## F4. Aggregation and uncertainty

- The reporting unit for the systematic survey is the (cluster, factor,
  threshold) cell; summaries aggregate to the **cluster** level
  (bootstrap over the 100 clusters) for interval estimates. Top-3
  factors per cluster and the single checkpoint are explicitly *not*
  treated as 300 independent replications — the paper text says so where
  aggregate numbers appear.
- Every systematic comparison of Local CorEx vs a control is paired: the
  same evaluation observations, the same thresholds, per-cluster pairing
  before aggregation.
- Fit failures, degenerate factors, and low-sample flags appear as rows
  (failure-as-data); no condition is dropped silently.
- ImageNet class-index mapping, wnid, and the run's git commit are part of
  every result table header row set (program reproducibility rules).

## F5. Case-study provenance and semantics

- The African-grey cluster selection follows the spec's deterministic
  rule, computed on discovery data only; the run records which cluster id
  it produced and that ids are run-specific.
- The case study is a **discovery-selected** example by construction; the
  paper text states that it is not a historically-revisited example unless
  it coincides with the preprint's cluster (in which case both provenances
  are stated).
- Output-class concentration and **semantic concentration are separate
  claims**: class-concentration metrics (entropy/dominant fraction) are
  quantitative; any semantic-concentration statement from DeepDream panels
  or top-shift tables is qualitative and labeled as such, unless a stated
  assessment is added (e.g., WordNet-level similarity analysis of the top
  decreased classes — to be added as a supplementary quantitative companion
  if the qualitative claim is made prominently).
- DeepDream parameters (steps, step size, regularization, jitter,
  preprocessing, starting images) are fixed and identical across the
  displayed factors; generated images are never used to rank or select.

## F6. Readout language (from feedback item 4)

- Any E5 conclusion is phrased as: which statistically identified feature
  groups contribute to which predictions (the intervention is exact: Δz =
  −W:,S h_S on cached pooled features). Phrases not permitted in results
  text without the qualification: "mechanistic structure", "downstream
  synergy", "interacts". "Localized" claims must carry the F3
  control-relative statistics.
- The systematic summaries the survey reports are the spec's seven
  (support sizes/densities; LC vs random; LC vs univariate; L_j
  distribution; TC-vs-magnitude relation; dominant-class decrease
  frequency; failure/low-sample rates) — the section reports them with
  the F3/F4 adjustments.

## F7. Execution status

- Reference implementation exists; features cache must be located or
  regenerated (≈ minutes-class backbone pass) and decoded validation
  images must be available for the F1 demonstration check and DeepDream.
  The freeze takes effect at run start; items F1–F6 are binding from then.
- The claim-to-evidence table's E5 section carries no results until this
  run completes; §4–7 prose of the paper need not wait for E5.

## Change log

| Date | Change | Reason |
|---|---|---|
| (none yet) | | |
