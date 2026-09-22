# Local CorEx TNNLS Experiment Program

## Purpose

This directory specifies the experiments needed for the journal extension of the Local CorEx paper. Each experiment has its own specification so that it can be implemented and validated independently. The specifications are written as requirements for a coding agent, not as prose for the final paper.

The experimental program must support two primary claims:

1. Localizing a CorEx model improves recovery when multivariate dependencies vary across regions of the data manifold, provided that the partition is informative.
2. Local CorEx factors learned from neural representations identify distributed groups of hidden units whose removal produces coherent, class-specific changes in model behavior.

The synthetic experiments establish the first claim under known ground truth. The MNIST experiment provides the principal quantitative neural-network evidence. The ImageNet experiment tests whether the phenomenon extends to a larger pretrained model. The dropout experiment tests a separate claim about representational redundancy.

## Files

- `01_gaussian_partition_recovery.md`: grouped-covariance recovery and partition-quality stress tests.
- `02_nonlinear_higher_order_recovery.md`: discrete parity and continuous nonlinear dependence with region-specific interactions.
- `03_redundancy_synergy_diagnostic.md`: demonstrates that total correlation detects both redundancy and synergy but does not distinguish them.
- `04_mnist_factor_ablation_benchmark.md`: comprehensive MNIST benchmark across clusters, layers, factors, and ablation support sizes.
- `05_resnet_imagenet_validation.md`: systematic final-layer ResNet-18 analysis plus the African-grey case study.
- `06_dropout_redundancy_experiment.md`: controlled dropout versus no-dropout comparison.

## Non-negotiable shared conventions

### Terminology

Use **multivariate dependence** as the general term. Use **higher-order interaction** only when the data-generating process establishes a dependence that cannot be detected pairwise, as in the parity experiments. Do not treat every CorEx factor as proof of irreducible synergy.

### Reproducibility

Every run must record:

- Git commit hash, if the implementation is in a Git repository.
- Full configuration after defaults are resolved.
- Random seed.
- Package versions.
- Hardware device and elapsed time.
- Dataset version or checksum.
- Model checkpoint identifier.

Use a single root seed for each replicate and derive component seeds deterministically. For example,

\[
s_{r,j} = \operatorname{hash}(s_r, j) \bmod (2^{32}-1),
\]

where \(s_r\) is the replicate seed and \(j\) identifies data generation, partitioning, CorEx initialization, or random ablation. Do not rely on global random-number-generator state.

### Data separation

Data used to discover clusters, fit CorEx factors, rank factors, choose supports, or select qualitative examples must not be used to estimate final intervention effects. Synthetic experiments use independently generated fitting and evaluation data when out-of-sample quantities are needed. Neural experiments use a discovery split and a held-out evaluation split.

No test-set result may be used to change:

- the number of clusters;
- the number of CorEx factors;
- the support thresholds;
- the class-specificity thresholds;
- the choice of baseline;
- the choice of example shown in the paper.

### CorEx factor representation

For learned factor \(Z_j\) and observed variable \(X_i\), store a nonnegative association score

\[
a_{ji} = I(Z_j;X_i).
\]

If an implementation returns a transformed score, also store the raw returned value and document the transformation. Never replace missing or negative numerical values silently. Flag the fit and record the reason.

The explained total correlation for factor \(j\), denoted \(TC_j\), is used to rank factors. Association scores \(a_{ji}\) are used to define factor supports.

### Cumulative-MI support

Given the hidden-unit or observed-variable scores for factor \(j\), sort them so that

\[
a_{j(1)} \ge a_{j(2)} \ge \cdots \ge a_{j(p)}.
\]

For a requested cumulative fraction \(q\), define

\[
s_j(q)=\min\left\{s:\frac{\sum_{r=1}^{s}a_{j(r)}}
{\sum_{r=1}^{p}a_{j(r)}}\ge q\right\},
\]

and

\[
S_j(q)=\{(1),\ldots,(s_j(q))\}.
\]

If \(\sum_i a_{ji}\le 10^{-12}\), mark the factor as degenerate and do not ablate it. Store the requested fraction, achieved fraction, support size, and support density \(s_j(q)/p\).

### Statistical summaries

The replicate or independently trained model is the primary unit of replication. Multiple factors from the same cluster and model are not independent replicates.

At minimum, report:

- mean;
- standard deviation;
- standard error;
- 95% confidence interval;
- number of valid replicates and number of failed fits.

Use paired differences when methods are applied to the same generated dataset or paired model seed. Preserve all raw results so alternative summaries can be calculated without rerunning experiments.

### Output format

Write raw results in tidy form. Every row must have enough keys to reconstruct the experimental condition. Prefer Parquet for large arrays and CSV for compact summary tables. Store large prediction-shift tensors in compressed NumPy, Zarr, or Parquet form and provide a CSV index.

Use this directory structure:

```text
outputs/
  configs/
  raw/
  summaries/
  figures/
  logs/
  checkpoints/
```

Never overwrite a completed run. A run identifier should contain the experiment name, configuration hash, and seed.

### Failure handling

A failed fit is data. Record it with:

- condition keys;
- seed;
- failure stage;
- exception type and message;
- whether retry was attempted;
- retry configuration, if applicable.

Do not repeatedly retry until a favorable result is obtained. One deterministic numerical-stability retry is permitted if its rule is specified before running the experiment.

## Minimum code deliverables

Each experiment must include:

1. A configurable runner.
2. A small smoke-test configuration.
3. Unit tests for data generation and metrics.
4. A summarization script that reads raw outputs rather than rerunning models.
5. A figure or table script that reads summary outputs.
6. A machine-readable resolved configuration saved with every run.
7. A short run report listing successful and failed conditions.

## Recommended execution order

1. Implement shared metrics, matching, seeding, and result schemas.
2. Run smoke tests for all synthetic generators.
3. Run the Gaussian core grid.
4. Run the nonlinear core grid.
5. Audit and test the neural ablation mechanism.
6. Run one MNIST model seed end to end.
7. Run the remaining MNIST seeds.
8. Run the ResNet cached-feature benchmark.
9. Run the paired dropout experiment.
10. Generate summaries and only then decide which panels belong in the paper.

## Global acceptance criteria

The experiment suite is not complete until:

- all splits and seeds can be reproduced from saved configurations;
- every plot can be regenerated from saved raw or summary outputs;
- no qualitative example is selected using held-out evaluation performance;
- synthetic matching is one-to-one;
- neural ablations alter only the intended hidden coordinates;
- random controls are matched exactly on layer and support size;
- uncertainty is shown for replicated results;
- negative or null results remain in the saved outputs and summaries.

