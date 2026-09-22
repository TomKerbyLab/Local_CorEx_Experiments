# Experiment 6: Dropout and Representational Redundancy

## Scientific question

Does training with dropout increase multivariate dependence or redundancy in later hidden representations, and does any statistical change correspond to greater functional robustness under factor-guided deletion?

This experiment is required only if the paper retains the claim that dropout increases total correlation in later layers. If the experiment is not completed, remove that claim from the results and conclusion.

## Primary hypotheses

1. Models trained with dropout probability 0.5 will have larger global and local Gaussian total correlation in H2 and H3 than paired models trained without dropout.
2. The H1 difference will be smaller or absent.
3. Dropout models will require denser cumulative-MI supports to explain the same fraction of factor-to-unit MI.
4. If increased dependence corresponds to functional redundancy, factor-guided deletion should cause less excess damage relative to matched random deletion in dropout models than in no-dropout models.

The fourth hypothesis is separate from the first three. Higher total correlation alone does not prove functional robustness.

## Controlled architectural ablation

Use the MNIST architecture in `04_mnist_factor_ablation_benchmark.md`, with one and only one experimental difference:

- `no_dropout`: dropout probability \(p=0\) in all three hidden layers;
- `dropout`: dropout probability \(p=0.5\) in all three hidden layers.

Batch normalization remains enabled in both conditions. Hidden widths, initialization, optimizer, learning-rate schedule, batch order, early stopping rule, and data splits must be identical within a paired seed.

For paired root seed \(s\):

1. initialize one base set of weights and batch-normalization parameters;
2. clone that initialization into the two dropout conditions;
3. use the same deterministic minibatch order;
4. use separately derived randomness only for dropout masks;
5. train and checkpoint the two models independently under the same stopping rule.

Use 10 paired seeds, 6100 through 6109.

## Performance comparability

Record discovery and test accuracy, cross-entropy, calibration error, and selected epoch for each model.

Flag a pair if final test accuracies differ by more than 0.5 percentage points. Do not discard flagged pairs. Report the accuracy difference and run a sensitivity analysis that adjusts the redundancy comparison for test accuracy or restricts descriptively to pairs within the tolerance.

Do not alter training duration post hoc to force the desired total-correlation result.

## Evaluation representations

Use the fixed 10,000-image discovery set and fixed 20-cluster assignment from the MNIST benchmark. Run models in evaluation mode and extract H1, H2, and H3 after batch normalization and ReLU.

For every model, layer, and coordinate:

- record mean, variance, zero-activation rate, skewness, and kurtosis;
- flag variance below \(10^{-12}\);
- standardize nondegenerate coordinates before covariance-based TC calculations.

## Statistical dependence metrics

### Gaussian total correlation

For standardized layer representation \(H\in\mathbb R^{n\times d}\), estimate a regularized correlation matrix \(\widehat R\) using Ledoit-Wolf shrinkage. Define

\[
\widehat{TC}_{G}(H)
=-\frac12\log\det(\widehat R).
\]

Use a numerically stable log-determinant. If \(\operatorname{sign}\det(\widehat R)\le0\), record a failure instead of taking an absolute value.

Because layers have different widths, also report per-coordinate Gaussian TC:

\[
\widehat{TC}_{G}^{\mathrm{norm}}(H)
=\frac{\widehat{TC}_{G}(H)}{d}.
\]

Do not use the normalized value to compare models with different effective numbers of nondegenerate units without also reporting that number.

### CorEx-explained total correlation

Fit Linear CorEx with \(M=10\) factors to each standardized representation. Define

\[
\widehat{TC}_{\mathrm{explained}}(H)
=\sum_{j=1}^{10}\widehat{TC}_j.
\]

Store individual factor TC values and the cumulative fraction explained by the top \(k\in\{1,3,5,10\}\) factors.

Gaussian TC and CorEx-explained TC answer different questions and must be reported separately.

## Global and local calculations

For each model and layer, calculate both metrics:

1. globally on all 10,000 discovery observations;
2. separately within each of the 20 fixed Local CorEx clusters.

Define the cluster-size-weighted local mean

\[
\overline{TC}_{\mathrm{local}}
=\sum_{k=1}^{20}\frac{n_k}{n}TC(H\mid C=k).
\]

Also preserve the 20 cluster-specific values. Do not infer that

\[
TC(H)=\overline{TC}_{\mathrm{local}},
\]

because the global distribution also contains between-cluster dependence.

## Factor-support analysis

For each model, discovery cluster, and layer:

1. fit Linear CorEx with \(M=10\) factors to hidden activations only;
2. retain the top five factors by explained TC;
3. construct 60%, 70%, 80%, and 90% cumulative-MI supports;
4. store support sizes and densities.

The primary comparison is the paired difference in support density between dropout and no-dropout models at the same layer, cluster, factor rank, and cumulative threshold. Factor rank is used for pairing because hidden units and factor identities are not aligned across independently trained models.

Do not compare raw hidden-unit indices between the two trained conditions.

## Functional factor-deletion analysis

Functional deletion is needed if the paper uses words such as **robust**, **fault tolerant**, or **functionally redundant**.

To keep the experiment bounded, use:

- all 20 discovery clusters;
- all three hidden layers;
- top three factors by explained TC;
- only the 80% cumulative-MI support;
- five matched random supports per factor.

Apply the exact activation-mask intervention defined in the MNIST benchmark. Evaluate on the untouched MNIST test set and calculate the same classwise probability-shift, TV, entropy, dominant-fraction, and accuracy metrics.

For factor \(j\), define excess effect over random deletion:

\[
E_j=T_{\max,j}^{\mathrm{factor}}
-\frac{1}{R}\sum_{r=1}^{R}T_{\max,jr}^{\mathrm{random}},
\qquad R=5.
\]

Also define the stabilized effect ratio

\[
Q_j=
\frac{T_{\max,j}^{\mathrm{factor}}+10^{-6}}
{R^{-1}\sum_rT_{\max,jr}^{\mathrm{random}}+10^{-6}}.
\]

Compare \(E_j\) and \(\log Q_j\) between paired dropout and no-dropout model seeds. A smaller excess effect in dropout models is consistent with functional redundancy, but interpret it together with support density and classifier accuracy.

## Primary contrasts

For paired seed \(s\) and layer \(l\), calculate

\[
D^{G}_{sl}=TC^{\mathrm{dropout}}_{G,sl}
-TC^{\mathrm{no\ dropout}}_{G,sl},
\]

and analogous paired differences for:

- normalized Gaussian TC;
- global CorEx-explained TC;
- weighted local Gaussian TC;
- weighted local CorEx-explained TC;
- support density at each cumulative threshold;
- excess factor-deletion effect \(E\);
- log effect ratio \(\log Q\).

The paired model seed is the unit of replication. Report all 10 paired differences, their mean, standard error, and 95% paired confidence interval. Supplement with a paired permutation test over seed-level signs if desired.

## Required output data

### `dropout_models`

One row per seed and dropout condition with initialization hash, training configuration, selected epoch, performance metrics, and comparability flag.

### `dropout_representation_stats`

One row per seed, condition, layer, and global/local cluster with activation moments, effective dimension, Gaussian TC, normalized Gaussian TC, CorEx-explained TC, and fit diagnostics.

### `dropout_factor_supports`

One row per seed, condition, cluster, layer, factor rank, cumulative threshold, support size, density, achieved MI fraction, and factor TC.

### `dropout_functional_ablation`

One row per seed, condition, cluster, layer, factor rank, intervention method, and random draw with classwise intervention summaries, excess effect, and effect ratio.

### `dropout_paired_contrasts`

One row per seed, layer, and metric containing both condition values and their paired difference.

## Main-paper display

If the claim is retained, prepare:

1. a layer-wise paired plot of global and weighted-local Gaussian TC;
2. a layer-wise paired plot of CorEx-explained TC;
3. a compact support-density or functional-deletion panel.

The main paper should use no more than one combined figure. Full activation moments, cluster-level results, and functional-ablation distributions can be supplementary.

## Validation tests

1. Paired conditions begin from identical parameter tensors.
2. Minibatch orders are identical within a paired seed.
3. The no-dropout model applies no stochastic mask during training.
4. Both conditions are deterministic in evaluation mode.
5. Ledoit-Wolf correlation matrices are symmetric and positive definite within numerical tolerance.
6. Gaussian TC is approximately zero for simulated independent standardized Gaussian variables.
7. Gaussian TC increases in a simulated correlated-Gaussian test.
8. Local cluster assignments are identical across paired model conditions.
9. Factor-guided and random supports have identical sizes.
10. Functional ablations use test data only after supports are frozen.
11. No pair is excluded based on the direction of its TC contrast.

## Interpretation rules

- Higher Gaussian TC supports a statement about stronger Gaussian multivariate dependence.
- Higher CorEx-explained TC supports a statement that a fixed number of CorEx factors explains more dependence.
- Larger support density suggests that factor information is distributed across more units.
- Smaller factor-deletion damage relative to matched random deletion supports functional redundancy more directly.
- None of these results alone proves a causal mechanism by which dropout creates redundancy.

## Completion criterion

The experiment is complete when all 10 paired seeds have trained or recorded failures, global and local dependence metrics are available for all three layers, factor-support comparisons are complete, and any retained functional-robustness claim is backed by the specified held-out deletion analysis.

