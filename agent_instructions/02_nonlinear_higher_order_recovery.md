# Experiment 2: Nonlinear and Higher-Order Recovery

## Scientific question

Can a localized CorEx procedure recover region-specific multivariate dependencies when pairwise linear correlation is insufficient? This experiment contains:

1. discrete three-way and four-way parity, which provides genuine pairwise-independent higher-order ground truth;
2. continuous sine-sum dependence, which provides a nonlinear continuous analogue.

The experiment must distinguish the benefit of localization from the capability of the underlying CorEx estimator. Bio CorEx is the primary estimator. Linear CorEx is included as a negative or limited-capability baseline.

## Primary hypotheses

1. Global Bio CorEx will mix or miss region-specific groups when the participating variables change across regions.
2. Oracle-local Bio CorEx will recover the groups substantially better.
3. PHATE-local Bio CorEx will approach oracle performance when the regions are separable.
4. Balanced random localization will not produce the same recovery.
5. Linear CorEx will perform poorly on pairwise-independent parity factors even with an oracle partition.
6. Recovery will decrease smoothly with parity-flip or continuous observation noise.

## Shared region construction

Generate two equally sized regions \(C\in\{0,1\}\), with \(n\) observations per region. Append two routing variables

\[
R\mid C=c\sim\mathcal N(\nu_c,I_2),
\]

where

\[
\nu_0=(-2\alpha,0)^\top,
\qquad
\nu_1=(2\alpha,0)^\top.
\]

Thus the routing-mean distance is \(4\alpha\). Routing variables are included during PHATE partitioning but excluded from factor-recovery scoring. Run a secondary condition in which routing variables are also excluded from CorEx fitting. The primary condition includes them in the CorEx input so that the implementation reflects the actual Local CorEx workflow.

Use the same four true group sizes in each region: two groups of size three and two groups of size four. Region 0 uses

\[
\begin{aligned}
G^{(0)}_1&=\{1,2,3\},\\
G^{(0)}_2&=\{4,5,6\},\\
G^{(0)}_3&=\{7,8,9,10\},\\
G^{(0)}_4&=\{11,12,13,14\}.
\end{aligned}
\]

Region 1 uses

\[
\begin{aligned}
G^{(1)}_1&=\{1,4,7\},\\
G^{(1)}_2&=\{2,8,11\},\\
G^{(1)}_3&=\{3,5,9,12\},\\
G^{(1)}_4&=\{6,10,13,14\}.
\end{aligned}
\]

Each variable appears in exactly one group per region, and no group is identical between regions.

Append \(p_n\) mutually independent nuisance variables. Nuisance variables are Bernoulli\((1/2)\) in the discrete experiment and standard normal in the continuous experiment.

## Part A: Discrete parity

### Generating one parity group

For a group of size \(d\in\{3,4\}\), sample

\[
B_1,\ldots,B_{d-1}\overset{\mathrm{iid}}\sim
\operatorname{Bernoulli}(1/2)
\]

and define

\[
B_d=B_1\oplus\cdots\oplus B_{d-1}\oplus E,
\qquad
E\sim\operatorname{Bernoulli}(\eta),
\]

where \(\oplus\) is XOR and \(\eta\) is the parity-flip probability.

At \(\eta=0\), every proper subset of a parity group is mutually independent under this construction, while the full group is dependent. For \(d=3\), for example,

\[
I(B_1;B_2)=I(B_1;B_3)=I(B_2;B_3)=0,
\]

but

\[
TC(B_1,B_2,B_3)=1\text{ bit}.
\]

Generate the four parity groups independently conditional on the region.

### Discrete grids

#### Core grid

- \(n\in\{500,5000\}\) per region.
- \(\alpha\in\{0,0.5,1\}\).
- parity-flip probability \(\eta\in\{0,0.05,0.15\}\).
- \(p_n=14\) nuisance variables.
- 16 paired replicates with root seeds 2000 through 2015.

#### Supplementary grid

- Include \(n=50\) to show the small-sample failure regime.
- Add \(\eta\in\{0.10,0.20\}\).
- Use \(p_n\in\{0,14,56\}\) at \(n=500,\alpha=0.5,\eta=0.05\).
- Sweep requested Bio CorEx factors \(M\in\{2,4,6,8\}\).

## Part B: Continuous sine-sum dependence

### Generating one nonlinear group

For a group of size \(d\in\{3,4\}\), sample

\[
\Theta_1,\ldots,\Theta_{d-1}
\overset{\mathrm{iid}}\sim\operatorname{Uniform}(-\pi,\pi)
\]

and define

\[
Y=\sin\left(\sum_{r=1}^{d-1}\Theta_r\right)+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,\sigma^2).
\]

The observed group is

\[
(\Theta_1,\ldots,\Theta_{d-1},Y).
\]

Because adding an independent uniform angle modulo \(2\pi\) produces another uniform angle, the noiseless response is marginally independent of each individual angle, even though the full group is dependent. The implementation must empirically verify near-zero pairwise mutual information and nonzero multivariate dependence using a large diagnostic sample.

Standardize all continuous variables using fitting-data statistics before CorEx.

### Continuous grids

#### Core grid

- \(n\in\{500,5000\}\) per region.
- \(\alpha\in\{0,0.5,1\}\).
- \(\sigma\in\{0.05,0.20,0.50\}\).
- \(p_n=14\) nuisance variables.
- 16 paired replicates with root seeds 3000 through 3015.

#### Supplementary grid

- Include \(n=50\).
- Use \(p_n\in\{0,14,56\}\) at \(n=500,\alpha=0.5,\sigma=0.20\).
- Compare sine-sum groups with product groups of the form

\[
Y=\prod_{r=1}^{d-1}X_r+\epsilon,
\qquad
X_r\overset{\mathrm{iid}}\sim\mathcal N(0,1).
\]

The product construction is nonlinear but must not be described as pairwise independent without empirical and analytical qualification.

## Partition methods

Use the same partition definitions for the discrete and continuous parts:

1. Global, no partition.
2. Oracle local, using \(C\).
3. PHATE local, using a 10-dimensional embedding and \(k\)-means with \(k=2\), `n_init=50`.
4. Balanced random local, with 10 random partitions per generated dataset.
5. Raw-space \(k\)-means as a secondary baseline.

Align two-cluster partitions to true regions with Hungarian assignment on the contingency table. Record ARI, purity, aligned accuracy, and cluster sizes.

## CorEx estimators

### Primary

Fit Bio CorEx with \(M=4\) requested factors separately within every partition cell. Record convergence status and factor TC.

### Baseline

Fit Linear CorEx with \(M=4\) to the same standardized inputs. Its performance on parity is a diagnostic of the limitation of covariance-based recovery, not a failure of localization.

If the installed Bio CorEx implementation requires discretization for the continuous experiment, the discretization rule must be fixed before the core run. Use equal-frequency bins estimated only on the fitting data, with 10 bins per variable, and save the bin edges. Also retain a no-discretization result if the implementation natively supports continuous inputs.

## Factor matching and recovery metrics

Use the one-to-one Hungarian procedure in Experiment 1. Ground-truth vectors contain ones for variables in the relevant group and zeros for all other signal, nuisance, and routing variables.

The assignment must maximize cosine similarity and then be held fixed for all metrics. Report:

- matched cosine similarity;
- matched AUPRC;
- top-\(d\) precision, recall, F1, and Jaccard, where \(d\) is the true group size;
- nuisance false-positive proportion;
- routing-variable false-positive proportion;
- 80% cumulative-MI support size, density, F1, and false-positive proportions;
- partition ARI, purity, and aligned accuracy;
- runtime and fit failures.

Aggregate within true region, then within generated replicate. Preserve group-level values.

## Verification of higher-order structure

For each generator and noise condition, produce a diagnostic table using an independent sample of at least 100,000 observations:

1. Maximum absolute Pearson correlation among distinct variables within a true group.
2. Maximum estimated pairwise mutual information within the group.
3. Estimated group total correlation.
4. Estimated total correlation after independently permuting one variable.

The parity generator at \(\eta=0\) should have pairwise MI near zero, group TC near one bit, and permuted group TC near zero. Define numerical tolerances in the unit tests based on estimator bias from independent variables of the same sample size.

## Required output tables

### `nonlinear_runs`

One row per generator, dataset replicate, estimator, partition method, estimated cluster, and random-partition draw.

### `nonlinear_factor_matches`

One row per true region and matched factor, including generator parameters, factor TC, matching metrics, support metrics, and false-positive metrics.

### `nonlinear_structure_diagnostics`

One row per generator, group size, and noise condition containing the pairwise and multivariate diagnostic quantities.

## Main-paper display

Produce one figure with separate discrete and continuous rows:

1. Matched AUPRC by partition method and separation.
2. Matched AUPRC versus noise.
3. Oracle-local Bio CorEx versus oracle-local Linear CorEx.

Include a small diagnostic panel or table showing that pairwise association is absent or negligible while group dependence is nonzero. Full sample-size, nuisance, and factor-count sweeps belong in supplementary material.

## Validation tests

1. Every variable appears in exactly one true group per region.
2. No true group is identical across regions.
3. Parity equations hold exactly before bit-flip noise.
4. Empirical binary marginals are approximately Bernoulli\((1/2)\).
5. Pairwise parity MI is within the prespecified estimator tolerance of zero at \(\eta=0\).
6. Permuting one member of a factor removes its estimated group dependence within tolerance.
7. Routing and nuisance variables are excluded from ground-truth supports.
8. Discrete and continuous result schemas have compatible core keys.
9. Hungarian matching never assigns a learned factor more than once.
10. Global, oracle, PHATE, random, and raw-space methods receive identical observations within a paired replicate.

## Completion criterion

The experiment is complete when both core grids and the structure-diagnostic runs finish, all failures are recorded, and the saved outputs permit separate conclusions about localization, estimator capability, nonlinear recovery, and genuine pairwise-independent higher-order dependence.

