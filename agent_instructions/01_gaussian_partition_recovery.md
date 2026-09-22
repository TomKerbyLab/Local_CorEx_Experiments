# Experiment 1: Gaussian Grouped-Covariance Recovery

## Scientific question

When the covariance groups differ between two latent regions, does local fitting recover the region-specific dependency groups better than a single global CorEx model? How much of any gain is attributable to partition quality?

This experiment is a controlled test of **localized grouped-covariance recovery**. It is not, by itself, evidence of irreducible higher-order synergy.

## Primary hypotheses

1. Oracle-local Linear CorEx will recover region-specific covariance groups better than global Linear CorEx when the two regions have different group structures.
2. PHATE plus \(k\)-means will approach oracle performance when the regions are separable.
3. A balanced random partition will not reproduce the oracle advantage.
4. When the covariance groups are mostly shared, localization will be approximately neutral at adequate sample sizes and may be worse at small sample sizes.
5. Independent nuisance variables will primarily harm partition recovery; conditional on an oracle partition, factor recovery should degrade more slowly.

## Data-generating process

### Dimensions and true factors

Use \(p_s=20\) signal variables and \(m=5\) true groups of size four. Let

\[
G^{(0)}_1,\ldots,G^{(0)}_5
\]

partition the signal indices \(\{1,\ldots,20\}\), with

\[
G^{(0)}_j=\{4j-3,4j-2,4j-1,4j\}.
\]

Generate two equally likely regions, \(C\in\{0,1\}\). Conditional on \(C=c\),

\[
X_s\mid C=c \sim \mathcal N(\mu_c,\Sigma_c).
\]

Within each true group, use an equicorrelation block with \(\rho=0.7\):

\[
(\Sigma_c)_{ii}=1,
\qquad
(\Sigma_c)_{ij}=\rho
\quad\text{if }i\ne j\text{ and }i,j\in G^{(c)}_k,
\]

and set all other signal-variable covariances to zero. Assert numerically that every \(\Sigma_c\) is positive definite before sampling.

### Interaction regimes

Implement two regimes.

#### Mostly shared

Regions share three of five groups. Keep groups 1 through 3 unchanged. Repartition variables 13 through 20 into two new groups:

\[
G^{(1)}_4=\{13,14,17,18\},
\qquad
G^{(1)}_5=\{15,16,19,20\}.
\]

#### Disjoint

No region-1 group may be identical to a region-0 group. Use

\[
\begin{aligned}
G^{(1)}_1&=\{1,5,9,13\},\\
G^{(1)}_2&=\{2,6,10,17\},\\
G^{(1)}_3&=\{3,7,14,18\},\\
G^{(1)}_4&=\{4,11,15,19\},\\
G^{(1)}_5&=\{8,12,16,20\}.
\end{aligned}
\]

Every signal variable appears exactly once, and no region-1 group is identical to a region-0 group.

### Mean separation

Let

\[
v=\frac{1}{\sqrt{p_s}}(1,\ldots,1)^\top,
\qquad
\mu_0=-\frac{\alpha\Delta}{2}v,
\qquad
\mu_1=+\frac{\alpha\Delta}{2}v,
\]

with \(\Delta=4\). Therefore,

\[
\|\mu_1-\mu_0\|_2=4\alpha.
\]

This exact formula must be used so that \(\alpha=0\) gives identical means and \(\alpha=1\) gives the maximum planned separation. Do not scale the mean shift by sample size or nuisance dimension.

### Nuisance variables

Append \(p_n\) independent nuisance variables:

\[
N_i\overset{\mathrm{iid}}\sim\mathcal N(0,1),
\qquad
N\perp (X_s,C).
\]

All variables are standardized using statistics estimated from the generated fitting dataset before PHATE or CorEx is applied. Store the original, unstandardized data only when storage permits.

### Sample-size convention

The parameter \(n\) denotes observations **per true region**, so the total generated sample size is \(2n\). Both regions must contain exactly \(n\) observations before any shuffling.

## Experimental grids

### Core grid for the main paper

- \(n\in\{50,500,5000\}\) per region.
- \(\alpha\in\{0,0.5,1\}\).
- Regime in `{mostly_shared, disjoint}`.
- \(p_n=25\) nuisance variables.
- 16 paired data replicates with root seeds 1000 through 1015.

Every method must receive the same generated dataset within a replicate.

### Nuisance-variable stress test

- \(n=500\).
- \(\alpha=0.5\).
- Both interaction regimes.
- \(p_n\in\{0,25,100,475\}\).
- The same 16 root seeds.

### Supplementary sensitivity grid

Run only after the core grid passes validation:

- additional \(\alpha\in\{0.25,0.75\}\) at \(n=500,p_n=25\);
- PHATE dimensions \(d\in\{2,5,10\}\);
- requested cluster counts \(k\in\{1,2,4,8\}\);
- CorEx factor counts \(M\in\{3,5,8,10\}\).

Vary one sensitivity dimension at a time around the core condition \(n=500,\alpha=0.5,p_n=25,d=10,k=2,M=5\).

## Methods

Use Linear CorEx as the primary factor estimator.

1. **Global**: fit one Linear CorEx model to all observations.
2. **Oracle local**: fit separate models using true region labels.
3. **PHATE local**: compute a 10-dimensional PHATE embedding, then fit \(k\)-means with \(k=2\), using `n_init=50` and the replicate-derived seed.
4. **Random local**: create a balanced random partition with the same group sizes as the true regions. Use 10 independently derived random partitions per generated dataset.
5. **Raw-space local**: fit \(k\)-means directly to standardized observed variables. Treat this as a secondary baseline.
6. **PCA local**: project standardized data to 10 principal components and apply \(k\)-means. Treat this as supplementary unless it materially changes the interpretation.

Fit \(M=5\) CorEx factors in the core grid. Record convergence diagnostics and factor-specific explained TC.

Bio CorEx may be run on the smaller core conditions as a secondary estimator, but the primary Gaussian conclusions must not depend on it.

## Aligning estimated partitions

For methods returning two clusters, construct the \(2\times2\) contingency matrix between estimated cluster \(\widehat C\) and true region \(C\). Use Hungarian assignment to maximize the number of correctly aligned observations. Save both the raw estimated labels and aligned labels.

For \(k\ne2\) in the supplementary sensitivity analysis, do not force a one-to-one label map. Report ARI and purity directly, then score region-specific factors using the estimated cluster containing the largest number of observations from each true region. If the same estimated cluster is selected for both regions, record that fact.

## Matching learned factors to ground truth

For region \(c\), represent true group \(g\) by a binary vector \(b_{cg}\in\{0,1\}^{p_s+p_n}\). Represent learned factor \(j\) by its association vector

\[
a_j=(I(Z_j;X_1),\ldots,I(Z_j;X_{p_s+p_n})).
\]

Compute the cosine similarity

\[
\operatorname{cos}(a_j,b_{cg})=
\frac{a_j^\top b_{cg}}{\|a_j\|_2\|b_{cg}\|_2}.
\]

Use Hungarian assignment to maximize total cosine similarity across the five learned and five true factors. This assignment is then used for all factor-recovery metrics. Do not rematch separately for each metric.

For the global method, use the same learned factor set and independently match it to the ground truth for each true region. For local methods, use the factor set from the estimated cluster aligned to that region.

## Recovery metrics

For every matched learned-factor and true-group pair, calculate:

1. Cosine similarity.
2. AUPRC using \(a_j\) as the score and \(b_{cg}\) as the binary truth.
3. Top-4 precision, recall, F1, and Jaccard similarity, where the predicted support is the four variables with largest \(a_{ji}\).
4. Nuisance false-positive proportion:

\[
\operatorname{NFP}=
\frac{|\widehat S_j\cap\mathcal N|}{|\widehat S_j|},
\]

where \(\mathcal N\) is the nuisance-variable index set.

Also calculate an 80% cumulative-MI support as a secondary support definition and report its size, density, F1, and nuisance false-positive proportion.

Aggregate first across the five matched factors within a region, then across the two regions within a dataset replicate. Preserve factor-level results.

## Partition metrics

Calculate:

- adjusted Rand index;
- cluster purity;
- aligned assignment accuracy;
- estimated cluster sizes.

Plot factor recovery against ARI across PHATE conditions. This is needed to distinguish partition failure from CorEx failure.

## Required output tables

### `gaussian_runs`

One row per dataset, method, estimated cluster, and partition draw:

- experiment and run identifiers;
- seed;
- \(n,\alpha,p_n,\rho\), interaction regime;
- partition method and partition draw;
- PHATE, PCA, and \(k\)-means parameters;
- CorEx estimator and factor count;
- ARI, purity, aligned accuracy;
- runtime and fit status.

### `gaussian_factor_matches`

One row per true region and matched factor pair:

- all run keys;
- true group and learned factor identifiers;
- factor TC;
- cosine similarity and AUPRC;
- top-4 metrics;
- 80% cumulative-MI support metrics;
- nuisance false-positive metrics.

### `gaussian_summary`

Means, standard errors, confidence intervals, and paired differences relative to global and oracle-local methods.

## Main-paper display

Produce one multipanel figure:

1. Matched AUPRC versus sample size for global, oracle, PHATE, and random partitions, faceted by separation and interaction regime.
2. Matched AUPRC versus nuisance dimension.
3. Recovery versus partition ARI.

Produce one compact table containing the principal paired contrasts at \(n=500\) and \(5000\). Do not reproduce the current very wide table of all raw means.

## Validation tests

1. Every covariance matrix is symmetric and positive definite.
2. Each signal variable belongs to exactly one true group in each region.
3. Shared and disjoint regimes contain the intended number of identical groups.
4. Empirical within-group correlations converge to \(\rho\) for a large test sample.
5. Nuisance variables have zero population covariance with signal variables.
6. \(\|\mu_1-\mu_0\|_2=4\alpha\) numerically.
7. Hungarian partition alignment is invariant to swapping raw \(k\)-means labels.
8. Hungarian factor scoring does not reuse a learned factor.
9. Global and local methods see identical generated observations in paired runs.
10. A deliberately perfect association matrix produces recovery scores of one.

## Completion criterion

The experiment is complete when the core and nuisance grids have finished, all failures are accounted for, the recovery-versus-ARI relationship can be reproduced from saved outputs, and no conclusion depends on an unmatched or many-to-one factor score.
