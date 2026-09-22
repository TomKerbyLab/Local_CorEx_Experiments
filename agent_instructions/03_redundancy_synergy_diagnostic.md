# Experiment 3: Redundancy Versus Synergy Diagnostic

## Scientific question

Does a high total-correlation factor necessarily imply a synergistic higher-order interaction?

The answer is no. This diagnostic is intended to show that CorEx and total correlation can identify a dependent group in both redundant common-cause data and synergistic parity data. It establishes a limitation on interpretation rather than a headline performance claim.

## Primary conclusion to test

CorEx should recover the correct variable group in both regimes, while a separate signed multivariate diagnostic should distinguish redundancy from synergy. Therefore, Local CorEx factors should generally be described as localized multivariate dependencies unless the data-generating process or an additional measure establishes synergy.

## Information quantities

For a discrete vector \(X=(X_1,\ldots,X_d)\), define total correlation

\[
TC(X)=\sum_{i=1}^{d}H(X_i)-H(X).
\]

Define dual total correlation

\[
DTC(X)=H(X)-\sum_{i=1}^{d}H(X_i\mid X_{-i}),
\]

where \(X_{-i}\) contains all variables except \(X_i\). Define O-information

\[
\Omega(X)=TC(X)-DTC(X).
\]

Positive \(\Omega\) indicates redundancy-dominated dependence, while negative \(\Omega\) indicates synergy-dominated dependence. A value near zero is inconclusive or balanced.

Estimate all entropies with the empirical plug-in estimator and report Miller-Madow bias-corrected values as a sensitivity analysis. Use log base 2 so quantities are in bits.

## Data generators

Generate one true dependent group of size \(d\in\{3,4,5\}\) and append \(p_n=5d\) independent Bernoulli\((1/2)\) nuisance variables.

### Redundant common-cause generator

Sample

\[
Z\sim\operatorname{Bernoulli}(1/2)
\]

and independent noise variables

\[
E_i\sim\operatorname{Bernoulli}(\eta_R).
\]

Set

\[
X_i=Z\oplus E_i,
\qquad i=1,\ldots,d.
\]

### Synergistic parity generator

Sample

\[
X_1,\ldots,X_{d-1}\overset{\mathrm{iid}}\sim
\operatorname{Bernoulli}(1/2),
\]

then set

\[
X_d=X_1\oplus\cdots\oplus X_{d-1}\oplus E,
\qquad
E\sim\operatorname{Bernoulli}(\eta_S).
\]

### Noise matching

For each \(d\), choose \(\eta_R\) from

\[
\{0.01,0.025,0.05,0.10,0.20\}
\]

and choose \(\eta_S\) from the same grid so that the population or high-sample empirical TC values of the two generators are as close as possible. Use a deterministic calibration sample of 1,000,000 observations with seed 4000. Select the pair minimizing absolute TC difference, subject to a difference no greater than 0.05 bits. If no pair satisfies the tolerance, retain the closest pair and flag the mismatch.

Freeze the selected noise pairs before generating evaluation replicates.

## Experimental grid

- Group size \(d\in\{3,4,5\}\).
- Generator in `{redundant, synergistic}`.
- Sample size \(n\in\{500,5000\}\).
- 32 replicates with root seeds 4100 through 4131.
- One dependent group plus \(5d\) nuisance variables.

No clustering is needed. This experiment diagnoses CorEx interpretation, not localization.

## CorEx fitting

Fit Bio CorEx with three requested factors. Rank factors by explained TC. Use the highest-TC factor for the primary recovery analysis, but preserve all fitted factors.

Calculate association scores \(I(Z_j;X_i)\) for all signal and nuisance variables. Define a top-\(d\) support and an 80% cumulative-MI support.

## Metrics

For the true dependent group, report:

- true or high-sample \(TC\), \(DTC\), and \(\Omega\);
- sample plug-in and Miller-Madow estimates;
- highest CorEx factor TC;
- AUPRC for identifying group members;
- top-\(d\) precision, recall, F1, and Jaccard;
- 80% support size and nuisance false-positive proportion;
- fit failures and runtime.

The main comparison is between generator types after approximately matching TC. If CorEx recovery and factor TC are similar while \(\Omega\) has opposite signs, the intended limitation is demonstrated.

## Required output tables

### `redundancy_synergy_calibration`

One row per \(d,\eta_R,\eta_S\) candidate pair with TC values and the selected-pair flag.

### `redundancy_synergy_runs`

One row per replicate, generator, group size, and sample size with all information quantities, recovery metrics, and fit diagnostics.

## Display

This is designed for supplementary material or a compact limitations panel:

1. Plot CorEx factor TC for redundancy and synergy after TC matching.
2. Plot O-information, which should have opposite signs.
3. Plot support-recovery F1 to show that both groups can be found.

Do not claim that O-information is part of Local CorEx unless it is explicitly incorporated into the method. Here it is an external diagnostic.

## Validation tests

1. At zero noise, the redundant generator has positive O-information.
2. At zero noise, the parity generator has negative O-information.
3. Nuisance variables are mutually independent and independent of the true group.
4. Entropy calculations agree with hand-computed distributions for noiseless three-variable examples.
5. TC matching uses calibration data only and is frozen before replicate runs.
6. The highest-TC factor is selected without using support-recovery ground truth.

## Completion criterion

The experiment is complete when TC-matched redundant and synergistic groups have been generated, CorEx recovery has been evaluated without using ground truth for factor selection, and the results support a precise limitation statement about what total correlation can and cannot identify.

