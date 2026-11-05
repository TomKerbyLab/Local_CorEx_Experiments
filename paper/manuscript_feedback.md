**You are correct, and my earlier framing was too restrictive.** Dissertation-only results can form part of the journal article’s contribution even if they are publicly available in the dissertation. IEEE explicitly supports articles based on thesis or dissertation material, with appropriate citation. Material already published in the conference paper is treated separately: the journal submission should identify and substantively extend that publication. [IEEE Education Society](https://ieee-edusociety.org/publication/tlt/tlt-author-resources?utm_source=chatgpt.com)

For this manuscript, the distinction is:

| Material | How we should treat it |
|---|---|
| Results already in MLSP | Prior published work that the journal article extends. |
| Dissertation results absent from other publications | Eligible content for their first journal publication, with dissertation attribution. |
| New experiments and analyses | Further contributions that strengthen the journal article. |

Therefore, **ImageNet can legitimately be part of the journal extension beyond MLSP**. My recommendation to emphasize its systematic evaluation remains useful scientifically, but we do not need to establish novelty relative to everything in your dissertation. The introduction currently overcorrects by repeatedly positioning the manuscript as an extension beyond both documents.

**The revision is substantially improved, and I would keep its overall direction.** It now presents a more credible account of the mixed results:

- The Gaussian displays expose the conditions behind the averages.
- The ImageNet univariate comparison has the correct interpretation.
- Behavioral effects and interaction type are clearly separated.
- The local/global comparison now includes concentration and support size.
- The qualitative examples restore some intuition for what the method finds.

The next revision should focus on scientific consistency and a few targeted analyses, rather than another broad restructuring.

**1. The nonlinear experiment still needs reconciliation before it can carry headline claims.**

Figure 3a recomputes results from the match tables, while Figure 3b retains hard-coded numbers from a “frozen primary convention.” The appendix acknowledges that the two approaches disagree in their absolute results.

Labeling one analysis frozen does not explain the discrepancy. We need an explicit account of what differs:

- Which estimator and conditions enter each summary?
- Which candidate factors are eligible?
- Are nuisance settings, groups, and regions weighted identically?
- Is the difference caused by matching, filtering, or aggregation?

Figure 3b still lacks an estimator label, despite the stated intention to label every headline result.

I would ask Claude to produce one reproducible calculation for the primary results and a clearly defined sensitivity analysis. Both can remain, but readers should be able to understand why their numbers differ. Agreement in direction alone is insufficient reconciliation.

There is also an important interpretation problem: **the EM capability diagnostic does not explain the Linear CorEx parity result.** The diagnostic concerns an EM-fitted model, while Linear CorEx uses a different fitting procedure.

The observation that PHATE partitions help Linear CorEx while the true-region partition does not is interesting. One possible explanation is that PHATE conditioning creates locally pairwise-visible structure within otherwise pairwise-invisible populations. Other explanations could involve factor selection or scoring. The current evidence does not establish the cause.

I would report that pattern directly and reserve the EM explanation for the estimator actually tested by that diagnostic.

**2. The behavioral metric definitions are clearer, but the figures and prose still contradict them.**

The methods now define approximately:

\[
\Delta_{y,c}
=
\mathbb{E}\!\left[p^{S}_{c}(X)-p_c(X)\mid Y=y\right],
\qquad
T_y=\frac12\sum_c|\Delta_{y,c}|,
\qquad
T_{\max}=\max_y T_y.
\]

Thus, \(T_{\max}\) is the **largest total-variation distance between class-mean prediction vectors across input classes**.

It is neither a maximum over individual output coordinates nor a quantity measured in bits. Consequently:

- Figure 1’s “max per-output-class probability change” needs correction.
- Figure 6’s “bits of class-mean shift” is incorrect.
- The abstract’s closing description of concentration “across output classes” does not describe \(H_{\text{input}}\).

This distinction matters scientifically. Taking the distance between average prediction vectors allows opposing image-level changes to cancel. Small class-mean shifts therefore support a statement about this aggregate measurement, rather than establishing that random deletion is “behaviorally inert.” Flip rates or image-level distances can supply the complementary evidence already collected.

**3. The local/global figure is much better, but support size has been displayed rather than controlled.**

Figure 6 now reveals a more interesting result than the original “potent but blunt” account:

- Global factors have larger effects and greater input-class concentration at H1/H2.
- Local factors retain substantially larger effects at H3.
- Global and local supports differ in size.

That is a useful finding. However, placing support-size curves beside the results does not make the comparison size-controlled. The text should say that size differences are **reported explicitly**, unless an additional adjustment or matched-size analysis was performed.

The highest-value remaining analysis would compare the methods at common support cardinalities or layer fractions, using the same measurements and seed-level aggregation. That could use the existing factors and cached activations.

Two conclusions also remain too strong:

- “Global … carries no H3 structure” overstates a result that still exceeds random deletion by approximately 0.03.
- Unchanged top-five explained TC at \(M=100\) establishes unchanged top-five explained TC. It does not establish unchanged recovered structure or behavioral usefulness, since neural ground-truth groups are unavailable.

I would retain the conclusion about **different layer-dependent behavioral profiles**, with uncertainty shown for the direct comparison.

**4. The proposed explanation for the Gaussian capacity result is incomplete.**

The revision attributes the global decline from \(M=5\) to \(M=10\) to additional partial or degenerate factors matching poorly.

That explanation depends on the scoring rule. If five correct factors remain available, adding poor candidate factors cannot reduce optimal one-to-one coverage of a fixed set of true groups: the matching can ignore the extras.

A decline could arise because:

- Refitting changes or loses the previously good factors.
- Ranking or truncation excludes useful factors.
- The reported score penalizes unmatched predicted factors.
- The matching target or aggregation changes.

Claude should identify which mechanism actually applies. The current account describes a plausible association, but does not yet demonstrate the scoring mechanism.

The heatmaps also show that gains do **not** uniformly grow with sample size or mean separation. The strongest general pattern is the difference between covariance regimes. The introduction should describe sample-size and separation effects conditionally.

Finally, call the oracle a **true-region reference partition**, rather than a general upper bound. The nonlinear results themselves demonstrate that an estimated partition can outperform it for a particular estimator.

**5. The restored examples need their own experimental provenance.**

Restoring examples was helpful, but Figure 7 comes from an earlier analysis with a different setup. The dissertation describes a 200-dimensional MNIST hidden representation and partitioning of the test data. Local_CorEx_Dissertation.pdf

The new benchmark uses 500/400/300 units and a discovery/evaluation split. The older examples therefore should not inherit the new section’s blanket statement that all selection occurred on data separate from evaluation.

I would label them explicitly as **illustrative examples from the earlier analysis**, specifying the network and selection procedure. Reused conference panels should retain conference attribution as well.

There are also two concrete caption errors:

- Figure 7b describes the arrangement incorrectly. H1 is the left column and H2 the right column; factors occupy the rows.
- Figure 7d calls the −0.91 change a *logit* drop while discussing conservation of probability mass. Logits and probabilities need distinct labels; probability conservation applies after normalization.

Likewise, the newly explicit claim that the classifier never trained on the 10,000 discovery images should be checked against the actual checkpoint’s training indices. The prose is now unambiguous, provided it describes the run that produced the results.

**6. The remaining analysis flags should be resolved before polishing the manuscript.**

The appendix identifies a control summarizer that averages across thresholds instead of matching \(q=0.80\). That is an analysis correction, even if it changes only a few rows.

I would fix the pairing keys, regenerate the affected summaries, and document whether the conclusions change. This should not require retraining networks or refitting CorEx.

Similarly, replacing the impossible owl statement with “freed mass returns diffusely” is an improvement, but the detailed redistribution claim still needs verification from the saved probability vectors. Conservation guarantees that mass returns somewhere; it does not establish its distribution.

The plotting code also uses several uncertainty conventions:

- Figure 3a uses \(1.96\,SE\).
- Figure 5 uses \(2\,SE\).
- The methods describe t-based confidence intervals.

Those should be reconciled and labeled, including whether uncertainty is calculated across network seeds, synthetic replicates, or another unit.

**7. A few conceptual statements still need correction.**

The background currently attaches the KL-divergence definition to explained TC. The definitions should be separated:

\[
TC(X)=D_{\mathrm{KL}}\!\left(p_X\,\middle\|\,\prod_i p_{X_i}\right),
\qquad
TC(X;Z)=TC(X)-TC(X\mid Z).
\]

Figure 4’s “fails loudly” description is also misleading. An accurate O-information calculation on an incorrectly recovered support can return a value near zero without warning the analyst that important variables were missed. That is precisely the useful limitation this experiment demonstrates. I would describe it as **the effect of support-recovery error on subsequent characterization**.

Finally, “piecewise linear” applies to the local Linear CorEx configuration. It does not establish general robustness to nonlinear dependence, and it should not describe every CorEx estimator.

**The story I would now carry forward**

The revision supports a coherent account: informative partitioning improves recovery of heterogeneous dependence groups; nonlinear success depends on both partitioning and the estimator; and neural applications reveal consequential groups whose advantages over simpler selection vary across layers and measurements.

That accommodates the strong Gaussian results, the H3 finding, and the competitive ImageNet univariate baseline without forcing a universal superiority claim.

I would keep E3 compact, retain one clearly contextualized worked example, and resolve the analysis issues above before drafting the final abstract. The most useful additional computation is the direct comparison at common support sizes. A simple grouping baseline would be the next priority if we decide further evidence is needed.