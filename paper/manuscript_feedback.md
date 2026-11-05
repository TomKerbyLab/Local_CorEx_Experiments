**I would keep the experimental program, but substantially revise the organizing story before turning this into an Overleaf manuscript.** The new work offers a stronger journal extension than the current draft conveys. Its main weakness is that it reads like an experiment audit, with several conclusions stronger than the comparisons actually shown.

The central claim I would build around is:

**Local CorEx discovers groups of dependent variables whose structure varies across contexts. Controlled experiments establish when localization improves recovery, and neural interventions assess the behavioral relevance of the resulting groups.**

That gives us a coherent progression from identifying structure to understanding its usefulness, while leaving room for the nonlinear limitations and mixed baseline results.

The progression from the earlier work matters. In Learning_Local_Higher-Order_Interactions_with_Total_Correlation.pdf, the main evidence was Gaussian recovery and selected MNIST examples. In Local_CorEx_Dissertation.pdf, you already expanded the applications to tabular data, pixel groups, dropout, and ResNet/ImageNet. Consequently, the journal contribution should emphasize **controlled comparisons, systematic evaluation, and stronger experimental separation**. ImageNet’s inclusion alone is not the main advance over the dissertation.

I am treating the numerical results as reported here. The underlying result tables referenced by the draft were not attached, so this is a review of the supplied manuscript, figures, and plotting code.

**The role I would give each experiment**

| Component | Scientific question | Role in the paper |
|---|---|---|
| Gaussian recovery | When does partitioning improve recovery of context-dependent dependence groups? | Establish the method’s operating conditions through heterogeneity, partition quality, sample size, and factor-budget controls. |
| Nonlinear recovery | How far do those benefits extend beyond covariance structure? | Test the boundary of the framework and distinguish localization benefits from estimator limitations. |
| Matched-TC experiment | What information about dependence structure survives fitting and support selection? | Provide a concise interpretive boundary, with the larger diagnostic study in the supplement. |
| MNIST interventions | Do discovered groups have reproducible behavioral effects, and what does localization add? | Supply the main systematic neural experiment, including direct local/global and univariate comparisons. |
| ImageNet interventions | Does the analysis remain useful on a larger pretrained representation? | Demonstrate transfer of the analysis protocol, supported by a survey and an interpretable example. |

This retains all five components without making them five equally weighted contributions. Gaussian recovery and neural analysis should carry most of the narrative.

**1. The local-versus-global neural comparison needs a stronger presentation.**

Including global CorEx was the right decision. However, Figure 5d does not establish the draft’s description of global factors as “potent but blunt.”

That panel reports effect magnitude relative to random controls. Broad effects and class-concentrated effects could both produce large magnitude. Likewise, a small H3 effect does not by itself demonstrate poor concentration.

The central comparison should put local and global supports on the same measurements:

- Effect magnitude.
- Concentration across the relevant classes.
- Localization across input cohorts.
- Support cardinality or fraction of the layer removed.

**The same cumulative-MI threshold does not imply the same intervention size.** A global factor and a local factor can require very different numbers of units to reach \(q=0.9\). Comparing each against its own size-matched random control is useful, but it does not fully resolve a direct local/global comparison.

I would show local and global results together, preferably across support sizes or with support size explicitly accounted for. Global supports can be evaluated on the same regional cohorts, with any region or class association selected using discovery data.

The factor-budget language also needs clarification. Distinguish factors fitted, retained, and evaluated. A global fit with \(M=100\), followed by examination of only its top five factors, cannot establish that the other factors contain no additional useful structure. Unchanged top-five TC supports a much narrower conclusion.

Until those comparisons are clear, I would describe the current result as **evidence that local and global fitting identify groups with different layer-dependent behavioral profiles**.

**2. The univariate results should shape the story more directly.**

The ImageNet text contains a consequential sign inconsistency.

Figure 6 defines its contrasts as Local CorEx minus univariate. It reports approximately:

| Measurement | Local CorEx minus univariate |
|---|---:|
| Effect magnitude | −0.019 |
| Localization contrast | −0.017 |

Under the stated interpretation that larger localization contrast means greater localization, these values favor the univariate control on both measurements. The prose instead says that Local CorEx trades magnitude for better localization. **That tradeoff is not supported by the displayed signs.** Either the numbers, labels, or interpretation need correction.

The MNIST abstract also overgeneralizes when it says effects far exceed both random and univariate controls. The reported advantage over univariate selection is concentrated at H3; H1 and H2 are competitive or favor the control on the displayed magnitude metric.

These findings can still support a useful paper. They suggest that:

- Local CorEx selects behaviorally consequential groups relative to arbitrary groups of the same size.
- Its advantage over a prediction-guided single-unit selector depends on the representation and measurement.
- Factor organization and behavioral impact provide complementary descriptions of a representation.

The random comparison establishes consequential selection. The univariate comparison asks whether the multivariate factor model adds something beyond simpler selection. Those are different questions, and the manuscript should let each result answer its own question.

If we add one further neural baseline, I would prioritize **a simple alternative grouping method using the same discovery data and intervention budget** over a costly Bio CorEx expansion.

**3. E3 needs a different scientific argument.**

I agree with retaining this experiment, but its current framing mixes three distinct issues:

1. Population TC does not uniquely determine redundancy versus synergy dominance.
2. Fitted explained TC depends on the estimator and optimization procedure.
3. Statistical dependence does not determine a group’s behavioral role in a trained network.

The matched-population-TC construction addresses the first issue. The large difference in fitted TC addresses the second. Interventions address the third.

The draft sometimes treats these as one chain of reasoning: TC is type-blind, therefore intervention reveals what TC misses. That is too strong. **Deleting a group does not establish whether its statistical dependence is redundancy-dominated or synergy-dominated.** O-information addresses that statistical distinction; intervention measures a different property. The original O-information work makes that characterization distinction explicit. [arxiv.org](https://arxiv.org/abs/1902.11239?utm_source=chatgpt.com)

There is also a tension between describing the fitted summaries as “indistinguishable” and plotting sharply different fitted TC values. The defensible conclusion is that those values do not provide a reliable, general interpretation of interaction type: they also reflect recovery and fitting failures.

I would keep a short main-text section establishing these boundaries, then move most of the optimization and support-characterization diagnostics to the supplement. E3 should sharpen the interpretation of the paper without becoming its central novelty claim.

**4. The synthetic experiments need to expose the conditions behind the averages.**

The Gaussian design is a strong foundation. Oracle, estimated, and random partitions, together with homogeneous controls and equal factor budgets, address the right questions.

However, Figure 2a compresses the experiment too aggressively. An average across 27 conditions conceals the relationship that motivates Local CorEx:

**How does the benefit change as dependence structures become less compatible and the partition becomes more informative?**

I would make the main display show local-minus-global recovery by covariance regime, separation, and sample size. Partition recovery quality could provide an additional explanatory axis. The overall mean can remain as a compact summary.

Two setup details deserve particular attention:

- **The Gaussian experiment validates recovery of multivariate dependence groups.** Its structure is specified through covariance, so it does not establish recovery of interactions irreducible to pairwise relationships.
- The decline in global recovery when increasing the factor budget in the nondisjoint condition deserves a focused check of matching, duplicate factors, convergence, and aggregation. It may be a real estimator behavior, but it should not carry a broad capacity argument without that explanation.

For nonlinear recovery, the distinction between oracle and estimated partitions is essential. The headline currently emphasizes oracle-versus-global gains. That establishes that localization can help when the relevant regions are known. The practical method needs its estimated-partition results visible alongside that result.

The nonlinear section should also identify the CorEx estimator behind every headline number. At present, positive parity results sit close to a negative Linear CorEx result, making the source of the successful recovery unnecessarily difficult to follow.

Finally, E1 uses one-to-one matching while E2 emphasizes best-factor matching. Both can be informative, but they answer different recovery questions. Preserve the chosen primary analysis, explain the distinction, and show whether the conclusion survives a comparable matching convention. A chance reference should use the same candidate-factor and matching procedure.

The atom-seeded parity diagnostic is useful evidence of representational capability under that configuration. It should remain a diagnostic rather than implying that the practical fitting problem has been solved.

**5. The neural setup needs a precise, shared methods section.**

The current methods are too compressed for the weight placed on the neural results.

I would establish the following before presenting either neural experiment:

| Design element | Required clarification |
|---|---|
| Discovery and evaluation | Exact image pools, counts, and overlap, including whether “held out” means held out from network training or factor discovery. |
| Partition construction | What representation is clustered, whether partitions are shared across layers, and how evaluation images are assigned. |
| Factor inputs | Activation-only versus activation-plus-logit conditions, including their preprocessing. |
| Support construction | Whether the cumulative-MI denominator includes only activation coordinates after logits are removed. |
| Univariate control | Which output association is used, how its target is selected, and whether it receives the same regional information. |
| Behavioral measurements | Explicit definitions of magnitude, concentration, and localization, including aggregation order. |
| Uncertainty | Network seeds for MNIST; cluster-based uncertainty conditional on the analyzed ImageNet network and split. |

The MNIST wording particularly needs reconciliation: it describes 10,000 held-out discovery images and evaluation on all 10,000 test images. That may represent two separate pools, but the manuscript does not currently establish that.

The metric definitions also need to distinguish **input classes, input regions, and output probability coordinates**. This distinction could explain another apparent inconsistency: the owl example says its probability drops by roughly 0.86 while all other per-class shifts are at most \(10^{-6}\). If those are output coordinates on the same cohort, probability conservation requires the lost mass to appear elsewhere. If “other classes” refers to other input cohorts, say so explicitly.

For ImageNet, the final linear head gives an especially clean interpretation:

\[
\Delta z=-W_{[:,S]}h_S.
\]

That supports precise statements about the contribution of selected feature groups to predictions. It also means the deletion result alone cannot demonstrate synergistic downstream computation among those features.

**What I would retain from the dissertation**

The dissertation provides something the new draft largely loses: an intuitive view of what the factors capture.

I would restore one compact example combining actual inputs, a factor’s feature associations, and its intervention effects across cohorts or classes. A well-chosen example would make the systematic results much easier to understand. Decoder or DeepDream images could support that example, with their interpretation kept appropriately modest.

I would leave the communities application and dropout study outside the main manuscript for now. They introduce additional scientific questions, whereas the recovery-to-neural-analysis progression is already substantial.

The methods section should also make one distinction visually obvious: **PHATE partitions observations; CorEx groups variables within those partitions.** CorEx is then fitted to the original variables, rather than to the plotting coordinates.

**Before moving to Overleaf**

I would make three substantive revisions first:

1. Rewrite the contribution statement around conditional recovery and systematic behavioral analysis.
2. Resolve the direct local/global comparison and the univariate sign inconsistencies.
3. Write the shared experimental definitions so that every result has an unambiguous target, comparison, and evaluation population.

The remaining editorial work can follow naturally: remove branch histories and internal decision logs from the main text, replace the qualifications panel in Figure 3 with actual diagnostic evidence, and reconcile figure/text discrepancies such as the Bio global recovery value.

A working title that fits this direction is *Local CorEx: Learning Context-Dependent Structure in Data and Neural Representations*. It preserves the method’s breadth and gives the experiments room to establish exactly which kinds of structure and usefulness the evidence supports.