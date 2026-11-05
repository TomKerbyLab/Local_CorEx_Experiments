**Yes. I think factor heterogeneity is the right next question.** The pooled results answer, “What happens when we delete supports from highly ranked factors?” Your question is more specific: “Which factors have class-related, cluster-related, or broadly shared influence, and can we recognize those differences before intervening?”

Those deserve separate analyses. High TC does not imply class specificity, so averaging the top five factors together can obscure meaningful differences.

The screenshots also suggest that cancellation is relatively small overall, given the reported \(T/U\) ratios. The larger remaining concern is **pooling different factors and evaluating them on cohorts that are too coarse.**

**I would push back on the current interpretation of “in-cluster” effects.**

The analysis shown measures effects on the discovery cluster’s **dominant digit throughout the test set**. That is a class view. Even a cluster containing 99% zeros could represent only one particular style of zero. Averaging over every zero can hide an effect confined to that style.

The proposed **0.86 “cluster-proportional null” is also unjustified**. Discovery-cluster purity describes which digits occur inside that cluster. It does not imply that 86% of an intervention’s behavioral effect should fall on the dominant digit. Those are different quantities.

Consequently, I would hold off on statements such as “the factors spare their discovery cluster” or “the majority actively avoid their discovery cluster.” The reported mean and median differences suggest heterogeneity worth investigating, but they do not establish those conclusions.

I would approach the next analysis as follows.

1. **Measure effects on actual held-out cluster members.**

   Assign test images to the existing discovery clusters using a fixed rule derived from discovery data. PHATE provides an interpolation path for new observations, although the assignment behavior should be checked before relying on it. [PHATE 1.0.11 documentation](https://phate.readthedocs.io/en/stable/_modules/phate/phate.html?utm_source=chatgpt.com)

   For each factor, compare effects on:

   - Images assigned to its discovery cluster.
   - Images of the same digit assigned to other clusters.
   - Images of other digits.

   That middle comparison is especially valuable. It distinguishes a factor affecting a particular handwriting style from one affecting the digit generally. For mixed clusters, make the comparison within digit and then combine using fixed weights.

   Use **average effect per image** in each cohort, with cohort sizes reported. Total effect mass would otherwise favor larger cohorts.

2. **Give each factor a behavioral profile before aggregating.**

   For factor \(f\), compute an image’s intervention magnitude:

   \[
   a_f(x)=\tfrac12\left\|p_f^{A}(x)-p^0(x)\right\|_1.
   \]

   Then summarize that quantity separately across the ten true classes and the twenty evaluation clusters. Keep the signed probability shifts and recall changes alongside it.

   This separates several questions that are currently easy to conflate:

   - How strong is the effect?
   - Which input images experience it?
   - Which output probabilities increase or decrease?
   - Does the effect concentrate near the discovery region?

   For example, a factor associated with the “7” logit could change the probability of seven across many kinds of images. Another factor could primarily affect images of sevens by increasing a competing class’s probability. Both are class-related, through different relationships.

3. **Treat factor association and intervention behavior as separate axes.**

   I would retain profiles such as these:

   | Discovery association | Held-out intervention behavior | Interpretation supported |
   |---|---|---|
   | Strong class association | Strong, concentrated effect aligned with that class | Candidate for targeted intervention |
   | Strong class association | Strong, broad effect | Class association with shared behavioral influence |
   | Strong class association | Small effect | Associated factor with limited influence under this deletion probe |
   | Weak class association | Strong, concentrated effect | Behavioral specificity poorly captured by the association score |
   | Weak class association | Small effect | Limited behavioral influence under this probe |

   Cluster locality would be an additional annotation on each profile. A factor could have broad class effects yet unusually strong effects within its discovery cluster.

   Also, **every factor has an argmax logit association**. Calling that logit its “associated class” can make a weak or ambiguous association sound substantive. Report absolute association strength, the gap to competing classes, and association concentration.

   Within a nearly pure discovery cluster, CorEx is learning dependencies among variation *within* a digit. Those dependencies may concern style or confidence and need not distinguish that digit from other digits.

4. **Define useful strata independently of the effects being evaluated.**

   Your proposed distinction becomes particularly informative if factors with stronger discovery-side class association subsequently produce more selective held-out effects.

   Keep the full-grid result, then report conditional results for those independently defined strata, including their prevalence. Apply the same cohorts and target definitions to their matched controls.

   Grouping factors by observed test effects is still useful for describing heterogeneity. However, selecting the most selective factors and then reporting their selectivity does not establish that the method can identify them prospectively. Independent selection and evaluation are needed for that stronger claim. [PubMed](https://pubmed.ncbi.nlm.nih.gov/19396166/?utm_source=chatgpt.com)

   Since the current outcomes have already been examined, the new stratified analysis should be recorded as exploratory.

**The most useful next display would be two aligned heatmaps:** one row per factor, with ten digit columns in one panel and twenty evaluation-cluster columns in the other. Add discovery-cluster identity, association strength, and support density beside the rows. Show both absolute effects and differences from matched random controls. Summarize the resulting patterns across the five model seeds.

That would reveal whether a modest average comes from mostly quiet factors, a mixture of selective and broad factors, or factors whose influence concentrates on different regions.

The \(q=0.95/0.99\) extension can complement this, but increasing \(q\) mainly changes intervention size. At \(q=0.99\), inspect support density and overlap between factors: distinct factors may produce nearly identical large masks. The cluster and factor profiles are the analyses that directly answer your concern about what the current averages are hiding.