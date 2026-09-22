# Experiment 4: Comprehensive MNIST Factor-Ablation Benchmark

## Scientific question

Across discovery clusters, hidden layers, and highly ranked Local CorEx factors, do cumulative-MI supports identify distributed sets of hidden units whose deletion produces substantial and class-specific changes in classifier behavior?

This is the principal quantitative neural-network experiment. Decoder images and selected examples are explanatory visualizations, not the primary evidence.

## Primary hypotheses

1. Local CorEx supports will produce larger and more class-concentrated prediction changes than matched-size random supports.
2. Local CorEx supports will provide information beyond a univariate ranking of units by correlation with one logit.
3. Factor supports and intervention specificity will differ systematically across H1, H2, and H3.
4. Increasing the cumulative-MI threshold will increase support density and effect magnitude, but may reduce class specificity.
5. The output class most associated with a factor will often be among the output classes most reduced by deleting that factor's support.

## Dataset and data splits

Use the canonical MNIST split of 60,000 training images and 10,000 test images.

Create a fixed stratified split of the original training set:

- 45,000 classifier-training observations.
- 5,000 classifier-validation observations used only for checkpoint selection and early stopping.
- 10,000 discovery observations.
- 10,000 untouched test observations for final intervention evaluation.

Use split seed 5000. Stratification must preserve digit proportions as closely as possible. Save the exact observation indices.

The classifier-training set is used only to fit classifier parameters. The classifier-validation set is used only for checkpoint selection and early stopping. The discovery set is used for PHATE clustering, CorEx fitting, factor ranking, support construction, and qualitative-example selection. The test set is used only for final intervention evaluation.

## Classifier

### Architecture

Flatten each \(28\times28\) image to \(x\in\mathbb R^{784}\). Use:

\[
\begin{aligned}
h_1 &= \operatorname{ReLU}(\operatorname{BN}_1(W_1x+b_1)),
&&\dim(h_1)=500,\\
\widetilde h_1 &= \operatorname{Dropout}_{0.5}(h_1),\\
h_2 &= \operatorname{ReLU}(\operatorname{BN}_2(W_2\widetilde h_1+b_2)),
&&\dim(h_2)=400,\\
\widetilde h_2 &= \operatorname{Dropout}_{0.5}(h_2),\\
h_3 &= \operatorname{ReLU}(\operatorname{BN}_3(W_3\widetilde h_2+b_3)),
&&\dim(h_3)=300,\\
\widetilde h_3 &= \operatorname{Dropout}_{0.5}(h_3),\\
\ell &= W_4\widetilde h_3+b_4,\\
p &= \operatorname{softmax}(\ell).
\end{aligned}
\]

The representations called H1, H2, and H3 are \(h_1,h_2,h_3\), after batch normalization and ReLU and before dropout. At inference time dropout is disabled.

If an existing project checkpoint uses a different operation order, do not silently reinterpret it. Add an architecture-adapter test, record the actual intervention tensors, and either reproduce the published architecture exactly or retrain all seeds using the architecture above.

### Training

Train five independently initialized models with root seeds 5100 through 5104.

Default training configuration:

- Adam optimizer.
- Initial learning rate \(10^{-3}\).
- Batch size 128.
- Cross-entropy loss.
- Maximum 100 epochs.
- Early stopping on classifier-validation loss with patience 10.
- Restore the checkpoint with lowest classifier-validation loss.

No neural-ablation result may influence checkpoint selection. Record test accuracy but do not exclude a trained seed merely because its ablation behavior is atypical. A training failure criterion must be based only on classifier performance and fixed before examining ablations. Use test accuracy below 97% as a failure flag, not an automatic reason for silent retraining.

## Fixed discovery clusters

Create one cluster assignment on the 10,000 discovery images and reuse it for all model seeds and hidden layers:

1. Scale pixels to \([0,1]\) and flatten them.
2. Create a 10-dimensional PHATE embedding with root seed 5200.
3. Apply \(k\)-means with \(k=20\), `n_init=50`, and seed 5201.
4. Save the PHATE coordinates, raw cluster labels, cluster sizes, and label compositions.

The digit labels are not used to construct the partition. They are used only afterward to describe cluster composition and evaluate effects by class.

## CorEx inputs and fits

For every model seed \(s\), discovery cluster \(k\in\{1,\ldots,20\}\), and layer \(l\in\{H1,H2,H3\}\):

1. Run the classifier in evaluation mode on discovery observations in cluster \(k\).
2. Save the hidden representation matrix \(H_{skl}\) and pre-softmax logit matrix \(L_{sk}\in\mathbb R^{n_k\times10}\).
3. Standardize every hidden coordinate and every logit using only observations in that cluster. Coordinates with variance below \(10^{-12}\) are flagged and excluded from CorEx fitting, but their original indices are retained in the metadata.
4. Concatenate the standardized matrices:

\[
V_{skl}=[H_{skl}\mid L_{sk}].
\]

5. Fit Linear CorEx with \(M=10\) requested factors and a deterministic seed derived from model, cluster, and layer.
6. Rank valid factors by explained total correlation \(TC_j\), not by their eventual ablation effect.
7. Retain the top five factors for intervention analysis.

The primary analysis is output-aware because logits are included in \(V\), but it does not use true digit labels. Describe it as **label-free and output-aware**, not fully output-agnostic.

## Required method-input ablation

At cumulative-MI threshold \(q=0.8\), repeat the benchmark with CorEx fit to hidden representations alone:

\[
V^{\text{hidden-only}}_{skl}=H_{skl}.
\]

After fitting, associate each factor with logits by estimating \(I(Z_j;L_c)\) on the discovery observations. This ablation determines whether concatenating logits is essential to obtain class-associated interventions.

Run this hidden-only condition for all clusters, layers, model seeds, and top five factors, but only at \(q=0.8\) and with five matched random controls per support.

## Factor metadata

For each retained factor, store:

- model seed, cluster, and layer;
- CorEx initialization seed and convergence status;
- factor rank and explained \(TC_j\);
- all factor-to-hidden-unit MI scores;
- all factor-to-logit MI scores;
- dominant associated output class

\[
c^*_j=\arg\max_{c\in\{0,\ldots,9\}}I(Z_j;L_c);
\]

- gap between the largest and second-largest factor-to-logit MI;
- number of valid and excluded hidden coordinates.

Do not choose factors using test-set predictions or ablation effects.

## Hidden-unit supports

Construct supports using factor-to-hidden-unit MI only. Logit coordinates must never enter a deletion support.

For each factor, use cumulative-MI thresholds

\[
q\in\{0.60,0.70,0.80,0.90\}.
\]

Use the support definition in `00_README_experiment_program.md`. Store:

- requested cumulative fraction;
- achieved cumulative fraction;
- support size \(s_j(q)\);
- support density \(s_j(q)/d_l\), where \(d_l\in\{500,400,300\}\);
- ordered hidden-unit indices and their MI scores.

The supports must be nested:

\[
S_j(0.60)\subseteq S_j(0.70)
\subseteq S_j(0.80)\subseteq S_j(0.90).
\]

## Exact intervention semantics

### Local CorEx intervention

For layer \(l\), factor \(j\), and support \(S_j(q)\), insert a deterministic mask immediately after the layer's batch normalization and ReLU and before dropout or the next affine layer:

\[
\widetilde h_{l,i}^{\text{abl}}=
\begin{cases}
0,&i\in S_j(q),\\
h_{l,i},&i\notin S_j(q).
\end{cases}
\]

The model must be in evaluation mode. Do not:

- rescale surviving units;
- alter batch-normalization running statistics;
- zero a downstream bias;
- retrain or fine-tune after deletion;
- recompute the factor support on test data.

Activation masking is the canonical implementation because it is independent of weight-matrix orientation. An equivalent cloned-model implementation may zero the corresponding outgoing weight rows or columns only after a unit test demonstrates numerical equivalence to activation masking for every layer.

### Baseline intervention

The baseline uses the same model with an all-ones mask. Baseline predictions should be cached once per model seed.

## Intervention controls

### Matched random supports

For every Local CorEx support of size \(s\), sample 20 sets uniformly without replacement from the valid hidden coordinates in the same layer. Derive seeds from the complete intervention key. Random supports must:

- contain exactly \(s\) unique indices;
- exclude coordinates that were excluded from the CorEx fit;
- be regenerated independently for distinct factor keys;
- remain fixed if the evaluation is resumed.

### Univariate logit-association support

For factor \(j\), use its dominant associated logit \(c^*_j\). Within the same discovery cluster, calculate

\[
u_i=|\operatorname{corr}(H_i,L_{c^*_j})|.
\]

Select the \(s\) hidden units with largest \(u_i\), where \(s=|S_j(q)|\). Break ties by lower hidden-unit index. This produces one size-matched univariate support for each Local CorEx support.

The comparison asks whether multivariate factor membership adds value beyond ranking individual units by association with one logit.

## Evaluation by true digit class

Evaluate every intervention on all 10,000 untouched test images. For true digit \(y\), define the baseline and ablated mean probability vectors

\[
\bar p_y^{(0)}=
\frac{1}{n_y}\sum_{r:Y_r=y}p^{(0)}(x_r),
\qquad
\bar p_y^{(A)}=
\frac{1}{n_y}\sum_{r:Y_r=y}p^{(A)}(x_r).
\]

Define the signed shift

\[
\Delta_{y,c}=\bar p_{y,c}^{(A)}-\bar p_{y,c}^{(0)}.
\]

Store the full \(10\times10\) matrix \(\Delta\) for every intervention.

### Effect magnitude for each input class

\[
T_y=\frac12\sum_{c=0}^{9}|\Delta_{y,c}|.
\]

Also store

\[
T_{\max}=\max_y T_y,
\qquad
T_{\mathrm{weighted}}=\sum_y\frac{n_y}{n}T_y.
\]

### Concentration across affected input classes

If \(\sum_yT_y>0\), define

\[
r_y=\frac{T_y}{\sum_{u=0}^{9}T_u}.
\]

Then calculate

\[
H_{\mathrm{input}}=
-\frac{\sum_y r_y\log r_y}{\log 10},
\qquad
D_{\mathrm{input}}=\max_y r_y.
\]

Low \(H_{\mathrm{input}}\) and high \(D_{\mathrm{input}}\) indicate that the intervention is concentrated on a small number of true digit classes.

### Concentration across changed output probabilities

For each true input class \(y\), define

\[
q_{y,c}=\frac{|\Delta_{y,c}|}{\sum_u|\Delta_{y,u}|}
\]

when the denominator is nonzero, and calculate

\[
H_{\mathrm{output},y}=
-\frac{\sum_c q_{y,c}\log q_{y,c}}{\log 10},
\qquad
D_{\mathrm{output},y}=\max_c q_{y,c}.
\]

For the most affected true class

\[
y^*=\arg\max_yT_y,
\]

store the most decreased and most increased output classes:

\[
c^- = \arg\min_c\Delta_{y^*,c},
\qquad
c^+ = \arg\max_c\Delta_{y^*,c}.
\]

### Predeclared class-specificity criterion

Use the following primary descriptive criterion:

\[
T_{\max}\ge0.10
\quad\text{and}\quad
H_{\mathrm{input}}\le0.50.
\]

Call such an intervention **substantial and class-concentrated**. These cutoffs must not be changed after inspecting the full-run outcomes. Report continuous metrics regardless of threshold classification.

Predeclare a supplementary sensitivity table for

\[
T_{\max}\in\{0.05,0.10,0.15\}
\quad\text{and}\quad
H_{\mathrm{input}}\in\{0.40,0.50,0.60\}.
\]

Do not select the threshold combination that makes Local CorEx look best.

### Directional concordance

Record whether the factor-associated output class is the output class most reduced by ablation:

\[
\operatorname{concordant}=\mathbb 1(c^- = c^*_j).
\]

Also store the rank of \(c^*_j\) when output classes are ordered from most negative to most positive \(\Delta_{y^*,c}\).

### Accuracy metrics

Probability shifts are primary. Also store:

- overall accuracy change;
- per-class recall change;
- full confusion-matrix change;
- mean cross-entropy change.

Do not discard an intervention because accuracy remains unchanged. Softmax shifts can reveal effects that do not cross the decision boundary.

## Comparisons with controls

For each Local CorEx intervention, compare \(T_{\max}\), \(T_{\mathrm{weighted}}\), \(H_{\mathrm{input}}\), and \(D_{\mathrm{input}}\) with its 20 matched random supports.

Calculate a one-sided empirical random-control probability for a statistic \(g\):

\[
\widehat p_g=
\frac{1+\sum_{r=1}^{R}\mathbb 1[g_r\ge g_{\mathrm{LC}}]}{R+1}
\]

when larger values are more extreme. Reverse the inequality for entropy. With \(R=20\), these values are coarse and should be interpreted descriptively.

Compare Local CorEx with the univariate support using paired differences at the same model, cluster, layer, factor, threshold, and support size.

## Aggregation and uncertainty

The complete primary grid contains

\[
5\text{ models}\times20\text{ clusters}\times3\text{ layers}
\times5\text{ factors}\times4\text{ thresholds}=6000
\]

Local CorEx interventions, plus controls.

Summarize first within each model seed. Report model-seed means and 95% intervals across the five seeds. For factor-level distributions, use a hierarchical bootstrap that resamples model seeds, then discovery clusters within seeds. Do not treat all 6000 interventions as independent observations.

Primary summaries:

1. Fraction of Local CorEx interventions that are substantial and class-concentrated, by layer and support threshold.
2. Paired Local CorEx versus random differences in \(T_{\max}\) and \(H_{\mathrm{input}}\).
3. Paired Local CorEx versus univariate-support differences.
4. Support density by layer and threshold.
5. Relationship between factor \(TC_j\), support density, and intervention magnitude.
6. Directional concordance rate by layer.
7. Output-aware versus hidden-only factor comparison at \(q=0.8\).

## Qualitative reconstruction example

Retain one MNIST decoder example only to interpret a factor that is already included in the quantitative benchmark.

Use this predeclared discovery rule:

1. Among clusters with at least 200 observations, choose the cluster maximizing

\[
\min(n_{k,4},n_{k,9}),
\]

where \(n_{k,d}\) is the discovery count of digit \(d\) in cluster \(k\).
2. Within that cluster and each layer, choose the retained factor with the largest

\[
\max\{I(Z_j;L_4),I(Z_j;L_9)\}.
\]
3. Use only model seed 5100 for the visualization.
4. Show the unaltered reconstruction and symmetric factor-directed perturbations, but do not use the visualization to select the factor.

If the original saved cluster-16 partition is available and must be reproduced for continuity with the prior paper, run it as a separately labeled replication. Do not mix its selected factor with the predeclared new benchmark.

## Required output data

### `mnist_models`

One row per model seed with training configuration, selected epoch, losses, accuracies, checkpoint hash, and status.

### `mnist_clusters`

One row per discovery observation with saved split index, PHATE coordinates, cluster label, and true label.

### `mnist_factors`

One row per fitted factor with model, cluster, layer, input condition, factor rank, explained TC, factor-to-logit MI values, dominant logit class, and fit diagnostics.

### `mnist_supports`

One row per factor, threshold, and hidden coordinate selected, including MI score, MI rank, achieved cumulative MI, support size, and support density.

### `mnist_ablation_summary`

One row per intervention method, model, cluster, layer, factor, threshold, and random draw with all aggregate effect, specificity, accuracy, and concordance metrics.

### `mnist_probability_shifts`

One row per intervention, true input class, and output class containing baseline probability, ablated probability, and signed change. Parquet is preferred because this table is large.

## Main-paper displays

Prepare three candidate displays:

1. A layer-by-threshold plot of \(T_{\max}\) and \(H_{\mathrm{input}}\), comparing Local CorEx with matched random supports.
2. A support-density versus intervention-effect plot, colored or faceted by layer.
3. A compact table comparing Local CorEx, random, and univariate supports, including class-concentrated prevalence and directional concordance.

The paper should use at most two quantitative displays plus one small qualitative reconstruction panel. Full threshold sensitivity, hidden-only input ablation, and factor-level distributions can be supplementary.

## Ablation validation tests

1. An all-ones mask reproduces baseline logits within absolute tolerance \(10^{-7}\) on CPU or \(10^{-5}\) on GPU.
2. An empty support produces baseline predictions.
3. Masked activation coordinates are exactly zero at the intervention point.
4. Unmasked coordinates at the intervention point equal their baseline values.
5. Batch-normalization running statistics do not change during evaluation.
6. Repeating an ablation with a different evaluation batch size gives the same predictions within tolerance.
7. Activation masking agrees with outgoing-weight deletion on a small deterministic network where equivalence applies.
8. Support sizes match exactly for Local CorEx, random, and univariate interventions.
9. Random supports contain no repeated indices.
10. Test labels or predictions are never loaded by factor-fitting or support-construction functions.
11. Cumulative-MI supports are nested.
12. Probability shifts sum to zero across output classes within numerical tolerance:

\[
\sum_c\Delta_{y,c}=0.
\]

13. Every saved intervention key uniquely determines its mask.
14. Resume logic reuses previously saved random masks rather than drawing new ones.

## Completion criterion

The experiment is complete when all five model seeds, 20 clusters, three layers, five factors, and four support thresholds have valid Local CorEx results or recorded failures; matched controls have been evaluated; all probability shifts are held-out; and the aggregate figures can be regenerated without loading or rerunning a neural model.
