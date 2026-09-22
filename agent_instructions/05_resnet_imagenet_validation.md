# Experiment 5: ResNet-18 and ImageNet Validation

## Scientific question

Do Local CorEx factors in the final pooled representation of a pretrained ResNet-18 identify feature groups whose removal produces semantically concentrated changes in ImageNet predictions?

This experiment has two roles:

1. provide a systematic summary across many discovery clusters and factors;
2. retain the African-grey example as an interpretable case study.

The MNIST benchmark remains the primary quantitative neural experiment. ImageNet is a larger-scale validation, not a substitute for replicated MNIST results.

## Model and representation

Use the torchvision ResNet-18 checkpoint with `IMAGENET1K_V1` weights unless the existing project checkpoint is different. Record:

- exact framework and package version;
- weight enum or checkpoint hash;
- preprocessing transform;
- ImageNet class-index mapping;
- model architecture hash if available.

Run the model in evaluation mode. For input image \(x\), extract:

- the 512-dimensional average-pooled representation \(h(x)\) immediately before the final fully connected layer;
- the 1000-dimensional pre-softmax logits

\[
\ell(x)=Wh(x)+b;
\]

- the softmax probabilities \(p(x)\).

Cache these quantities so intervention evaluation does not repeatedly run the convolutional backbone.

## Data split

Use the 50,000-image ImageNet validation set. For every true class, sort images by canonical relative path and assign:

- the first 25 images to the discovery half;
- the remaining 25 images to the evaluation half.

This gives 25,000 discovery and 25,000 evaluation images with exactly 25 images per class in each half. Save image paths and split assignments.

If any class does not have exactly 50 valid images after decoding, flag the dataset integrity problem and stop. Do not silently rebalance using another class.

## Clustering

On discovery representations only:

1. Standardize the 512 pooled coordinates.
2. Fit a 10-dimensional PHATE embedding with seed 7000.
3. Fit \(k\)-means with \(k=100\), `n_init=50`, and seed 7001.
4. Save the embedding, cluster assignments, centroids, cluster sizes, and true-class compositions.

Transform evaluation representations using the fitted PHATE mapping and assign them to the nearest saved \(k\)-means centroid. If the PHATE implementation does not support a stable out-of-sample transform, use a fixed supervised-free nearest-neighbor transfer:

\[
\widehat C(x)=C(x_{i^*}),
\qquad
i^*=\arg\min_i\|h(x)-h(x_i^{\mathrm{disc}})\|_2,
\]

where distances are computed in standardized 512-dimensional representation space. Record which assignment method is used and do not mix methods within a run.

## CorEx fitting

For every discovery cluster \(k\):

1. Standardize pooled coordinates and logits using only discovery images in that cluster.
2. Concatenate

\[
V_k=[H_k\mid L_k]\in\mathbb R^{n_k\times1512}.
\]

3. Fit Linear CorEx with \(M=10\) requested factors and a seed derived from cluster \(k\).
4. Retain the top three valid factors by explained TC for the systematic benchmark.
5. Store factor-to-pooled-unit and factor-to-logit MI separately.

For retained factor \(j\), define the dominant associated output class

\[
c^*_j=\arg\max_{c\in\{1,\ldots,1000\}}I(Z_j;L_c).
\]

The primary systematic support uses \(q=0.80\) cumulative MI among the 512 pooled units. Do not include logit coordinates in the support.

Run \(q\in\{0.60,0.70,0.90\}\) only for the selected qualitative cluster and as a supplementary sensitivity analysis.

## Exact pooled-feature ablation

For support \(S_j\), define

\[
h_i^{(A)}(x)=
\begin{cases}
0,&i\in S_j,\\
h_i(x),&i\notin S_j.
\end{cases}
\]

Then compute ablated logits directly from the cached representation:

\[
\ell^{(A)}(x)=Wh^{(A)}(x)+b.
\]

Equivalently,

\[
\ell^{(A)}(x)=
\ell(x)-W_{:,S_j}h_{S_j}(x).
\]

Use the equation above as an efficient sparse update and unit test it against a full final-layer forward pass.

Do not:

- alter the final-layer bias;
- re-estimate batch-normalization statistics;
- rerun or fine-tune the convolutional backbone;
- select supports from evaluation images;
- use DeepDream output quality to select a factor.

## Controls

### Matched random pooled features

For every retained support of size \(s\), generate five supports sampled uniformly without replacement from the 512 pooled coordinates. Use deterministic seeds derived from the complete factor key.

### Univariate logit-association support

Within the discovery cluster, calculate

\[
u_i=|\operatorname{corr}(H_i,L_{c^*_j})|
\]

and select the top \(s\) pooled coordinates. This baseline is matched on cluster, dominant logit, representation layer, and support size.

## Computationally bounded evaluation set

For a factor learned in cluster \(k\), evaluate on:

1. every evaluation image assigned to cluster \(k\);
2. an equally sized off-cluster control sample drawn without replacement from evaluation images assigned to other clusters, stratified as evenly as possible across their true classes.

If cluster \(k\) has fewer than 50 evaluation images, include all its evaluation images but flag the factor as low-sample. If it has more than 1000, use a deterministic stratified sample of 1000 for the core evaluation and preserve an option to evaluate the full cluster later.

The same evaluation observations must be used for the Local CorEx, random, and univariate supports attached to a factor.

## Metrics

### Image-level changes

For each image \(x_r\), store or summarize:

\[
TV_r=\frac12\sum_{c=1}^{1000}
|p_c^{(A)}(x_r)-p_c^{(0)}(x_r)|.
\]

Also record:

- change in probability of the true class;
- change in probability of \(c^*_j\);
- change in top-1 correctness;
- baseline and ablated top-5 class sets;
- most decreased and most increased output classes.

### Classwise changes

For every true class \(y\) with at least five observations in the factor's evaluation set, calculate the mean probability shift

\[
\Delta_{y,c}=
\bar p^{(A)}_{y,c}-\bar p^{(0)}_{y,c}
\]

and classwise total variation

\[
T_y=\frac12\sum_{c=1}^{1000}|\Delta_{y,c}|.
\]

Normalize across represented input classes:

\[
r_y=\frac{T_y}{\sum_uT_u},
\qquad
H_{\mathrm{input}}=
-\frac{\sum_y r_y\log r_y}{\log K},
\]

where \(K\) is the number of represented true classes with at least five evaluation observations. Store the dominant affected-input fraction \(\max_y r_y\).

Because \(K\) varies by cluster, always store \(K\) and do not compare raw entropy values that use different normalizations.

### Source-versus-off-cluster localization

Calculate mean image-level TV separately for source-cluster and off-cluster images:

\[
\overline{TV}_{\mathrm{source}},
\qquad
\overline{TV}_{\mathrm{off}}.
\]

Define the bounded localization contrast

\[
L_j=
\frac{\overline{TV}_{\mathrm{source}}-
\overline{TV}_{\mathrm{off}}}
{\overline{TV}_{\mathrm{source}}+
\overline{TV}_{\mathrm{off}}+10^{-12}}.
\]

Positive \(L_j\) indicates a larger effect near the factor's discovery region. This is secondary to classwise semantic concentration.

### Semantic output summary

For storage efficiency, preserve the full 1000-class probability-shift vector in compressed arrays but write the 20 most decreased and 20 most increased classes to a tidy table. Include ImageNet class names and WordNet identifiers.

For the most affected true class, calculate output redistribution entropy normalized by \(\log(1000)\) and the dominant output-shift fraction, using the definitions from the MNIST experiment.

## Systematic summaries

Across all 100 clusters and their top three factors, report:

1. Support-size and support-density distributions.
2. Local CorEx versus matched-random differences in source-cluster TV.
3. Local CorEx versus univariate-support differences.
4. Distribution of localization contrast \(L_j\).
5. Relationship between factor TC and intervention magnitude.
6. Fraction of factors for which \(c^*_j\) is among the five most decreased output classes.
7. Fit-failure and low-sample rates.

Use discovery cluster as the resampling unit for bootstrap intervals. Random-support draws are controls, not independent scientific replicates.

## African-grey case study

Select the case-study cluster using a deterministic rule on discovery data:

1. Identify the ImageNet class index for `African grey` from the saved class mapping.
2. Among clusters with at least 100 discovery images, choose the cluster with the largest proportion of African-grey images.
3. Break ties by larger African-grey count, then lower raw cluster identifier.
4. Within that cluster, select the two retained factors with largest MI with the African-grey logit.

If the saved original clustering identifies this as cluster 58, retain that identifier in the narrative. If a clean rerun produces another numeric identifier, use the deterministic selection above and report that cluster IDs are run-specific.

For each selected factor:

- report factor TC and factor-to-African-grey-logit MI;
- report 60%, 70%, 80%, and 90% support sizes;
- evaluate all four supports with matched controls;
- show the 20 largest output-probability decreases and increases;
- report results for the source cluster and the most affected secondary cluster.

### DeepDream visualization

Use the factor-to-pooled-unit MI vector as a fixed nonnegative objective weight. For input image \(x\), optimize

\[
J(x)=\sum_{i=1}^{512}\widetilde a_{ji}h_i(x),
\qquad
\widetilde a_{ji}=
\frac{a_{ji}}{\sum_r a_{jr}}.
\]

Use the same number of steps, step size, regularization, jitter, and image preprocessing for both selected factors and for all displayed starting images. Save the complete visualization configuration. Do not rank factors or supports using the appearance of generated images.

## Required output data

### `imagenet_split`

Image path, true class, discovery/evaluation assignment, and integrity status.

### `imagenet_clusters`

Discovery and evaluation cluster assignments, assignment method, PHATE coordinates when available, and cluster composition.

### `imagenet_factors`

Cluster, factor rank, factor TC, factor-to-unit MI, factor-to-logit MI, dominant logit, and fit diagnostics.

### `imagenet_supports`

Factor, cumulative-MI threshold, selected pooled index, MI score, support size, and density.

### `imagenet_ablation_summary`

Factor and control keys, source/off-cluster TV, localization contrast, accuracy changes, entropy metrics, concordance metrics, sample counts, and runtime.

### `imagenet_top_probability_shifts`

Factor and control keys, true-class group, changed output class, class name, WordNet ID, signed probability shift, and shift rank.

## Validation tests

1. Cached baseline logits equal \(Wh+b\) within tolerance.
2. Sparse-update ablated logits equal a full final-layer forward pass.
3. An empty support reproduces baseline predictions.
4. Random and univariate supports exactly match Local CorEx support size.
5. Discovery images never enter intervention evaluation.
6. Cluster assignment for evaluation images is deterministic.
7. Every class has the intended 25/25 split.
8. Softmax changes sum to zero across output classes within tolerance.
9. The African-grey cluster and factors are selected without evaluation-set intervention results.
10. DeepDream code cannot write back into quantitative factor-ranking outputs.

## Main-paper display

Use at most:

1. one compact systematic comparison of Local CorEx and matched controls;
2. one African-grey factor visualization and ablation panel;
3. one secondary-cluster probability-shift panel if space permits.

The remaining distributions, support thresholds, and DeepDream examples belong in supplementary material.

## Completion criterion

The experiment is complete when all 100 discovery clusters have a valid fit or recorded failure, top-three-factor 80% supports and matched controls have been evaluated on held-out images, the African-grey selection rule has been applied without outcome-based selection, and all figures can be regenerated from cached features and saved result tables.

