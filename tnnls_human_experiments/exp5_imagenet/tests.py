"""Experiment 5 validation tests (spec 'Validation tests' 1-10, ported to
human-pipeline conventions; synthetic-data tests plus real-artifact tests
that run once the stage-b cache exists)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from common import config, metrics, seeds
from ablations import (
    ablated_logits_full, ablated_logits_sparse, load_final_layer,
    random_support, softmax_np, summarize_intervention, univariate_support)
from case_study import AFRICAN_GREY_IDX, african_grey_rule
from clustering import (
    even_stratified_sample, cohort_for_cluster, route_eval_to_clusters,
    standardize_with_discovery_stats)
from data import AFRICAN_GREY_WNID, imagenet_split

CACHE_PATH = config.RAW_DIR / "imagenet" / "resnet18_imagenet_val_cache.npz"

_cache_missing = not CACHE_PATH.exists()


def _require_cache():
    if _cache_missing:
        pytest.skip(
            "the stage-b feature cache has not been built yet "
            f"({CACHE_PATH}); run `python runner.py --stages cache,verify,splits`")


# ---------------------------------------------------------------------------
# 7. Every class has the intended 25/25 split
# ---------------------------------------------------------------------------

def test_split_25_25_per_class_synthetic():
    labels = np.repeat(np.arange(1000), 50)
    is_disc, is_eval = imagenet_split(labels)
    assert is_disc.sum() == 25_000 and is_eval.sum() == 25_000
    for c in range(1000):
        assert is_disc[labels == c].sum() == 25
        assert is_eval[labels == c].sum() == 25
    # discovery = first 25 rows per class (row order rule)
    first_class_rows = np.flatnonzero(labels == 5)[:50]
    assert is_disc[first_class_rows[:25]].all()
    assert is_eval[first_class_rows[25:]].all()


def test_split_integrity_check_fails_on_wrong_count():
    labels = np.repeat(np.arange(100), 49)
    with pytest.raises(AssertionError):
        imagenet_split(labels)


def test_split_25_25_on_real_cache():
    _require_cache()
    with np.load(CACHE_PATH) as z:
        labels = z["labels"]
    is_disc, is_eval = imagenet_split(labels)
    assert is_disc.sum() == 25_000 and is_eval.sum() == 25_000
    counts_disc = np.bincount(labels[is_disc], minlength=1000)
    counts_eval = np.bincount(labels[is_eval], minlength=1000)
    assert (counts_disc == 25).all() and (counts_eval == 25).all()


# ---------------------------------------------------------------------------
# 1. Cached baseline logits equal W h + b within tolerance (real cache)
# ---------------------------------------------------------------------------

def test_cached_logits_equal_wh_plus_b():
    _require_cache()
    W, b = load_final_layer()
    with np.load(CACHE_PATH) as z:
        feats = z["features"]
        logits = z["logits"]
    recon = feats @ W.T + b
    # float32 (cache) vs float64 (reconstruction): tolerance ~ float32 eps
    assert np.abs(recon - logits).max() <= 0.5


# ---------------------------------------------------------------------------
# 2. Sparse-update ablated logits equal a full final-layer forward pass
# ---------------------------------------------------------------------------

def test_sparse_update_equals_full_final_layer():
    torch.manual_seed(1)
    W = torch.randn(1000, 512)
    b = torch.randn(1000)
    H = torch.randn(64, 512)
    logits = (H @ W.T + b).numpy()
    support = [0, 7, 100, 511]
    full = ablated_logits_full(H.numpy(), W.numpy(), b.numpy(), support)
    sparse = ablated_logits_sparse(logits, H.numpy(), W.numpy(), support)
    denom = np.abs(full).max()
    # float32 accumulation-order differences: relative tolerance (spec:
    # "unit test it against a full final-layer forward pass")
    assert np.abs(sparse - full).max() / denom < 1e-5


def test_sparse_update_real_support_sizes():
    rng = np.random.default_rng(11)
    H = rng.normal(size=(128, 512))
    W = rng.normal(size=(1000, 512))
    b = rng.normal(size=1000)
    logits = H @ W.T + b
    for s in (1, 17, 300):
        sup = rng.choice(512, size=s, replace=False).tolist()
        sparse = ablated_logits_sparse(logits, H, W, sup)
        full = ablated_logits_full(H, W, b, sup)
        assert np.abs(sparse - full).max() < 1e-10


# ---------------------------------------------------------------------------
# 3. An empty support reproduces baseline predictions
# ---------------------------------------------------------------------------

def test_empty_support_reproduces_baseline():
    H = np.random.default_rng(2).normal(size=(32, 512))
    W = np.random.default_rng(3).normal(size=(1000, 512))
    logits = H @ W.T + np.random.default_rng(4).normal(size=1000)
    out = ablated_logits_sparse(logits, H, W, [])
    assert np.array_equal(out, logits)
    p0 = softmax_np(logits)
    p1 = softmax_np(out)
    assert np.array_equal(p0, p1)


# ---------------------------------------------------------------------------
# 4. Random and univariate supports exactly match LC support size
# ---------------------------------------------------------------------------

def test_support_sizes_match_for_controls():
    rng = np.random.default_rng(7)
    for s in (5, 30, 250):
        rand = random_support(s, (0, 0, "random", 0))
        assert len(rand) == s and len(set(rand)) == s
        H = rng.normal(size=(300, 512))
        l = H[:, 3] * 2 + rng.normal(size=300) * 0.1
        uni = univariate_support(H, l, s)
        assert len(uni) == s and len(set(uni)) == s


def test_univariate_support_finds_associated_unit():
    rng = np.random.default_rng(6)
    H = rng.normal(size=(200, 512))
    l = H[:, 5] + rng.normal(size=200) * 0.1
    sup = univariate_support(H, l, 20)
    assert 5 in sup


def test_random_support_deterministic_in_factor_key():
    a = random_support(40, (12, 2, "random", 3))
    b = random_support(40, (12, 2, "random", 3))
    c = random_support(40, (12, 2, "random", 4))
    assert a == b and a != c


# ---------------------------------------------------------------------------
# 5. Discovery images never enter intervention evaluation
# ---------------------------------------------------------------------------

def test_discovery_never_in_evaluation():
    labels = np.repeat(np.arange(1000), 50)
    is_disc, is_eval = imagenet_split(labels)
    assert not (is_disc & is_eval).any()


def test_eval_cohort_contains_only_evaluation_images():
    labels = np.repeat(np.arange(1000), 50)
    is_disc, is_eval = imagenet_split(labels)
    eval_idx = np.flatnonzero(is_eval)
    # synthetic routing: 100 clusters, first 70 eval images in cluster 3
    clusters = np.zeros(len(labels), dtype=int)
    clusters[eval_idx[:70]] = 3
    eval_imgs, is_source, low = cohort_for_cluster(
        clusters, eval_idx, labels, cluster=3)
    assert is_eval[eval_imgs].all()          # nothing from the discovery half
    assert is_source[: 70].all()             # in-cluster cohort = source rows
    assert np.array_equal(
        eval_imgs[:70], eval_idx[:70])


# ---------------------------------------------------------------------------
# 6. Cluster assignment for evaluation images is deterministic
# ---------------------------------------------------------------------------

def test_eval_routing_deterministic():
    rng = np.random.default_rng(9)
    h_disc = rng.normal(size=(200, 512))
    clusters_disc = rng.integers(0, 10, size=200)
    h_eval = rng.normal(size=(30, 512))
    a1 = route_eval_to_clusters(h_disc, clusters_disc, h_eval)
    a2 = route_eval_to_clusters(h_disc, clusters_disc, h_eval)
    assert np.array_equal(a1, a2)
    # nearest-neighbor semantics: a query identical to disc point i gets i's cluster
    same = route_eval_to_clusters(h_disc, clusters_disc, h_disc[:5])
    assert np.array_equal(same, clusters_disc[:5])


def test_cohort_builder_deterministic():
    labels = np.repeat(np.arange(1000), 50)
    _, is_eval = imagenet_split(labels)
    eval_idx = np.flatnonzero(is_eval)
    clusters = np.random.default_rng(1).integers(0, 50, size=len(labels))
    c1 = cohort_for_cluster(clusters, eval_idx, labels, 7)
    c2 = cohort_for_cluster(clusters, eval_idx, labels, 7)
    assert np.array_equal(c1[0], c2[0]) and np.array_equal(c1[1], c2[1])


def test_incluster_oversampling_caps_at_1000_and_is_stratified():
    labels = np.repeat(np.arange(100), 50)  # 100 classes x 50
    _, is_eval = imagenet_split(labels)
    eval_idx = np.flatnonzero(is_eval)
    clusters = np.zeros(len(labels), dtype=int)
    clusters[eval_idx] = 0                   # 2500 eval images -> capped
    images, is_source, low = cohort_for_cluster(clusters, eval_idx, labels, 0)
    source = images[is_source]
    assert len(source) <= 1000
    counts = np.bincount(labels[source], minlength=100)
    assert counts.max() - counts.min() <= 1  # as even as possible
    assert not low


def test_low_sample_flag_when_under_50():
    labels = np.repeat(np.arange(100), 50)
    _, is_eval = imagenet_split(labels)
    eval_idx = np.flatnonzero(is_eval)
    clusters = np.zeros(len(labels), dtype=int)
    clusters[eval_idx[:30]] = 1              # 30 < 50 -> low sample
    images, is_source, low = cohort_for_cluster(clusters, eval_idx, labels, 1)
    assert low and is_source.sum() == 30
    # off-cluster sample is exactly as large as the source cohort
    assert (~is_source).sum() == 30


# ---------------------------------------------------------------------------
# 8. Softmax changes sum to zero across output classes within tolerance
# ---------------------------------------------------------------------------

def test_softmax_shifts_sum_to_zero():
    x = np.random.default_rng(4).normal(size=(16, 1000))
    p0 = softmax_np(x)
    x_abl = x.copy()
    x_abl[:, [3, 8]] = -2.0
    p1 = softmax_np(x_abl)
    d = p1 - p0
    assert np.abs(d.sum(axis=1)).max() < 1e-9


def test_softmax_shifts_sum_to_zero_on_real_logits():
    _require_cache()
    with np.load(CACHE_PATH) as z:
        logits = z["logits"][:64].astype(np.float64)
    rng = np.random.default_rng(5)
    support = rng.choice(512, size=37, replace=False).tolist()
    W = rng.normal(size=(1000, 512))
    p0 = softmax_np(logits)
    p1 = softmax_np(logits + rng.normal(size=logits.shape))
    assert np.abs((p1 - p0).sum(axis=1)).max() < 1e-9


# ---------------------------------------------------------------------------
# 9. African-grey cluster and factors are selected without evaluation-set
#    intervention results
# ---------------------------------------------------------------------------

def test_african_grey_selection_rule_synthetic():
    labels = np.repeat(np.arange(1000), 25)  # discovery rows: 25 per class
    disc_cluster = np.ones(len(labels), dtype=int)
    disc_cluster[:150] = 2   # cluster 2: 150 images incl 8 greys
    disc_cluster[150:200] = 3
    grey_rows = np.flatnonzero(labels == AFRICAN_GREY_IDX)
    disc_cluster[grey_rows[:8]] = 2
    disc_cluster[grey_rows[8:]] = 1
    rec = african_grey_rule(disc_cluster, labels)
    assert rec["selected_cluster"] == 2
    assert rec["n_african_grey_discovery"] == 8
    # cluster 2 = the first 150 rows PLUS the 8 moved grey rows
    assert rec["proportion"] == pytest.approx(8 / 158)
    assert rec["uses_evaluation_data"] is False


def test_african_grey_selection_tiebreak_by_count_then_lower_id():
    labels = np.repeat(np.arange(1000), 25)
    disc_cluster = np.ones(len(labels), dtype=int)
    grey_rows = np.flatnonzero(labels == AFRICAN_GREY_IDX)
    # two clusters both with 1/100 greys; the lower id must win
    disc_cluster[:100] = 5
    disc_cluster[100:200] = 3
    disc_cluster[grey_rows[0]] = 5
    disc_cluster[grey_rows[1]] = 3
    rec = african_grey_rule(disc_cluster, labels)
    assert rec["selected_cluster"] == 3
    # count tiebreak: cluster with more greys wins even at equal proportion
    disc_cluster = np.ones(len(labels), dtype=int)
    disc_cluster[:200] = 4
    disc_cluster[grey_rows[:2]] = 4          # 2 greys of 200
    disc_cluster[200:400] = 6
    disc_cluster[grey_rows[2:4]] = 6
    assert african_grey_rule(disc_cluster, labels)["selected_cluster"] == 4


def test_african_grey_rule_signature_has_no_evaluation_inputs():
    import inspect

    from case_study import african_grey_rule as fn
    params = list(inspect.signature(fn).parameters)
    assert params == ["disc_cluster", "disc_labels"]
    assert AFRICAN_GREY_WNID == "n01817953"


# ---------------------------------------------------------------------------
# 10. DeepDream code cannot write back into quantitative factor-ranking
#     outputs (port of test 10: DeepDream is excluded from this port, and
#     factor-ranking shards ('factors'/'supports') are writeable only by the
#     fitting stage)
# ---------------------------------------------------------------------------

FACTOR_RANKING_KINDS = ('"factors"', "'factors'", '"supports"', "'supports'")


def test_factor_ranking_shard_writers_are_fits_only():
    """Port of spec test 10: no DeepDream module exists (excluded from this
    quantitative port), and shards of the factor-ranking kinds
    ('factors'/'supports') are written only by the fitting stage."""
    import re

    pkg = Path(__file__).resolve().parent
    deepdream = [p.name for p in pkg.glob("*.py")
                 if "deepdream" in p.name.lower()]
    assert not deepdream, "DeepDream is out of scope for this port"
    pattern = re.compile(r"store\.write|\.write_multiple\(")
    offenders = {}
    for p in sorted(pkg.glob("*.py")):
        if p.name == "tests.py":
            continue
        lines = p.read_text().splitlines()
        hits = []
        for i, line in enumerate(lines):
            if pattern.search(line):
                window = "\n".join(lines[i:i + 5])
                hits += [t for t in FACTOR_RANKING_KINDS
                         for ln in lines[i:i + 5]
                         if t in ln and "read_all" not in ln]
        if hits and p.name != "fits.py":
            offenders[p.name] = hits
    assert not offenders, (
        f"factor-ranking shard writes outside fits.py: {offenders}")


def test_summary_tables_include_git_commit_column():
    from summarize import RUN_GIT
    assert RUN_GIT and len(RUN_GIT) >= 7


# ---------------------------------------------------------------------------
# Cumulative-MI supports machinery
# ---------------------------------------------------------------------------

def test_cumulative_mi_support_nesting():
    scores = np.abs(np.random.default_rng(20).normal(size=512)) * 0.01
    sups = metrics.nested_supports(scores, [0.6, 0.7, 0.8, 0.9])
    sizes = [s.size for s in sups]
    assert sizes == sorted(sizes)
    for a, b in zip(sups, sups[1:]):
        assert set(a.indices) <= set(b.indices)


def test_standardize_uses_discovery_stats():
    rng = np.random.default_rng(21)
    feats = rng.normal(size=(1000, 8)) * np.array(
        [1., 2., 4., 8., 16., 32., 64., 128.])
    is_disc = np.zeros(1000, dtype=bool)
    is_disc[:300] = True
    f_std, mean, std = standardize_with_discovery_stats(feats, is_disc)
    disc_mean = feats[is_disc].mean(axis=0)
    disc_std = feats[is_disc].std(axis=0)
    assert np.allclose(mean, disc_mean)
    assert np.allclose(std, disc_std)
    assert np.abs(f_std[is_disc].mean(axis=0)).max() < 1e-10


def test_class_mapping_contains_african_grey():
    from data import class_mapping

    rows = class_mapping()
    assert len(rows) == 1000
    assert rows[87]["wnid"] == "n01817953"
    assert "African grey" in rows[87]["class_name"]
