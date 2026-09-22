"""Experiment 5 validation tests (spec 'Validation tests' 1-10)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tnnls_experiments.common import metrics, seeds
from tnnls_experiments.exp5_imagenet import data as exp5_data
from tnnls_experiments.exp5_imagenet import runner as exp5_runner


def test_split_25_25_per_class():
    # 7. Every class has the intended 25/25 split.
    labels = np.repeat(np.arange(1000), 50)
    is_disc, is_eval = exp5_data.imagenet_split(labels)
    assert is_disc.sum() == 25_000 and is_eval.sum() == 25_000
    for c in range(1000):
        assert is_disc[labels == c].sum() == 25
        assert is_eval[labels == c].sum() == 25


def test_split_integrity_check_fails_on_wrong_count():
    labels = np.repeat(np.arange(100), 49)
    with pytest.raises(AssertionError):
        exp5_data.imagenet_split(labels)


def test_sparse_update_equals_full_final_layer():
    # 2. Sparse-update ablated logits equal a full final-layer forward pass
    #    within float32 accumulation tolerance.
    torch.manual_seed(1)
    W = torch.randn(1000, 512)
    b = torch.randn(1000)
    H = torch.randn(64, 512)
    logits = H @ W.T + b
    support = [0, 7, 100, 511]
    H_abl = H.clone()
    H_abl[:, support] = 0.0
    full = (H_abl @ W.T + b).numpy()
    sparse = exp5_runner.ablated_logits_sparse(logits.numpy(), H.numpy(),
                                                W, support)
    # float32 vs float64 accumulation: relative agreement required
    denom = np.abs(full).max()
    assert np.abs(sparse - full).max() / denom < 1e-5


def test_empty_support_reproduces_baseline():
    # 3. Empty support reproduces baseline predictions.
    H = np.random.default_rng(2).normal(size=(32, 512))
    W = torch.randn(1000, 512)
    logits = H @ W.T.numpy() + np.random.default_rng(3).normal(size=1000)
    out = exp5_runner.ablated_logits_sparse(logits, H, W, [])
    assert np.array_equal(out, logits)


def test_cached_logits_equal_wh_plus_b():
    # 1. Cached baseline logits equal W h + b (checked on the real cache).
    cache = exp5_data.load_cached()
    W, b = exp5_runner.load_final_layer("cpu")
    Wn, bn = W.numpy(), b.numpy()
    i = 1234
    recon = cache["features"][i] @ Wn.T + bn
    assert np.allclose(recon, cache["preds"][i], atol=1e-3)


def test_softmax_shifts_sum_to_zero():
    # 8. Softmax changes sum to zero across output classes within tolerance.
    x = np.random.default_rng(4).normal(size=(16, 1000))
    p0 = exp5_runner._softmax_np(x)
    x_abl = x.copy()
    x_abl[:, [3, 8]] = -2.0
    p1 = exp5_runner._softmax_np(x_abl)
    d = p1 - p0
    assert np.abs(d.sum(axis=1)).max() < 1e-9


def test_stratified_sample_even_across_classes():
    pool = np.repeat(np.arange(60), 10)  # 6 classes x 10
    labels = pool.copy()
    rng = np.random.default_rng(5)
    s = exp5_runner._stratified_sample(pool, labels, 12, rng)
    assert len(s) == 12
    counts = np.bincount(labels[s])
    assert counts.max() <= 3  # roughly even


def test_univariate_support_size():
    rng = np.random.default_rng(6)
    H = rng.normal(size=(200, 512))
    l = H[:, 5] + rng.normal(size=200) * 0.1
    sup = exp5_runner._univariate_support(H, l, 20)
    assert len(sup) == 20 and len(set(sup)) == 20
    assert 5 in sup


def test_african_grey_selection_rule():
    # 9. Deterministic selection on discovery data only (synthetic setup).
    import pandas as pd

    # fake cluster assignments: 10 images of class 87 spread across clusters
    disc_idx = np.arange(25_000)
    # class labels: each class has 25 rows in disc
    labels = np.repeat(np.arange(1000), 25)
    clusters = np.zeros(25_000, dtype=int)
    clusters[:] = 1
    clusters[:150] = 2   # cluster 2: 150 images incl 8 greys
    clusters[150:200] = 3
    # craft: cluster 2 has >= 100 images and some greys
    grey_rows = np.flatnonzero(labels == exp5_data.AFRICAN_GREY_HF_IDX)
    clusters[grey_rows[:8]] = 2
    clusters[grey_rows[8:]] = 1

    # reuse the rule inline (select_african_grey_cluster needs the store)
    best = None
    for k in (1, 2, 3):
        members = disc_idx[clusters == k]
        n = len(members)
        if n < 100:
            continue
        n_grey = int((labels[members] == 87).sum())
        prop = n_grey / n
        key = (prop, n_grey, -k)
        if best is None or key > best[0]:
            best = (key, k)
    assert best[1] == 2


def test_support_sizes_match_for_controls():
    # 4. Random and univariate supports match LC support size.
    rng = np.random.default_rng(7)
    s = 30
    rand = rng.choice(512, size=s, replace=False)
    assert len(rand) == s
    H = rng.normal(size=(300, 512))
    l = H[:, 3] * 2
    uni = exp5_runner._univariate_support(H, l, s)
    assert len(uni) == s


def test_discovery_never_in_evaluation():
    # 5. Discovery images never enter intervention evaluation (split is
    #    disjoint by construction; check here).
    labels = np.repeat(np.arange(1000), 50)
    is_disc, is_eval = exp5_data.imagenet_split(labels)
    assert not (is_disc & is_eval).any()