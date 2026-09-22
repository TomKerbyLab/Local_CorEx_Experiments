"""Experiment 4 validation tests (spec 'Ablation validation tests' 1-14)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tnnls_experiments.common import corex, metrics, seeds
from tnnls_experiments.exp4_mnist.classifier import (
    MLP, weight_rows_deletion_logits)
from tnnls_experiments.exp4_mnist.data import make_splits, stratified_indices
from tnnls_experiments.exp4_mnist.interventions import (
    accuracy_metrics, baseline_cache, classwise_shift_matrix, effect_metrics,
    evaluate_intervention, predict_proba, random_control_probability)


def _small_model(seed: int = 0) -> MLP:
    torch.manual_seed(seed)
    m = MLP(dropout_p=0.5)
    m.eval()
    return m


def _dummy_data(n=512, seed=1):
    torch.manual_seed(seed)
    x = torch.rand(n, 784)
    y = torch.randint(0, 10, (n,))
    return x, y


def test_all_ones_mask_reproduces_baseline():
    # 1. All-ones mask == baseline within 1e-7 (CPU).
    m = _small_model()
    x, _ = _dummy_data()
    m.clear_masks()
    base = m(x)
    # mask with empty support equals all-ones mask
    m.set_mask(2, [])
    with torch.no_grad():
        out = m(x)
    assert torch.allclose(out, base, atol=1e-7)


def test_empty_support_baseline():
    # 2. Empty support produces baseline predictions.
    m = _small_model()
    x, _ = _dummy_data()
    base = baseline_cache(m, x)
    m.set_mask(1, None)
    with torch.no_grad():
        out = m(x)
    assert torch.allclose(out, base["logits"], atol=1e-7)


def test_masked_coordinates_exactly_zero():
    # 3. Masked activation coordinates are exactly zero at intervention point.
    m = _small_model()
    x, _ = _dummy_data(256)
    idx = [3, 17, 200, 499]
    m.set_mask(1, idx)
    h1 = m.hidden_forward(x, 1)
    assert torch.equal(h1[:, idx], torch.zeros_like(h1[:, idx]))


def test_unmasked_coordinates_unchanged():
    # 4. Unmasked coordinates equal their baseline values.
    m = _small_model()
    x, _ = _dummy_data(256)
    m.clear_masks()
    h1_base = m.hidden_forward(x, 1)
    keep = [i for i in range(500) if i not in (3, 17, 200, 499)]
    m.set_mask(1, [3, 17, 200, 499])
    h1_masked = m.hidden_forward(x, 1)
    assert torch.allclose(h1_masked[:, keep], h1_base[:, keep], atol=1e-7)


def test_bn_running_stats_unchanged_during_eval():
    # 5. BN running statistics do not change during evaluation.
    m = _small_model()
    x, _ = _dummy_data(256)
    before = m.bn1.running_mean.clone(), m.bn1.running_var.clone()
    m.set_mask(1, [5, 10])
    with torch.no_grad():
        _ = m(x)
    assert torch.equal(before[0], m.bn1.running_mean)
    assert torch.equal(before[1], m.bn1.running_var)


def test_batch_size_invariance():
    # 6. Repeating an ablation with a different evaluation batch size gives
    #    the same predictions within tolerance.
    m = _small_model()
    x, _ = _dummy_data(1024)
    m.set_mask(3, [1, 2, 3])
    p1 = predict_proba(m, x, batch_size=128)
    p2 = predict_proba(m, x, batch_size=1024)
    assert torch.allclose(p1, p2, atol=1e-6)


def test_activation_mask_equals_weight_deletion():
    # 7. Activation masking agrees with outgoing-weight deletion on a
    #    deterministic network at eval mode (dropout disabled).
    torch.manual_seed(3)
    m = MLP(dropout_p=0.5)
    m.eval()  # dropout off
    x, _ = _dummy_data(256)
    idx = [0, 5, 111, 299]
    # activation mask on layer 3
    m.set_mask(3, idx)
    with torch.no_grad():
        act = m(x)
    # weight-column deletion clone
    w = weight_rows_deletion_logits(m, x, 3, idx)
    assert torch.allclose(act, w, atol=1e-6), \
        "activation masking and weight deletion must agree in eval mode"


def test_support_sizes_match_exactly():
    # 8. Local CorEx, random, and univariate supports can be made identical
    #    sizes by construction (checked end-to-end in benchmark tests); here:
    #    random supports sampled to requested size.
    rng = np.random.default_rng(5)
    valid = np.arange(300)
    s = 17
    draw = rng.choice(valid, size=s, replace=False)
    assert len(set(draw.tolist())) == s


def test_random_supports_no_repeats():
    # 9. Random supports contain no repeated indices (set_mask enforces).
    m = _small_model()
    with pytest.raises(AssertionError):
        m.set_mask(2, [4, 4, 5])


def test_cumulative_supports_nested():
    # 11. Cumulative-MI supports are nested.
    scores = np.abs(np.random.default_rng(6).normal(size=300))
    sups = metrics.nested_supports(scores, [0.6, 0.7, 0.8, 0.9])
    for a, b in zip(sups, sups[1:]):
        assert set(a.indices) <= set(b.indices)


def test_probability_shifts_sum_to_zero():
    # 12. Shifts sum to zero across output classes within tolerance.
    torch.manual_seed(7)
    m = MLP(dropout_p=0.5).eval()
    x, y = _dummy_data(512)
    base = baseline_cache(m, x)
    m.set_mask(2, [10, 20, 30])
    p_abl = predict_proba(m, x).numpy()
    p_0 = base["probs"].numpy()
    delta = classwise_shift_matrix(p_abl, p_0, y.numpy())
    assert np.abs(delta.sum(axis=1)).max() < 1e-6


def test_mask_uniquely_determines_intervention():
    # 13. Every saved intervention key uniquely determines its mask: set_mask
    #     with given indices is deterministic and recoverable.
    m = _small_model()
    idx = [7, 42, 299]
    m.set_mask(1, idx)
    active = m.active_masks()
    assert set(active[1]) == set(idx)
    m.clear_masks()
    assert m.active_masks() == {}


def test_effect_metrics_hand_check():
    # hand-check T_y, H_input on a synthetic shift matrix
    delta = np.zeros((10, 10))
    delta[3, 3] = -0.6   # strong single-class effect
    delta[3, 5] = 0.6
    n = np.ones(10) * 100
    m = effect_metrics(delta, n)
    assert m["T_max"] == pytest.approx(0.6)
    assert m["y_star"] == 3
    assert m["c_minus"] == 3
    assert m["c_plus"] == 5
    assert m["D_input"] == pytest.approx(1.0)
    assert m["H_input"] == pytest.approx(0.0, abs=1e-9)
    assert m["H_output"] == pytest.approx(np.log10(2), abs=1e-9)  # two equal masses
    assert m["D_output"] == pytest.approx(0.5)


def test_random_control_probability():
    # p = (1 + #{r >= lc}) / (R+1)
    p = random_control_probability(0.5, [0.1, 0.2, 0.6])
    assert p == pytest.approx(2 / 4)   # one random >= 0.5 -> (1+1)/4
    p_hi = random_control_probability(0.9, [0.1, 0.2])
    assert p_hi == pytest.approx(1 / 3)
    p_lo = random_control_probability(0.1, [0.1, 0.2], larger_is_extreme=False)
    assert p_lo == pytest.approx(2 / 3)


def test_stratified_split_proportions():
    # splits preserve digit proportions and are deterministic
    labels = np.repeat(np.arange(10), 600)  # 6000 labels
    np.random.shuffle(labels)
    s1 = stratified_indices(labels, {"a": 2700, "b": 300, "c": 3000}, seed=5000)
    s2 = stratified_indices(labels, {"a": 2700, "b": 300, "c": 3000}, seed=5000)
    assert all(np.array_equal(s1[k], s2[k]) for k in s1)
    for k, sizes in (("a", 2700), ("b", 300), ("c", 3000)):
        props = np.bincount(labels[s1[k]], minlength=10) / sizes
        # small splits (300 = 30/class) have integer granularity ~1/30 per class
        tol = max(0.02, 1.5 / (sizes / 10))
        assert np.abs(props - 0.1).max() < tol


def test_hidden_states_dims():
    m = _small_model()
    x, _ = _dummy_data(64)
    logits, h1, h2, h3 = m.hidden_states(x)
    assert logits.shape == (64, 10)
    assert h1.shape == (64, 500) and h2.shape == (64, 400) and h3.shape == (64, 300)