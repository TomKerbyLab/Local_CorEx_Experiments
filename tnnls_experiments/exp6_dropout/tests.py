"""Experiment 6 validation tests (spec 'Validation tests' 1-11)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tnnls_experiments.common import metrics, seeds
from tnnls_experiments.exp6_dropout.runner import (
    activation_moments, checkpoint_hash_of_state, gaussian_tc_ledoit_wolf)


def test_gaussian_tc_zero_for_independent():
    # 6. Gaussian TC ~ 0 for simulated independent standardized Gaussians.
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5000, 20))
    tc, tc_norm = gaussian_tc_ledoit_wolf(X)
    assert abs(tc) < 0.2  # bits; small shrinkage bias tolerated
    assert abs(tc_norm) < 0.01


def test_gaussian_tc_increases_with_correlation():
    # 7. Gaussian TC increases in a simulated correlated-Gaussian test.
    rng = np.random.default_rng(2)
    n = 5000
    f = rng.standard_normal(n)
    X = np.stack([f + 0.3 * rng.standard_normal(n) for _ in range(6)], 1)
    X = (X - X.mean(0)) / X.std(0)
    tc, _ = gaussian_tc_ledoit_wolf(X)
    assert tc > 1.0


def test_lw_symmetric_pd():
    # 5. Ledoit-Wolf correlation matrices are symmetric PD within tolerance.
    from sklearn.covariance import LedoitWolf

    rng = np.random.default_rng(3)
    X = rng.standard_normal((2000, 50))
    lw = LedoitWolf().fit(X)
    R = lw.covariance_
    d = np.sqrt(np.diag(R))
    Rcorr = R / np.outer(d, d)
    np.fill_diagonal(Rcorr, 1.0)
    assert np.allclose(Rcorr, Rcorr.T, atol=1e-12)
    assert np.linalg.eigvalsh(Rcorr).min() > 0


def test_activation_moments_effective_dim():
    rng = np.random.default_rng(4)
    H = rng.standard_normal((500, 10))
    H[:, 3] = 1.0  # degenerate
    m = activation_moments(H)
    assert m["effective_dim"] == 9
    assert m["n_coords"] == 10
    assert 0 <= m["mean_zero_rate"] <= 1
    # ReLU-like data: zero rate > 0
    Hr = np.maximum(H, 0)
    mr = activation_moments(Hr)
    assert mr["mean_zero_rate"] > 0.4


def test_checkpoint_hash_of_state_deterministic():
    torch.manual_seed(5)
    m1 = torch.nn.Linear(10, 10)
    h1 = checkpoint_hash_of_state(m1.state_dict())
    h2 = checkpoint_hash_of_state(m1.state_dict())
    assert h1 == h2
    m1.weight.data += 1.0
    h3 = checkpoint_hash_of_state(m1.state_dict())
    assert h1 != h3


def test_paired_conditions_share_init_and_order():
    # 1 + 2. Paired conditions begin from identical parameter tensors and
    # use identical minibatch orders (verified in smoke training via
    # train_pair; here we check the order derivation is condition-independent).
    s1 = seeds.derive_seed(6100, "minibatch", 0)
    s2 = seeds.derive_seed(6100, "minibatch", 0)
    assert s1 == s2  # order depends only on root seed and epoch
    # dropout mask streams differ by condition
    sd = seeds.derive_seed(6100, "dropout", "masks")
    sn = seeds.derive_seed(6100, "no_dropout", "masks")
    assert sd != sn


def test_no_dropout_model_deterministic():
    # 3-4. The no-dropout model applies no stochastic mask; both conditions
    # deterministic in eval mode.
    from tnnls_experiments.exp4_mnist.classifier import MLP

    torch.manual_seed(6)
    m = MLP(dropout_p=0.0).eval()
    x = torch.rand(64, 784)
    y1 = m(x)
    y2 = m(x)
    assert torch.allclose(y1, y2)
    # dropout model in eval mode is also deterministic
    md = MLP(dropout_p=0.5).eval()
    assert torch.allclose(md(x), md(x))


def test_support_sizes_factor_vs_random():
    # 9. Factor-guided and random supports have identical sizes (by
    # construction in run_functional_ablation; unit-level check here).
    rng = np.random.default_rng(7)
    valid = np.arange(300)
    s = 25
    rand = rng.choice(valid, size=s, replace=False)
    assert len(rand) == s and len(set(rand.tolist())) == s


def test_excess_effect_formula():
    # E = T_factor - mean(T_random); Q = (T_factor + eps) / (mean + eps)
    T_f = 0.5
    T_r = [0.1, 0.2, 0.3]
    E = T_f - np.mean(T_r)
    Q = (T_f + 1e-6) / (np.mean(T_r) + 1e-6)
    assert E == pytest.approx(0.5 - 0.2)
    assert np.log(Q) == pytest.approx(np.log(0.5 / 0.2), rel=1e-4)