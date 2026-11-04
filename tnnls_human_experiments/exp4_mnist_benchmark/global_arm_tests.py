"""Unit tests for the global CorEx comparison arm (mirrors the benchmark's
intervention-equivalence and support-matching checks)."""

import numpy as np
import pytest
import torch

from benchmark import TrunkEvaluator
from common import config, seeds
from data import load_splits
from global_correx_arm import BUDGETS, fit_seed, load_model, tv_grouped


@pytest.fixture(scope="module")
def splits():
    return load_splits(save_dir=config.RAW_DIR / "mnist")


@pytest.fixture(scope="module")
def trunk(splits):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(5100, device)
    return TrunkEvaluator(model, splits["test_x"], device), device


def test_checkpoint_determinism(trunk):
    """Two loads of the same checkpoint give the same baseline."""
    trunk_env, device = trunk
    other = TrunkEvaluator(load_model(5100, device), trunk_env.test_x, device)
    assert np.allclose(trunk_env.baseline_probs, other.baseline_probs)


def test_empty_support_reproduces_baseline(trunk):
    trunk_env, _ = trunk
    for layer in (1, 2, 3):
        p = trunk_env.ablated_probs(layer, [])
        assert np.allclose(p, trunk_env.baseline_probs)


def test_cached_trunk_equals_model_mask_forward(trunk, splits):
    """The trunk path and a masked full forward pass agree on a sample."""
    trunk_env, device = trunk
    rng = np.random.default_rng(0)
    for layer in (2, 3):
        sup = rng.choice(trunk_env.h[layer].shape[1],
                         size=30, replace=False).tolist()
        p_trunk = trunk_env.ablated_probs(layer, sup)
        p_mask = trunk_env.probs_from_model_mask(layer, sup)
        assert np.allclose(p_trunk, p_mask, atol=1e-6)


def test_random_supports_exact_size(trunk):
    trunk_env, _ = trunk
    s = 137
    seed = seeds.derive_seed(5100, "global_rand", "output_aware", 2, 10, 0,
                             0.8, 3)
    sup = np.random.default_rng(seed).choice(
        trunk_env.h[2].shape[1], size=s, replace=False)
    assert len(sup) == len(set(sup)) == s


def test_fit_seed_matches_benchmark_convention():
    a = seeds.derive_seed(5100, "corex", "global", "output_aware", 1, 10)
    b = seeds.derive_seed(5101, "corex", "global", "output_aware", 1, 10)
    assert a != b
    assert a == seeds.derive_seed(5100, "corex", "global", "output_aware", 1, 10)


def test_fit_seed_pools_differ_by_condition():
    assert fit_seed(5100, "output_aware", 1, 10) != fit_seed(
        5100, "hidden_only", 1, 10)


def test_budgets_distinct():
    assert len(set(BUDGETS)) == len(BUDGETS)


def test_tv_grouped(trunk, splits):
    trunk_env, _ = trunk
    rng = np.random.default_rng(1)
    test_cluster = rng.integers(0, 20, size=len(trunk_env.baseline_probs))
    tv = rng.random(len(trunk_env.baseline_probs))
    grouped = tv_grouped(tv, test_cluster)
    assert len(grouped) == 20
    for k in range(20):
        idx = np.flatnonzero(test_cluster == k)
        assert len(idx) == 0 or abs(grouped[k] - tv[idx].mean()) < 1e-9
