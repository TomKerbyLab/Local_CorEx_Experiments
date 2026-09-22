"""Experiment 6: dropout and representational redundancy.

Paired seeds 6100-6109: for each seed, one base initialization is cloned into
two conditions (dropout 0.0 vs 0.5); minibatch order is identical within a
pair; only dropout masks differ (separately derived randomness). Batch
normalization stays on in both conditions.

Metrics per (seed, condition, layer):
  - activation moments (mean, variance, zero rate, skew, kurtosis), effective
    (nondegenerate) dimension
  - Gaussian TC: -0.5 log det Ledoit-Wolf-shrunk correlation, plus per-unit
    normalization
  - CorEx-explained TC: Linear CorEx M=10 on standardized representations
  - both globally (10k discovery) and within the 20 fixed MNIST clusters
    (cluster-size-weighted local mean)

Support analysis: top-5 factors, q in {0.6,0.7,0.8,0.9} supports; primary
comparison is the paired difference in support density at matched (layer,
cluster, factor rank, threshold).

Functional deletion: top-3 factors, q=0.8, 5 matched random supports, exact
activation-mask interventions evaluated on the untouched test set; excess
effect E and stabilized ratio Q.
"""

from __future__ import annotations

import copy
import json
import logging
import time

import numpy as np
import torch

from ..common import config, corex, metrics, seeds
from ..common.io import FailureLog, write_run_report
from ..common.runtime import setup_logging
from ..common.store import ResultStore
from ..common.information import gaussian_total_correlation
from ..exp4_mnist.benchmark import (
    TrunkEvaluator, build_discovery_clusters, extract_representations,
    standardize_with_flags)
from ..exp4_mnist.classifier import MLP
from ..exp4_mnist.data import load_splits
from ..exp4_mnist.interventions import (
    classwise_shift_matrix, effect_metrics)
from ..exp4_mnist.train import checkpoint_hash

EXPERIMENT = "dropout"
logger = logging.getLogger(f"tnnls.{EXPERIMENT}")

PAIR_SEEDS = tuple(range(6100, 6110))
LAYERS = (1, 2, 3)
N_FACTORS = 10
N_RETAINED_SUPPORTS = 5
N_RETAINED_FUNCTIONAL = 3
SUPPORT_THRESHOLDS = (0.60, 0.70, 0.80, 0.90)
FUNCTIONAL_Q = 0.80
N_RANDOM_FUNCTIONAL = 5
ACC_PAIR_TOL = 0.005
VAR_FLOOR = 1e-12

TRAIN_CFG = {
    "lr": 1e-3,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 10,
}

CONDITIONS = ("dropout", "no_dropout")


# ---------------------------------------------------------------------------
# Paired training
# ---------------------------------------------------------------------------

def train_pair(root_seed: int, splits: dict, device: str,
               cfg: dict | None = None, max_epochs_override: int | None = None,
               save: bool = True) -> dict:
    """Train the two dropout conditions from one base initialization.

    Returns {"dropout": model, "no_dropout": model, "meta": {...}}.
    """
    cfg = {**TRAIN_CFG, **(cfg or {})}
    torch.manual_seed(root_seed)

    # one base init; clone into both conditions
    base = MLP(dropout_p=0.5)
    models = {
        "dropout": copy.deepcopy(base),
        "no_dropout": copy.deepcopy(base),
    }
    models["dropout"].drop = torch.nn.Dropout(p=0.5)
    models["no_dropout"].drop = torch.nn.Dropout(p=0.0)
    for m in models.values():
        m.to(device)

    tx = splits["train_x"].to(device)
    ty = splits["train_y"].to(device)
    vx = splits["val_x"].to(device)
    vy = splits["val_y"].to(device)
    lossf = torch.nn.CrossEntropyLoss()

    # identical minibatch orders within a pair (derived per epoch)
    orders = {epoch: torch.randperm(len(tx), generator=torch.Generator().manual_seed(
        seeds.derive_seed(root_seed, "minibatch", epoch))).to(device)
        for epoch in range(max_epochs_override or cfg["max_epochs"])}

    results = {}
    for condition, model in models.items():
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        # separate randomness streams for dropout masks
        torch.manual_seed(seeds.derive_seed(root_seed, condition, "masks"))
        best_val, best_state, best_epoch, no_improve = float("inf"), None, -1, 0
        t0 = time.time()
        max_epochs = max_epochs_override or cfg["max_epochs"]
        for epoch in range(max_epochs):
            model.train()
            perm = orders[epoch]
            for i in range(0, len(tx), cfg["batch_size"]):
                idx = perm[i : i + cfg["batch_size"]]
                loss = lossf(model(tx[idx]), ty[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = model(vx)
                val_loss = lossf(vl, vy).item()
            if val_loss < best_val - 1e-6:
                best_val, best_epoch, no_improve = val_loss, epoch, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= cfg["patience"]:
                    break
        model.load_state_dict(best_state)
        model.eval()
        # discovery + test metrics
        with torch.no_grad():
            disc_logits = model(splits["discovery_x"].to(device))
            disc_acc = (disc_logits.argmax(1)
                        == splits["discovery_y"].to(device)).float().mean().item()
            test_logits = model(splits["test_x"].to(device))
            test_acc = (test_logits.argmax(1)
                        == splits["test_y"].to(device)).float().mean().item()
            test_ce = lossf(test_logits, splits["test_y"].to(device)).item()
        init_hash = checkpoint_hash_of_state(base.state_dict())
        results[condition] = {
            "model": model, "init_hash": init_hash,
            "meta": {
                "root_seed": root_seed, "condition": condition,
                "best_epoch": best_epoch, "best_val_loss": best_val,
                "discovery_acc": disc_acc, "test_acc": test_acc,
                "test_ce": test_ce, "selected_epoch": best_epoch,
                "runtime_s": time.time() - t0,
            },
        }
        if save:
            ckpt_dir = config.CHECKPOINTS_DIR / EXPERIMENT
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / f"dropout_seed{root_seed}_{condition}.pt"
            torch.save(best_state, path)
            results[condition]["meta"]["checkpoint"] = str(path)
            results[condition]["meta"]["checkpoint_hash"] = checkpoint_hash(path)
    results["meta"] = {
        "root_seed": root_seed,
        "init_hash": results["dropout"]["init_hash"],
        "same_init": results["dropout"]["init_hash"] == results["no_dropout"]["init_hash"],
        "acc_diff": abs(results["dropout"]["meta"]["test_acc"]
                        - results["no_dropout"]["meta"]["test_acc"]),
        "comparability_flag": abs(
            results["dropout"]["meta"]["test_acc"]
            - results["no_dropout"]["meta"]["test_acc"]) > ACC_PAIR_TOL,
    }
    return results


def checkpoint_hash_of_state(sd: dict) -> str:
    import hashlib

    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Dependence metrics
# ---------------------------------------------------------------------------

def activation_moments(H: np.ndarray) -> dict:
    """Per-coordinate moments aggregated: mean, variance, zero rate, skew,
    kurtosis; plus effective dimension (variance >= floor)."""
    mean = H.mean(0)
    var = H.var(0)
    zero_rate = (H == 0).mean(0)
    Hc = H - mean
    sd = np.sqrt(var) + 1e-12
    skew = (Hc ** 3).mean(0) / sd ** 3
    kurt = (Hc ** 4).mean(0) / sd ** 4 - 3.0
    return {
        "mean_of_means": float(mean.mean()),
        "mean_var": float(var.mean()),
        "mean_zero_rate": float(zero_rate.mean()),
        "mean_skew": float(skew.mean()),
        "mean_kurtosis": float(kurt.mean()),
        "effective_dim": int((var > VAR_FLOOR).sum()),
        "n_coords": int(H.shape[1]),
    }


def gaussian_tc_ledoit_wolf(H_std: np.ndarray) -> tuple[float, float]:
    """Gaussian TC from Ledoit-Wolf-shrunk correlation (bits) + normalized.

    Returns (TC, TC/d). Raises on non-PD shrunk correlation (recorded as a
    failure by the caller instead of taking an absolute value).
    """
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(H_std)
    R = lw.covariance_
    # correlation from shrunk covariance
    d = np.sqrt(np.diag(R))
    Rcorr = R / np.outer(d, d)
    np.fill_diagonal(Rcorr, 1.0)
    sign = np.linalg.slogdet(Rcorr)[0]
    if sign <= 0:
        raise ValueError("Ledoit-Wolf correlation not positive definite")
    tc = gaussian_total_correlation(Rcorr)
    return tc, tc / Rcorr.shape[0]


def corex_explained_tc(H_std: np.ndarray, n_factors: int, seed: int):
    fit = corex.fit_linear_corex(H_std, n_factors, seed=seed)
    tcs_sorted = np.sort(fit.tcs)[::-1]
    cum = np.cumsum(tcs_sorted) / max(fit.total_tc(), 1e-12)
    return {
        "total_explained": float(fit.total_tc()),
        "factor_tcs": [float(t) for t in tcs_sorted],
        "cum_frac_top1": float(cum[0]) if len(cum) else float("nan"),
        "cum_frac_top3": float(cum[2]) if len(cum) >= 3 else float("nan"),
        "cum_frac_top5": float(cum[4]) if len(cum) >= 5 else float("nan"),
        "cum_frac_top10": float(cum[-1]) if len(cum) else float("nan"),
        "converged": fit.converged,
        "runtime_s": fit.runtime_s,
    }


def run_representations_and_metrics(
    store: ResultStore, root_seed: int, condition: str, model: MLP,
    splits: dict, cluster_labels: np.ndarray, device: str, cfg: dict,
) -> None:
    """Representation stats + dependence metrics for one trained model."""
    cond = {"stage": "reps", "root_seed": root_seed, "condition": condition}
    if store.is_done(cond, ("rep_stats",)):
        return
    x_disc = splits["discovery_x"]
    h1, h2, h3, logits = extract_representations(model, x_disc.to(device), device)
    rep_rows, support_rows = [], []
    for layer in LAYERS:
        H = {1: h1, 2: h2, 3: h3}[layer]
        moments = activation_moments(H)
        H_std, valid, _ = standardize_with_flags(H)
        s_fit = seeds.derive_seed(root_seed, condition, layer, "global")
        # global metrics
        try:
            tc_g, tc_g_norm = gaussian_tc_ledoit_wolf(H_std[:, valid])
        except ValueError as e:
            tc_g, tc_g_norm = float("nan"), float("nan")
            logger.warning("LW not PD: seed %d cond %s layer %d", root_seed,
                           condition, layer)
        ce = corex_explained_tc(H_std[:, valid], cfg["n_factors"], s_fit)
        ce_total = ce.pop("total_explained")
        rep_rows.append({
            "root_seed": root_seed, "condition": condition, "layer": layer,
            "scope": "global", "n_obs": int(H.shape[0]),
            **moments, "gaussian_tc": tc_g, "gaussian_tc_norm": tc_g_norm,
            "corex_explained_tc": ce_total,
            **ce,
        })
        # local (per fixed cluster) metrics
        tc_local_list, wts, corex_local = [], [], []
        for k in range(cfg["n_clusters"]):
            idx = np.flatnonzero(cluster_labels == k)
            if len(idx) < 30:
                continue
            Hk = H_std[idx][:, valid]
            try:
                tk, _ = gaussian_tc_ledoit_wolf(Hk)
            except ValueError:
                tk = float("nan")
            s_k = seeds.derive_seed(root_seed, condition, layer, k)
            ck = corex_explained_tc(Hk, cfg["n_factors"], s_k)
            tc_local_list.append(tk * len(idx))
            wts.append(len(idx))
            corex_local.append(ck["total_explained"] * len(idx))
            # supports for the factor-support analysis (hidden-only fit)
            fit = corex.fit_linear_corex(Hk, cfg["n_factors"], seed=s_k)
            order = np.argsort(-fit.tcs)[: cfg["n_retained_supports"]]
            mis = fit.association_scores()
            for rank, j in enumerate(order):
                sups = metrics.nested_supports(
                    mis[j], list(cfg["support_thresholds"]))
                for q, sup in zip(cfg["support_thresholds"], sups):
                    support_rows.append({
                        "root_seed": root_seed, "condition": condition,
                        "layer": layer, "cluster": k, "factor_rank": rank,
                        "threshold": float(q), "support_size": sup.size,
                        "support_density": sup.density,
                        "achieved_fraction": sup.achieved_fraction,
                        "factor_tc": float(fit.tcs[j]),
                    })
        n_total = sum(wts)
        rep_rows.append({
            "root_seed": root_seed, "condition": condition, "layer": layer,
            "scope": "weighted_local",
            "n_obs": int(H.shape[0]),
            "gaussian_tc": float(np.sum(tc_local_list) / n_total) if n_total else float("nan"),
            "gaussian_tc_norm": float("nan"),
            "corex_explained_tc": float(np.sum(corex_local) / n_total) if n_total else float("nan"),
        })
    store.write(cond, "rep_stats", rep_rows)
    store.write(cond, "supports", support_rows)


# ---------------------------------------------------------------------------
# Functional factor-deletion
# ---------------------------------------------------------------------------

def run_functional_ablation(
    store: ResultStore, root_seed: int, condition: str, model: MLP,
    splits: dict, cluster_labels: np.ndarray, device: str, cfg: dict,
) -> None:
    """Top-3 factors, q=0.8 supports, 5 matched random supports; activation
    masking on the untouched test set; E and Q per factor."""
    cond_meta = {"stage": "functional", "root_seed": root_seed,
                 "condition": condition}
    if store.is_done(cond_meta, ("functional_ablation",)):
        return
    labels_test = splits["test_y"].numpy()
    trunk = TrunkEvaluator(model, splits["test_x"].to(device), device)
    x_disc = splits["discovery_x"]
    rows = []
    for layer in LAYERS:
        for k in range(cfg["n_clusters"]):
            idx = np.flatnonzero(cluster_labels == k)
            if len(idx) < 30:
                continue
            xc = x_disc[torch.tensor(idx)].to(device)
            h1, h2, h3, _ = extract_representations(model, xc, device)
            H = {1: h1, 2: h2, 3: h3}[layer]
            H_std, valid, _ = standardize_with_flags(H)
            Hk = H_std[:, valid]
            s_fit = seeds.derive_seed(root_seed, condition, layer, k)
            fit = corex.fit_linear_corex(Hk, cfg["n_factors"], seed=s_fit)
            order = np.argsort(-fit.tcs)[: cfg["n_retained_functional"]]
            mis = fit.association_scores()
            for rank, j in enumerate(order):
                sup = metrics.cumulative_mi_support(
                    mis[j], cfg["functional_q"])
                sup_list = [int(np.flatnonzero(valid)[u]) for u in sup.indices]
                if not sup_list:
                    continue
                s = len(sup_list)
                base_int = {
                    "root_seed": root_seed, "condition": condition,
                    "layer": layer, "cluster": k, "factor_rank": rank,
                    "support_size": s, "threshold": cfg["functional_q"],
                }
                p_lc = trunk.ablated_probs(layer, sup_list)
                m_lc = _classwise_metrics(p_lc, trunk.baseline_probs, labels_test)
                rand_T = []
                for draw in range(cfg["n_random_functional"]):
                    s_r = seeds.derive_seed(root_seed, condition, "randfun",
                                           layer, k, rank, draw)
                    rng = np.random.default_rng(s_r)
                    rand_sup = rng.choice(np.flatnonzero(valid), size=s,
                                          replace=False).tolist()
                    p_r = trunk.ablated_probs(layer, rand_sup)
                    m_r = _classwise_metrics(p_r, trunk.baseline_probs, labels_test)
                    rand_T.append(m_r["T_max"])
                    rows.append({**base_int, "method": "random",
                                 "random_draw": draw, **m_r,
                                 "factor_tc": float(fit.tcs[j])})
                E = m_lc["T_max"] - float(np.mean(rand_T))
                Q = (m_lc["T_max"] + 1e-6) / (float(np.mean(rand_T)) + 1e-6)
                rows.append({**base_int, "method": "local_corex",
                             "random_draw": 0, **m_lc,
                             "excess_effect_E": E,
                             "log_effect_ratio_Q": float(np.log(Q)),
                             "factor_tc": float(fit.tcs[j])})
    store.write(cond_meta, "functional_ablation", rows)


def _classwise_metrics(p_abl, p_base, labels):
    delta = classwise_shift_matrix(p_abl, p_base, labels)
    n_per_class = np.bincount(labels, minlength=10).astype(float)
    return effect_metrics(delta, n_per_class)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg: dict, stages=("train", "reps", "functional"),
         seeds_list: tuple[int, ...] = PAIR_SEEDS) -> dict:
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    defaults = {
        "n_factors": N_FACTORS,
        "n_retained_supports": N_RETAINED_SUPPORTS,
        "n_retained_functional": N_RETAINED_FUNCTIONAL,
        "support_thresholds": list(SUPPORT_THRESHOLDS),
        "functional_q": FUNCTIONAL_Q,
        "n_random_functional": N_RANDOM_FUNCTIONAL,
        "n_clusters": 20,
        "layers": list(LAYERS),
    }
    defaults.update({k: v for k, v in cfg.items() if k in defaults})
    run_id = config.make_run_id(EXPERIMENT, defaults, 0)
    setup_logging(EXPERIMENT, run_id)
    if config.is_complete(run_id):
        return {"run_id": run_id, "status": "already_complete"}
    config.save_resolved_config(run_id, defaults,
                                extra={"stages": list(stages), "seeds": list(seeds_list)})
    store = ResultStore(EXPERIMENT)
    failure_log = FailureLog(EXPERIMENT)

    splits = load_splits(save_dir=config.RAW_DIR / "mnist")
    mnist_store = ResultStore("mnist")
    cluster_labels = build_discovery_clusters(splits, mnist_store,
                                              int(defaults["n_clusters"]))
    done = 0
    for root_seed in seeds_list:
        try:
            pair = train_pair(root_seed, splits, device, max_epochs_override=cfg.get("max_epochs"))
            for condition in CONDITIONS:
                model = pair[condition]["model"]
                if "reps" in stages:
                    run_representations_and_metrics(
                        store, root_seed, condition, model, splits,
                        cluster_labels, device, defaults)
                if "functional" in stages:
                    run_functional_ablation(
                        store, root_seed, condition, model, splits,
                        cluster_labels, device, defaults)
            done += 1
            logger.info("pair %d done (acc diff %.4f, flag=%s)", root_seed,
                        pair["meta"]["acc_diff"],
                        pair["meta"]["comparability_flag"])
        except Exception as e:
            import traceback
            failure_log.record(condition={"root_seed": root_seed},
                               seed=root_seed, stage="pair", exc=e,
                               message=traceback.format_exc(limit=3))
    config.mark_complete(run_id, {"pairs_done": done})
    write_run_report(EXPERIMENT, run_id, done, len(seeds_list) - done, {})
    return {"run_id": run_id, "pairs_done": done}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--stages", type=str, default="train,reps,functional")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = config.load_config(args.config) if args.config else {}
    if args.smoke:
        cfg = {"n_clusters": 2, "n_factors": 4, "n_retained_supports": 2,
               "n_retained_functional": 1, "support_thresholds": [0.6, 0.8],
               "n_random_functional": 2, "max_epochs": 1, "layers": [3]}
        seeds_list = (6100,)
    else:
        seeds_list = tuple(args.seeds) if args.seeds else PAIR_SEEDS
    if args.epochs:
        cfg["max_epochs"] = args.epochs
    main(cfg, stages=tuple(args.stages.split(",")), seeds_list=seeds_list)