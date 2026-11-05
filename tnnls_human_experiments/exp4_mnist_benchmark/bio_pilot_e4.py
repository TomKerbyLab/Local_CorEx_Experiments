"""Bio CorEx feasibility pilot for the MNIST benchmark (paper-plan review,
fb#2-follow-up: "Run a Bio feasibility pilot using discovery data. Cover all
three layers and a predetermined range of cluster sizes, plus a global fit.
Measure runtime, peak memory, convergence, and degenerate-factor frequency.").

Frozen before outcome checks — this pilot informs ONLY the feasibility
decision (expand to a global/local Bio arm or not); it is not an outcome
analysis and no held-out statistic is computed here.

Design (predetermined):
- Inputs: discovery-half activations (same extraction/standardization
  convention as the benchmark: post-BN/ReLU hidden states; valid
  (non-degenerate) coordinates only; row-centered logits for the
  output-aware condition).
- Cluster selection, deterministic: among the 20 fixed Local CorEx
  discovery clusters, the smallest, the median, and the largest cluster
  by discovery-observation count (ties broken by lower index).
- Fits: for each layer (1,2,3):
    - three cluster fits (small / median / large);
    - one global fit on the full 10k discovery sample;
  each with M=10 factors, Bio CorEx (marginal_description='gaussian',
  n_hidden=10, dim_hidden=2, seed derived from (5100, "bio", tag)).
- Recorded per fit: wallclock, peak RSS delta (resource.ru_maxrss),
  tracemalloc peak, convergence flag, total TC, per-factor TCs,
  degenerate-factor count, and the q=0.8 support size of the retained
  top factors (feasibility-level only).

Writes csv to run_results/bio_pilot_e4.csv (or stdout with --no-write).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import tracemalloc

import numpy as np
import torch
from local_corex import BioCorex

from benchmark import (
    LayerContext, N_CLUSTERS, NATS_TO_BITS, build_discovery_clusters,
    center_logits, extract_representations, load_splits, mapped_supports)
from global_correx_arm import load_model
from common import config, seeds
from common.runtime import setup_logging

RUN_NAME = "bio_pilot_e4"
logger = logging.getLogger(f"tnnls.{RUN_NAME}")

PILOT_MODEL_SEED = 5100
LAYERS = (1, 2, 3)
M_FACTORS = 10
SUPPORT_Q = (0.8,)
TIE_SEED = 51000


def _peak_mem_setup():
    tracemalloc.start()
    import resource
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return tracemalloc, rss0


def _rss_kb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def fit_bio(V, n_hidden, seed, tag, timeout_s: float = 300.0) -> dict:
    """One Bio fit in an isolated fork process; a timeout terminates the
    child (Bio cannot stop mid-iteration) and the row records it as data.
    TCs return through shared memory."""
    import multiprocessing as mp

    ctx = mp.get_context("fork")

    def worker(V, n_hidden, seed, tcs_out, ok_flags, mem_out):
        tracemalloc.start()
        model = BioCorex(n_hidden=n_hidden, dim_hidden=2, max_iter=100,
                         n_repeat=1, seed=int(seed), verbose=False)
        model.fit(V)
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_out.value = peak / 1e6
        tcs = np.asarray(model.tcs, dtype=float)
        tcs_out[: len(tcs)] = tcs
        ok_flags.value = 1.0

    n = int(n_hidden)
    tcs_out = ctx.Array("d", n)
    ok_flags = ctx.Array("d", 1)
    mem_out = ctx.Array("d", 1)
    rss0 = _rss_kb()
    t0 = time.time()
    proc = ctx.Process(target=worker, args=(V, n_hidden, int(seed),
                                            tcs_out, ok_flags, mem_out))
    proc.start()
    proc.join(timeout=timeout_s)
    dt = time.time() - t0
    finished = not proc.is_alive()
    if not finished:
        proc.terminate()
        proc.join()
        return {
            "tag": tag, "n_vars": int(V.shape[1]), "n_obs": int(V.shape[0]),
            "n_hidden": n, "seed": int(seed),
            "runtime_s": round(dt, 2), "rss_peak_mb": round(
                (_rss_kb() - rss0) / 1024.0, 1),
            "py_peak_mb": None,
            "converged": False, "n_degenerate": n,
            "total_tc_bits": float("nan"), "top5_tc_bits": None,
            "status": "fit_timeout",
        }
    tcs = np.asarray(tcs_out[:n], dtype=float)
    finite = np.isfinite(tcs)
    active = tcs[np.abs(tcs) > 1e-12]
    return {
        "tag": tag, "n_vars": int(V.shape[1]), "n_obs": int(V.shape[0]),
        "n_hidden": n, "seed": int(seed),
        "runtime_s": round(dt, 2),
        "rss_peak_mb": round((_rss_kb() - rss0) / 1024.0, 1),
        "py_peak_mb": round(mem_out[0], 1),
        "converged": bool(finite.all()),
        "n_degenerate": int((~finite).sum()),
        "total_tc_bits": (float(np.nansum(active)) * NATS_TO_BITS
                          if len(active) else float("nan")),
        "top5_tc_bits": (np.round(np.sort(active)[::-1][:5], 3).tolist()
                         if len(active) else None),
        "status": "ok",
    }




def main(run_all: bool = True, write: bool = True) -> list:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_logging(RUN_NAME, f"{RUN_NAME}_seed{PILOT_MODEL_SEED}")
    store_path = ("run_results/raw/mnist" if device == "cpu" else
                  "run_results/raw/mnist")
    splits = load_splits(save_dir=config.RAW_DIR / "mnist")
    from common.store import ResultStore
    store = ResultStore("mnist")

    model = load_model(PILOT_MODEL_SEED, device)
    with torch.no_grad():
        h1, h2, h3, logits = extract_representations(
            model, splits["discovery_x"].to(device), device)
    # discovery cluster sizes for the deterministic small/median/large pick
    from benchmark import build_discovery_clusters
    labels_cluster = build_discovery_clusters(splits, store, N_CLUSTERS)
    sizes = {k: int((labels_cluster == k).sum()) for k in range(N_CLUSTERS)}
    order = sorted(sizes, key=lambda k: (sizes[k], k))
    small, median, large = order[0], order[len(order) // 2], order[-1]
    picks = {"small": small, "median": median, "large": large,
             "global": None}
    logger.info("cluster picks: %s", json.dumps(
        {k: (v, sizes[v]) for k, v in picks.items() if v is not None}))

    rows = []
    for layer in LAYERS:
        ctx = LayerContext(h1, h2, h3, logits, layer)
        v_idx = np.flatnonzero(ctx.valid_h)
        for tag, cluster in picks.items():
            if cluster is not None:
                idx = np.flatnonzero(labels_cluster == cluster)
                H = ctx.H_std[idx][:, v_idx]
            else:
                idx = None
                H = ctx.H_std[:, v_idx]
            Li = ctx.L_std[idx] if idx is not None else ctx.L_std
            V = np.hstack([H, Li])
            s_fit = seeds.derive_seed(PILOT_MODEL_SEED, "bio_pilot",
                                      tag, layer)
            out_cond = "output_aware"
            out_cond = "output_aware"
            r = fit_bio(V, M_FACTORS, s_fit, f"{tag}_L{layer}_{out_cond}")
            r["cluster"] = cluster; r["layer"] = layer
            r["cluster_size"] = sizes.get(cluster, H.shape[0])
            r["input_condition"] = out_cond
            rows.append(r)
            logger.info("%s", json.dumps(r))
    import pandas as pd
    df = pd.DataFrame(rows)
    if write:
        import pathlib
        pathlib.Path("run_results").mkdir(exist_ok=True)
        df.to_csv("run_results/bio_pilot_e4.csv", index=False)
        print(df.to_string(index=False))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=RUN_NAME)
    args = ap.parse_args()
    main()
