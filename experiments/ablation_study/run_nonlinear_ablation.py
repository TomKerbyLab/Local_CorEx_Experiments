from __future__ import annotations

import json
from time import perf_counter
import traceback
from pathlib import Path

import pandas as pd

from synthetic_common import (
    evaluate_global_setting,
    evaluate_local_from_partition,
    partition_data,
)
from synthetic_nonlinear import generate_nonlinear_dataset


OUT_DIR = Path("./ablation_data_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_OUT = OUT_DIR / "nonlinear_ablation.csv"
FAILED_OUT = OUT_DIR / "nonlinear_ablation_failed_setups.csv"

REPS = 16
SAMPLE_SIZES = [50, 500, 5000]
N_NOISE = [0, 25, 75]
N_FACTORS = [5]
PARTITIONS = ["oracle", "phate", "raw_kmeans", "random"]
MODELS = ["linear", "bio"]


def main():
    rows = []
    failed_rows = []

    total_experiments = (
        len(range(REPS)) * len(SAMPLE_SIZES) * len(N_NOISE)
    )
    completed_experiments = 0
    start_time = perf_counter()

    def _save_rows(rows, out_path: Path):
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)

    def _format_eta(seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        minutes, secs = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h {minutes}m {secs}s"
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    for rep in range(REPS):
        for cluster_size in SAMPLE_SIZES:
            for n_noise in N_NOISE:
                completed_experiments += 1
                experiment_start = perf_counter()
                setup = {
                    "dataset": "nonlinear",
                    "rep": rep,
                    "cluster_size": cluster_size,
                    "n_noise": n_noise,
                }

                try:
                    data = generate_nonlinear_dataset(
                        rep=rep,
                        cluster_size=cluster_size,
                        n_noise=n_noise,
                        n_groups=3,
                        arity_a=4,
                        arity_b=4,
                    )

                    X = data["X"]
                    y_true = data["y_true"]
                    true_groups = data["true_groups"]
                    all_groups = true_groups["clust_A"] + true_groups["clust_B"]

                    # Global models: fit once per model / factor count.
                    for model_kind in MODELS:
                        for n_hidden in N_FACTORS:
                            global_scores = evaluate_global_setting(
                                X=X,
                                all_groups=all_groups,
                                model_kind=model_kind,
                                n_hidden=n_hidden,
                            )
                            rows.append(
                                {
                                    **setup,
                                    "partition_method": "global",
                                    "model_kind": model_kind,
                                    "n_hidden": n_hidden,
                                    **global_scores,
                                }
                            )

                    # Partition once per partition method, then reuse.
                    partition_cache = {}
                    for partition_method in PARTITIONS:
                        partition_cache[partition_method] = partition_data(
                            X=X,
                            y_true=y_true,
                            positive_label="clust_A",
                            method=partition_method,
                            rep=rep,
                            n_components=5,
                        )

                    for partition_method in PARTITIONS:
                        for model_kind in MODELS:
                            for n_hidden in N_FACTORS:
                                local_scores = evaluate_local_from_partition(
                                    X=X,
                                    y_true=y_true,
                                    true_groups=true_groups,
                                    partition_result=partition_cache[partition_method],
                                    cluster_names=["clust_A", "clust_B"],
                                    model_kind=model_kind,
                                    n_hidden=n_hidden,
                                )
                                rows.append(
                                    {
                                        **setup,
                                        "partition_method": partition_method,
                                        "model_kind": model_kind,
                                        "n_hidden": n_hidden,
                                        **local_scores,
                                    }
                                )

                    elapsed = perf_counter() - experiment_start
                    total_elapsed = perf_counter() - start_time
                    avg_elapsed = total_elapsed / completed_experiments
                    remaining = total_experiments - completed_experiments
                    eta = remaining * avg_elapsed

                    _save_rows(rows, RESULTS_OUT)
                    print(
                        f"done nonlinear rep={rep} size={cluster_size} noise={n_noise} "
                        f"[{completed_experiments}/{total_experiments}] "
                        f"elapsed={_format_eta(elapsed)} eta={_format_eta(eta)}"
                    )

                except BaseException as exc:
                    failed_row = {
                        **setup,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    failed_rows.append(failed_row)
                    _save_rows(rows, RESULTS_OUT)
                    _save_rows(failed_rows, FAILED_OUT)
                    total_elapsed = perf_counter() - start_time
                    avg_elapsed = total_elapsed / max(completed_experiments, 1)
                    remaining = total_experiments - completed_experiments
                    eta = remaining * avg_elapsed
                    if isinstance(exc, KeyboardInterrupt):
                        print(
                            f"interrupted nonlinear rep={rep} size={cluster_size} noise={n_noise} "
                            f"[{completed_experiments}/{total_experiments}] "
                            f"eta={_format_eta(eta)}"
                        )
                        print("  KeyboardInterrupt: interrupted by user; saved partial results and exiting.")
                        return

                    print(
                        f"failed nonlinear rep={rep} size={cluster_size} noise={n_noise} "
                        f"[{completed_experiments}/{total_experiments}] "
                        f"eta={_format_eta(eta)}"
                    )
                    print(f"  {type(exc).__name__}: {exc}")

    _save_rows(rows, RESULTS_OUT)
    _save_rows(failed_rows, FAILED_OUT)

    print(f"saved {RESULTS_OUT}")
    print(f"saved {FAILED_OUT}")

    print("failed experiment setups:")
    if failed_rows:
        for failed_row in failed_rows:
            print(
                json.dumps(
                    {key: value for key, value in failed_row.items() if key != "traceback"},
                    sort_keys=True,
                )
            )
    else:
        print("[]")


if __name__ == "__main__":
    main()