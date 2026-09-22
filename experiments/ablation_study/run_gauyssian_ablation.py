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
from synthetic_gaussian import generate_gaussian_dataset


OUT_DIR = Path("./ablation_data_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_OUT = OUT_DIR / "gaussian_ablation.csv"
FAILED_OUT = OUT_DIR / "gaussian_ablation_failed_setups.csv"

REPS = 16
SAMPLE_SIZES = [50, 500, 5000]
ALPHAS = [0.0, 0.5, 1.0]
N_NOISE = [0, 25, 75]
N_FACTORS = [5]
PARTITIONS = ["oracle", "phate", "raw_kmeans", "random"]
MODELS = ["linear", "bio"]


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


def main():
    rows = []
    failed_rows = []

    total_experiments = (
        2 * len(range(REPS)) * len(SAMPLE_SIZES) * len(ALPHAS) * len(N_NOISE)
    )
    completed_experiments = 0
    start_time = perf_counter()

    for disjoint in [False, True]:
        for rep in range(REPS):
            for cluster_size in SAMPLE_SIZES:
                for alpha in ALPHAS:
                    for n_noise in N_NOISE:
                        completed_experiments += 1
                        experiment_start = perf_counter()
                        setup = {
                            "dataset": "gaussian",
                            "disjoint": disjoint,
                            "rep": rep,
                            "cluster_size": cluster_size,
                            "alpha": alpha,
                            "n_noise": n_noise,
                        }

                        try:
                            data = generate_gaussian_dataset(
                                rep=rep,
                                cluster_size=cluster_size,
                                alpha=alpha,
                                disjoint=disjoint,
                                n_noise=n_noise,
                            )

                            X = data["X"]
                            y_true = data["y_true"]
                            true_groups = data["true_groups"]
                            all_groups = true_groups["clust_1"] + true_groups["clust_2"]

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

                            # Partition once per partition method, then reuse for all model kinds and factor counts.
                            partition_cache = {}
                            for partition_method in PARTITIONS:
                                partition_cache[partition_method] = partition_data(
                                    X=X,
                                    y_true=y_true,
                                    positive_label="clust_1",
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
                                            cluster_names=["clust_1", "clust_2"],
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
                                f"done gaussian rep={rep} size={cluster_size} alpha={alpha} "
                                f"noise={n_noise} disjoint={disjoint} "
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
                                    f"interrupted gaussian rep={rep} size={cluster_size} alpha={alpha} "
                                    f"noise={n_noise} disjoint={disjoint} "
                                    f"[{completed_experiments}/{total_experiments}] "
                                    f"eta={_format_eta(eta)}"
                                )
                                print("  KeyboardInterrupt: interrupted by user; saved partial results and exiting.")
                                return

                            print(
                                f"failed gaussian rep={rep} size={cluster_size} alpha={alpha} "
                                f"noise={n_noise} disjoint={disjoint} "
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
                    {
                        key: value
                        for key, value in failed_row.items()
                        if key != "traceback"
                    },
                    sort_keys=True,
                )
            )
    else:
        print("[]")


if __name__ == "__main__":
    main()