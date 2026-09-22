from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from local_corex import LinearCorex, partition_data
from mnist_factor_benchmark import (
    compute_factor_support,
    evaluate_model_probabilities,
    get_top_factor_indices,
    run_factor_ablation,
)
from local_corex.utils.nn_utils import (
    get_hidden_states,
    load_models,
)

from mnist_classifier.data import MNISTDataModule


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NUM_CLUSTERS = 20

TOP_K_FACTORS = 10

SUPPORT_LEVELS = (
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
)

N_HIDDEN_FACTORS = 30


def cluster_metadata(labels, idx):

    cluster_labels = labels[idx]

    counts = Counter(cluster_labels)

    majority_class = counts.most_common(1)[0][0]

    majority_fraction = (
        counts[majority_class]
        / len(cluster_labels)
    )

    probs = np.array(list(counts.values()))
    probs = probs / probs.sum()

    entropy = -(probs * np.log(probs)).sum()

    return {
        "cluster_size": len(cluster_labels),
        "majority_class": majority_class,
        "majority_fraction": majority_fraction,
        "cluster_entropy": entropy,
    }


def build_corex_input(
    hidden_rep,
    logits,
    cluster_idx,
):
    return np.concatenate(
        [
            hidden_rep[cluster_idx],
            logits[cluster_idx],
        ],
        axis=1,
    )


def extract_logit_mi(
    factor_mi,
    hidden_dim,
):

    logit_mi = factor_mi[hidden_dim:]

    order = np.argsort(np.abs(logit_mi))[::-1]

    result = {}

    for rank in range(min(3, len(order))):

        idx = int(order[rank])

        result[f"top_{rank+1}_logit_index"] = idx
        result[f"top_{rank+1}_logit_mi"] = float(
            logit_mi[idx]
        )

    return result


def main():

    output_dir = (
        PROJECT_ROOT
        / "benchmark_results"
        / "mnist_factor_benchmark"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_path = (
        PROJECT_ROOT
        / "experiments"
        / "paper_neural_networks"
        / "mnist_classifier"
        / "model_output"
        / "base_classifier"
        / "rep_1"
    )

    from mnist_classifier.model_output.do_classifier.rep_1.autoencoder_config import (
        conf as ae_conf,
    )

    from mnist_classifier.model_output.do_classifier.rep_1.config import (
        conf as clf_conf,
    )

    ae_conf = ae_conf["autoencoder"]
    clf_conf = clf_conf["classifier"]

    ae_ckpt = (
        base_path
        / "mnist_ae_epoch=091-val_loss=0.5937.ckpt"
    )

    clf_ckpt = (
        base_path
        / "mnist_clf_epoch=068-val_loss=0.0006.ckpt"
    )

    autoencoder, clf = load_models(
        ae_ckpt,
        clf_ckpt,
        ae_conf,
        clf_conf,
    )

    data_module = MNISTDataModule(
        clf_conf,
        "./mnist_classifier/",
    )

    data_module.prepare_data()
    data_module.setup("predict")

    model_data = get_hidden_states(
        clf,
        data_module,
        device=clf.device,
        num_layers=len(
            clf_conf["hidden_layers"]
        ),
    )

    logits = model_data[0]

    h1 = model_data[1]
    h2 = model_data[2]
    h3 = model_data[3]

    inputs = model_data[4]
    labels = model_data[5]

    hidden_states = [h1, h2, h3]

    hidden_dims = [500, 400, 300]

    try:
        with open("mnist_20_indexes.pkl", "rb") as f:
            indexes = pickle.load(f)
    except FileNotFoundError:
        indexes, _ = partition_data(
            inputs,
            n_partitions=NUM_CLUSTERS,
            phate_dim=10,
            n_jobs=-2,
            seed=42,
            return_pred=True,
        )

        with open("mnist_20_indexes.pkl", "wb") as f:
            pickle.dump(indexes, f)

    print("Computing baseline probabilities...")

    base_probs = evaluate_model_probabilities(clf, inputs)
    factor_metadata_rows = []
    ablation_rows = []

    for cluster_id in range(NUM_CLUSTERS):

        print(f"Cluster {cluster_id}")
        cluster_idx = indexes[cluster_id]
        cluster_stats = cluster_metadata(labels, cluster_idx)

        for layer_idx, (hidden_rep, hidden_dim) in enumerate(zip(hidden_states, hidden_dims)):
            print(f"  Layer H{layer_idx+1}")

            x = build_corex_input(
                hidden_rep,
                logits,
                cluster_idx,
            )

            corex_model = LinearCorex(n_hidden=N_HIDDEN_FACTORS, seed=42, gaussianize="outliers")
            corex_model.fit_transform(x)

            top_factors = get_top_factor_indices(corex_model, top_k=TOP_K_FACTORS)

            for factor_id in top_factors:
                factor_mi = (
                    corex_model.mis[
                        factor_id
                    ]
                )

                hidden_factor_mi = (
                    factor_mi[:hidden_dim]
                )

                supports = (
                    compute_factor_support(
                        hidden_factor_mi,
                        support_levels=SUPPORT_LEVELS,
                    )
                )

                factor_row = {
                    "cluster_id":
                        cluster_id,
                    "layer":
                        layer_idx + 1,
                    "factor_id":
                        int(factor_id),
                    "tc_explained":
                        float(
                            corex_model.tcs[
                                factor_id
                            ]
                        ),
                    **cluster_stats,
                }

                factor_row.update(
                    extract_logit_mi(
                        factor_mi,
                        hidden_dim,
                    )
                )

                for (
                    support_level,
                    support_info,
                ) in supports.items():

                    level_name = str(
                        int(
                            support_level
                            * 100
                        )
                    )

                    factor_row[
                        f"support_size_{level_name}"
                    ] = support_info[
                        "support_size"
                    ]

                    factor_row[
                        f"support_density_{level_name}"
                    ] = support_info[
                        "support_density"
                    ]

                    factor_row[
                        f"support_mi_{level_name}"
                    ] = support_info[
                        "cumulative_mi"
                    ]

                    factor_row[
                        f"support_nodes_{level_name}"
                    ] = json.dumps(
                        support_info[
                            "nodes"
                        ]
                    )

                factor_metadata_rows.append(
                    factor_row
                )

                for (
                    support_level,
                    support_info,
                ) in supports.items():

                    result = (
                        run_factor_ablation(
                            model=clf,
                            inputs=inputs,
                            labels=labels,
                            cluster_indices=cluster_idx,
                            hidden_layer_idx=layer_idx,
                            factor_id=factor_id,
                            support_level=support_level,
                            support_info=support_info,
                            base_probs=base_probs,
                        )
                    )

                    result[
                        "cluster_id"
                    ] = cluster_id

                    result[
                        "layer"
                    ] = layer_idx + 1

                    result[
                        "support_nodes"
                    ] = json.dumps(
                        support_info[
                            "nodes"
                        ]
                    )

                    ablation_rows.append(
                        result
                    )

    factor_df = pd.DataFrame(
        factor_metadata_rows
    )

    ablation_df = pd.DataFrame(
        ablation_rows
    )

    factor_df.to_csv(
        output_dir
        / "factor_metadata.csv",
        index=False,
    )

    ablation_df.to_csv(
        output_dir
        / "factor_ablations.csv",
        index=False,
    )

    print(
        "Saved:"
    )

    print(
        output_dir
        / "factor_metadata.csv"
    )

    print(
        output_dir
        / "factor_ablations.csv"
    )


if __name__ == "__main__":
    main()