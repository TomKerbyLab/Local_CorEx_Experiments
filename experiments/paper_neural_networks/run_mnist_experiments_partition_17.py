import pickle
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from deepdream import DeepDreamer, TargetSpec, dream_images_and_plot
from local_corex import LinearCorex, partition_data
from local_corex.utils.nn_plotting import plot_logit_effects, plot_perturved_accuracy
from local_corex.utils.nn_utils import compute_cluster_accuracies, get_hidden_states, load_models
from local_corex.utils.plotting import hidden_state_plot, multi_rep_plot
from mnist_classifier.data import MNISTDataModule

PARTITION = 17
NUM_CLUSTERS = 20


def _get_encoder_stage_modules(encoder):
    modules = list(encoder.children())
    stages = []
    current_stage = []

    for module in modules:
        if isinstance(module, torch.nn.Linear) and len(current_stage) > 0:
            stages.append(torch.nn.Sequential(*current_stage))
            current_stage = [module]
        else:
            current_stage.append(module)

    if len(current_stage) > 0:
        stages.append(torch.nn.Sequential(*current_stage))

    return stages


def arr_to_pil(arr):
    arr = arr * 0.3081 + 0.1307
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def plot_guided_reconstructions(
    corex_model,
    do_ae,
    do_state_df,
    neuron_indexes,
    sample_positions,
    indexes,
    corex_factor=0,
    final_scale=5,
    scale_step=2.5,
    cols=4,
    partition_idx=None,
    perturbation=None,
    encoder_layer=None,
    figsize_scale=3,
    add=True,
    save_path=None,
):
    if partition_idx is None:
        partition_idx = PARTITION

    if final_scale <= 0:
        raise ValueError("final_scale must be positive")
    if scale_step <= 0:
        raise ValueError("scale_step must be positive")

    scale_values = np.arange(scale_step, final_scale + scale_step, scale_step)
    scale_values = scale_values[scale_values <= final_scale + 1e-8]
    if len(scale_values) == 0 or not np.isclose(scale_values[-1], final_scale):
        scale_values = np.append(scale_values, final_scale)
    scale_values = scale_values.tolist()

    device = next(do_ae.parameters()).device
    encoder_stages = _get_encoder_stage_modules(do_ae.encoder)
    n_stages = len(encoder_stages)
    if n_stages == 0:
        raise ValueError("Autoencoder encoder has no stages to perturb")

    if encoder_layer is None:
        encoder_layer = n_stages - 1
    if encoder_layer < 0 or encoder_layer >= n_stages:
        raise ValueError(f"encoder_layer must be in [0, {n_stages - 1}], got {encoder_layer}")

    if perturbation is None:
        perturbation = corex_model.mis[corex_factor][:neuron_indexes]

    perturbation_tensor = torch.as_tensor(
        perturbation,
        dtype=torch.float32,
        device=device,
    ).flatten().unsqueeze(0)

    original_images = []
    reconstructed_images = []
    perturbed_images_by_scale = {scale: [] for scale in scale_values}

    do_ae.eval()
    with torch.no_grad():
        for pos in sample_positions:
            input_arr = do_state_df["input"][indexes[partition_idx]].iloc[pos]
            original_images.append(arr_to_pil(input_arr.reshape(28, 28)))

            x = torch.tensor(input_arr, dtype=torch.float32, device=device).unsqueeze(0)

            h = x
            for stage_idx in range(encoder_layer + 1):
                h = encoder_stages[stage_idx](h)

            stage_dim = h.shape[-1]
            if perturbation_tensor.shape[-1] != stage_dim:
                raise ValueError(
                    f"Perturbation length ({perturbation_tensor.shape[-1]}) does not match "
                    f"encoder layer {encoder_layer} width ({stage_dim})."
                )

            z_recon = h
            for stage_idx in range(encoder_layer + 1, n_stages):
                z_recon = encoder_stages[stage_idx](z_recon)
            x_recon = do_ae.decoder(z_recon).squeeze(0).detach().cpu().numpy()
            reconstructed_images.append(arr_to_pil(x_recon.reshape(28, 28)))

            for scale in scale_values:
                if add:
                    h_guided = h + scale * perturbation_tensor
                else:
                    h_guided = h - scale * perturbation_tensor

                z_guided = h_guided
                for stage_idx in range(encoder_layer + 1, n_stages):
                    z_guided = encoder_stages[stage_idx](z_guided)

                x_guided = do_ae.decoder(z_guided).squeeze(0).detach().cpu().numpy()
                perturbed_images_by_scale[scale].append(arr_to_pil(x_guided.reshape(28, 28)))

    n_images = len(sample_positions)
    n_cols = min(cols, n_images)
    n_row_groups = int(np.ceil(n_images / n_cols))
    rows_per_group = 2 + len(scale_values)

    fig, axes = plt.subplots(
        n_row_groups * rows_per_group,
        n_cols,
        figsize=(figsize_scale * n_cols, figsize_scale * rows_per_group * n_row_groups),
    )
    axes = np.array(axes, dtype=object).reshape(n_row_groups * rows_per_group, n_cols)

    for idx, orig_img in enumerate(original_images):
        row_group = idx // n_cols
        col = idx % n_cols
        base_row = row_group * rows_per_group

        ax_orig = axes[base_row, col]
        ax_orig.imshow(orig_img, cmap="gray")
        ax_orig.set_title(f"Original {sample_positions[idx]}")
        ax_orig.axis("off")

        ax_recon = axes[base_row + 1, col]
        ax_recon.imshow(reconstructed_images[idx], cmap="gray")
        ax_recon.set_title("Reconstruction")
        ax_recon.axis("off")

        for scale_idx, scale in enumerate(scale_values, start=2):
            ax_pert = axes[base_row + scale_idx, col]
            ax_pert.imshow(perturbed_images_by_scale[scale][idx], cmap="gray")
            ax_pert.set_title(f"Scale {scale:g}")
            ax_pert.axis("off")

    for idx in range(n_images, n_row_groups * n_cols):
        row_group = idx // n_cols
        col = idx % n_cols
        base_row = row_group * rows_per_group
        for row_offset in range(rows_per_group):
            axes[base_row + row_offset, col].axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, axes, original_images, reconstructed_images, perturbed_images_by_scale, scale_values


def main():
    base_path = PROJECT_ROOT / "experiments" / "paper_neural_networks" / "mnist_classifier" / "model_output" / "base_classifier" / "rep_1"
    output_dir = PROJECT_ROOT / "Figures" / "MNIST" / "cluster_17"
    output_dir.mkdir(parents=True, exist_ok=True)

    from mnist_classifier.model_output.do_classifier.rep_1.autoencoder_config import conf as ae_conf
    from mnist_classifier.model_output.do_classifier.rep_1.config import conf as clf_conf

    ae_conf = ae_conf["autoencoder"]
    ae_ckpt = base_path / "mnist_ae_epoch=091-val_loss=0.5937.ckpt"
    clf_conf = clf_conf["classifier"]
    clf_ckpt = base_path / "mnist_clf_epoch=068-val_loss=0.0006.ckpt"

    do_ae, do_clf = load_models(ae_ckpt, clf_ckpt, ae_conf, clf_conf)

    data_module = MNISTDataModule(clf_conf, "./mnist_classifier/")
    data_module.prepare_data()
    data_module.setup("predict")

    model_data = get_hidden_states(
        do_clf,
        data_module,
        device=do_clf.device,
        num_layers=len(clf_conf["hidden_layers"]),
    )

    inputs = model_data[4]
    labels = model_data[5]

    do_state_df = pd.DataFrame(
        {
            "input": list(inputs),
            "h1": list(model_data[1]),
            "h2": list(model_data[2]),
            "h3": list(model_data[3]),
            "output": list(model_data[0]),
        }
    )

    from_scratch = False

    if from_scratch:
        print("computing clusters from scratch")
        indexes, pred = partition_data(
            inputs,
            n_partitions=NUM_CLUSTERS,
            phate_dim=10,
            n_jobs=-2,
            seed=42,
            return_pred=True,
        )
        print("saving clusters")
        with open("mnist_20_indexes.pkl", "wb") as f:
            pickle.dump(indexes, f)
    else:
        print("loading clusters")
        with open("mnist_20_indexes.pkl", "rb") as f:
            indexes = pickle.load(f)

    for i in range(NUM_CLUSTERS):
        print("Group number:", i, Counter(labels[indexes[i]]))

    fig, axes = plt.subplots(2, 10, figsize=(20, 5))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(np.mean(inputs[indexes[i]], axis=0).reshape(28, 28))
        ax.set_title("Frame " + str(i))
    fig.savefig(output_dir / "cluster_overview.png", bbox_inches="tight")

    partition = PARTITION
    x = np.concatenate([model_data[3][indexes[partition]], model_data[0][indexes[partition]]], axis=1)

    h3_to_out = LinearCorex(n_hidden=30, seed=42, gaussianize="outliers", discourage_overlap=False)
    h3_to_out.fit_transform(x)
    print(h3_to_out.tcs)
    multi_rep_plot(h3_to_out, 3, dims=[(15, 20), (10, 1)], num_per_row=2, save_path=output_dir / f"h3_to_out_{partition}_multi_rep.png")
    hidden_state_plot(x, h3_to_out, do_ae, factors=list(range(3)), latent_dim=ae_conf["decoder_layers"][0], encoder_layer=3, scaler=1, save_path=output_dir / f"h3_to_out_{partition}_hidden_state.png")

    h2_to_out = LinearCorex(n_hidden=30, seed=42, gaussianize="outliers")
    x = np.concatenate([model_data[2][indexes[partition]], model_data[0][indexes[partition]]], axis=1)
    h2_to_out.fit_transform(x)
    print(h2_to_out.tcs)
    multi_rep_plot(h2_to_out, 3, dims=[(20, 20), (10, 1)], num_per_row=2, save_path=output_dir / f"h2_to_out_{partition}_multi_rep.png")
    hidden_state_plot(x, h2_to_out, do_ae, factors=list(range(3)), latent_dim=400, encoder_layer=2, scaler=1, save_path=output_dir / f"h2_to_out_{partition}_hidden_state.png")

    h1_to_out = LinearCorex(n_hidden=30, seed=42, gaussianize="outliers")
    x = np.concatenate([model_data[1][indexes[partition]], model_data[0][indexes[partition]]], axis=1)
    h1_to_out.fit_transform(x)
    print(h1_to_out.tcs)
    multi_rep_plot(h1_to_out, 3, dims=[(20, 25), (10, 1)], num_per_row=2, save_path=output_dir / f"h1_to_out_{partition}_multi_rep.png")
    hidden_state_plot(x, h1_to_out, do_ae, scaler=1, latent_dim=500, encoder_layer=1, save_path=output_dir / f"h1_to_out_{partition}_hidden_state.png")

    do_clf.to("cpu")
    pred_digit = do_clf(torch.tensor(inputs)).max(1).indices.detach().numpy()
    base_accuracies = []
    for i in range(NUM_CLUSTERS):
        base_accuracies.append(100 * np.mean(pred_digit[indexes[i]] == labels[indexes[i]]))
        print(i, np.round(np.mean(pred_digit[indexes[i]] == labels[indexes[i]]) * 100, 2))

    base_accuracies, base_probs = compute_cluster_accuracies(
        do_clf,
        data_module.predict_dataloader(),
        "cpu",
        indexes,
        return_probs=True,
    )

    diff_probs = plot_perturved_accuracy(
        do_clf,
        h1_to_out,
        inputs,
        labels,
        indexes,
        base_probs=base_probs,
        factor_num=0,
        hidden_layer_idx=0,
        num_clusters=20,
        num_drop=100,
        hidden_dim=500,
        save_path=output_dir / f"h1_to_out_{partition}_factor_0_perturbed_accuracy.png",
    )

    class_names = [str(i) for i in range(10)]
    plot_logit_effects(torch.mean(diff_probs, dim=0), class_names, bottom_vals=2, top_vals=2, save_path=output_dir / f"h1_to_out_{partition}_factor_0_logit_effects_all.png")
    plot_logit_effects(torch.mean(diff_probs[indexes[partition]], dim=0), class_names, bottom_vals=2, top_vals=2, save_path=output_dir / f"h1_to_out_{partition}_factor_0_logit_effects_partition.png")

    diff_probs = plot_perturved_accuracy(
        do_clf,
        h1_to_out,
        inputs,
        labels,
        indexes,
        base_probs=base_probs,
        factor_num=1,
        hidden_layer_idx=0,
        num_clusters=20,
        num_drop=155,
        hidden_dim=500,
        save_path=output_dir / f"h1_to_out_{partition}_factor_1_perturbed_accuracy.png",
    )
    plot_logit_effects(torch.mean(diff_probs, dim=0), class_names, bottom_vals=2, top_vals=2, save_path=output_dir / f"h1_to_out_{partition}_factor_1_logit_effects_all.png")
    plot_logit_effects(torch.mean(diff_probs[indexes[partition]], dim=0), class_names, bottom_vals=2, top_vals=2, save_path=output_dir / f"h1_to_out_{partition}_factor_1_logit_effects_partition.png")

    plt.figure()
    output = plt.hist(h1_to_out.moments["MI"][2], bins=50)
    print(np.sum(output[0][1:]))
    plt.savefig(output_dir / f"h1_to_out_{partition}_mi_hist.png", bbox_inches="tight")

    diff_probs = plot_perturved_accuracy(
        do_clf,
        h2_to_out,
        inputs,
        labels,
        indexes,
        base_probs=base_probs,
        factor_num=0,
        hidden_layer_idx=1,
        num_clusters=20,
        num_drop=140,
        hidden_dim=400,
        save_path=output_dir / f"h2_to_out_{partition}_factor_0_perturbed_accuracy.png",
    )
    plot_logit_effects(torch.mean(diff_probs, dim=0), class_names, bottom_vals=2, top_vals=2, save_path=output_dir / f"h2_to_out_{partition}_factor_0_logit_effects_all.png")
    plot_logit_effects(torch.mean(diff_probs[indexes[partition]], dim=0), class_names, bottom_vals=2, top_vals=2, save_path=output_dir / f"h2_to_out_{partition}_factor_0_logit_effects_partition.png")

    diff_probs = plot_perturved_accuracy(
        do_clf,
        h2_to_out,
        inputs,
        labels,
        indexes,
        base_probs=base_probs,
        factor_num=2,
        hidden_layer_idx=1,
        num_clusters=20,
        num_drop=200,
        hidden_dim=400,
        save_path=output_dir / f"h2_to_out_{partition}_factor_2_perturbed_accuracy.png",
    )
    plot_logit_effects(torch.mean(diff_probs, dim=0), class_names, bottom_vals=2, top_vals=2, save_path=output_dir / f"h2_to_out_{partition}_factor_2_logit_effects_all.png")
    plot_logit_effects(torch.mean(diff_probs[indexes[partition]], dim=0), class_names, bottom_vals=2, top_vals=2, save_path=output_dir / f"h2_to_out_{partition}_factor_2_logit_effects_partition.png")

    diff_probs = plot_perturved_accuracy(
        do_clf,
        h2_to_out,
        inputs,
        labels,
        indexes,
        base_probs=base_probs,
        factor_num=3,
        hidden_layer_idx=1,
        num_clusters=20,
        num_drop=200,
        hidden_dim=400,
        save_path=output_dir / f"h2_to_out_{partition}_factor_3_perturbed_accuracy.png",
    )
    plot_logit_effects(torch.mean(diff_probs, dim=0), class_names, bottom_vals=2, top_vals=2, save_path=output_dir / f"h2_to_out_{partition}_factor_3_logit_effects_all.png")
    plot_logit_effects(torch.mean(diff_probs[indexes[partition]], dim=0), class_names, bottom_vals=2, top_vals=2, save_path=output_dir / f"h2_to_out_{partition}_factor_3_logit_effects_partition.png")

    plt.figure()
    output = plt.hist(h2_to_out.moments["MI"][0], bins=50)
    print(np.sum(output[0][1:]))
    plt.savefig(output_dir / f"h2_to_out_{partition}_mi_hist.png", bbox_inches="tight")

    diff_probs = plot_perturved_accuracy(
        do_clf,
        h3_to_out,
        inputs,
        labels,
        indexes,
        base_probs=base_probs,
        factor_num=0,
        hidden_layer_idx=2,
        num_clusters=20,
        num_drop=160,
        hidden_dim=300,
        save_path=output_dir / f"h3_to_out_{partition}_factor_0_perturbed_accuracy.png",
    )
    plot_logit_effects(torch.mean(diff_probs, dim=0), class_names, bottom_vals=3, top_vals=3, save_path=output_dir / f"h3_to_out_{partition}_factor_0_logit_effects_all.png")
    plot_logit_effects(torch.mean(diff_probs[indexes[partition]], dim=0), class_names, bottom_vals=2, top_vals=2, save_path=output_dir / f"h3_to_out_{partition}_factor_0_logit_effects_partition.png")

    diff_probs = plot_perturved_accuracy(
        do_clf,
        h3_to_out,
        inputs,
        labels,
        indexes,
        base_probs=base_probs,
        factor_num=1,
        hidden_layer_idx=2,
        num_clusters=20,
        num_drop=130,
        hidden_dim=300,
        save_path=output_dir / f"h3_to_out_{partition}_factor_1_perturbed_accuracy.png",
    )
    plot_logit_effects(torch.mean(diff_probs, dim=0), class_names, bottom_vals=3, top_vals=3, save_path=output_dir / f"h3_to_out_{partition}_factor_1_logit_effects_all.png")
    plot_logit_effects(torch.mean(diff_probs[indexes[partition]], dim=0), class_names, bottom_vals=2, top_vals=2, save_path=output_dir / f"h3_to_out_{partition}_factor_1_logit_effects_partition.png")

    dreamer = DeepDreamer(
        do_clf,
        device="cpu",
    )

    factor_num = 1
    top_k_ind = np.argsort(h3_to_out.mis[factor_num][:300])[::-1][:100].astype(np.int64).copy()
    top_k_ind = top_k_ind.tolist()
    top_k_weights = h3_to_out.mis[factor_num][:300][top_k_ind].tolist()

    specs = [
        TargetSpec(
            layer="hidden_layers.2",
            weight=1.0,
            channels=top_k_ind,
            node_weights=top_k_weights,
            objective="relu_l2",
        )
    ]

    images_to_dream = [
        arr_to_pil(do_state_df["input"][indexes[partition]].iloc[0].reshape(28, 28)),
        arr_to_pil(do_state_df["input"][indexes[partition]].iloc[10].reshape(28, 28)),
        arr_to_pil(do_state_df["input"][indexes[partition]].iloc[25].reshape(28, 28)),
        arr_to_pil(do_state_df["input"][indexes[partition]].iloc[40].reshape(28, 28)),
    ]

    dream_images_and_plot(
        images=images_to_dream,
        dreamer=dreamer,
        specs=specs,
        show_original_above=True,
        steps=40,
        step_size=0.005,
        jitter=0,
        octaves=0,
        octave_scale=1.4,
        resize=None,
        grayscale=True,
        save_path=output_dir / f"h3_to_out_{partition}_deepdream_samples.png",
    )

    images_to_dream = [
        arr_to_pil(np.random.rand(784).reshape(28, 28)),
        arr_to_pil(np.random.rand(784).reshape(28, 28)),
        arr_to_pil(np.random.rand(784).reshape(28, 28)),
        arr_to_pil(np.random.rand(784).reshape(28, 28)),
    ]

    dream_images_and_plot(
        images=images_to_dream,
        dreamer=dreamer,
        specs=specs,
        show_original_above=True,
        steps=40,
        step_size=0.005,
        jitter=0,
        octaves=0,
        octave_scale=1.4,
        resize=None,
        grayscale=True,
        save_path=output_dir / f"h3_to_out_{partition}_deepdream_random.png",
    )

    plot_guided_reconstructions(
        corex_model=h3_to_out,
        do_ae=do_ae,
        do_state_df=do_state_df,
        indexes=indexes,
        corex_factor=0,
        neuron_indexes=300,
        sample_positions=[0, 3, 10, 25, 40, 55, 65, 89],
        encoder_layer=2,
        final_scale=3.5,
        scale_step=1.25,
        cols=8,
        add=True,
        save_path=output_dir / f"h3_to_out_{partition}_guided_reconstructions.png",
    )

    plot_guided_reconstructions(
        corex_model=h2_to_out,
        do_ae=do_ae,
        do_state_df=do_state_df,
        encoder_layer=1,
        indexes=indexes,
        corex_factor=0,
        neuron_indexes=400,
        sample_positions=[0, 3, 10, 25, 40, 55, 65, 89],
        final_scale=3.5,
        scale_step=1.25,
        cols=8,
        add=True,
        save_path=output_dir / f"h2_to_out_{partition}_guided_reconstructions.png",
    )

    plot_guided_reconstructions(
        corex_model=h1_to_out,
        do_ae=do_ae,
        do_state_df=do_state_df,
        indexes=indexes,
        encoder_layer=0,
        corex_factor=0,
        neuron_indexes=500,
        sample_positions=[0, 3, 10, 25, 40, 55, 65, 89],
        final_scale=3.5,
        scale_step=1.25,
        cols=8,
        add=True,
        save_path=output_dir / f"h1_to_out_{partition}_guided_reconstructions.png",
    )


if __name__ == "__main__":
    main()