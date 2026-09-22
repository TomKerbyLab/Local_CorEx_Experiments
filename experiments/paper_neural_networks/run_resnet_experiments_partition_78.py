import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import random
import numpy as np
from datasets import load_dataset
from resnet_utils import (
    load_data_model,
    generate_data_representations,
    plot_random_cluster_samples,
    plot_random_class_samples,
    seed_worker,
    run_deepdream,
    ImageNetDataset
)

from local_corex import partition, partition_data, LinearCorex
from local_corex.nn_utils import compute_cluster_accuracies
from local_corex.nn_plotting import plot_perturved_accuracy_resnet, plot_logit_effects
import pickle
import torchvision.models as models
from deepdream import DeepDreamer, TargetSpec, dream_images_and_plot


def load_or_save_features(from_scratch, resnet18, val_loader, num_clusters, device):
    if from_scratch:
        print('computing data representations from scratch')
        
        inputs, features, preds, labels = generate_data_representations(resnet18, val_loader, feat_name = 'avgpool', device=device)
        print('saving data representations')
        with open(f'resnet_{num_clusters}_feature_inputs.pkl', 'wb') as f:
            pickle.dump(inputs, f)
        with open(f'resnet_{num_clusters}_feature_features.pkl', 'wb') as f:
            pickle.dump(features, f)
        with open(f'resnet_{num_clusters}_feature_preds.pkl', 'wb') as f:
            pickle.dump(preds, f)
        with open(f'resnet_{num_clusters}_feature_labels.pkl', 'wb') as f:
            pickle.dump(labels, f)
    else:
        print("loading data representations")
        with open(f'resnet_{num_clusters}_feature_inputs.pkl', 'rb') as f:
            inputs = pickle.load(f)
        with open(f'resnet_{num_clusters}_feature_features.pkl', 'rb') as f:
            features = pickle.load(f)
        with open(f'resnet_{num_clusters}_feature_preds.pkl', 'rb') as f:
            preds = pickle.load(f)
        with open(f'resnet_{num_clusters}_feature_labels.pkl', 'rb') as f:
            labels = pickle.load(f)

    if from_scratch:
        print('computing clusters from scratch')
        indexes = partition_data(features, n_partitions=num_clusters, phate_dim=10, n_jobs=-2, seed=42)
        print('saving clusters')
        with open(f'resnet_{num_clusters}_indexes.pkl', 'wb') as f:
            pickle.dump(indexes, f)
    else:
        print("loading clusters")
        with open(f'resnet_{num_clusters}_indexes.pkl', 'rb') as f:
            indexes = pickle.load(f)

    return inputs, features, preds, labels, indexes


def main():
    base_dir = '/yunity/tkerby2/Local_CorEx/Local_CorEx_Experiments/Figures/ResNet/cluster_78'
    partition=78
    num_clusters=100
    corex_factors = 20
    from_scratch = False
    class_names = models.ResNet18_Weights.DEFAULT.meta['categories']

    resnet18, val_loader, ds, device = load_data_model()
    inputs, features, preds, labels, indexes = load_or_save_features(from_scratch, resnet18, val_loader, num_clusters, device)
    x = np.concatenate([features[indexes[partition]], preds[indexes[partition]]], axis=1)

    corex = LinearCorex(corex_factors, seed=42)
    corex.fit(x)
    # print(corex.tcs)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def preprocess(x):
        return (x - mean.to(x.device)) / std.to(x.device)

    dreamer = DeepDreamer(
        resnet18,
        preprocess=preprocess,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    plot_random_cluster_samples(ds, indexes, partition, class_names, sample_count=12, seed=148, save_path= base_dir + f'/resnet_cluster_{partition}_samples.png')

    base_accuracies, base_probs = compute_cluster_accuracies(
        resnet18, val_loader, device, indexes, return_probs=True
    )

    print("Exploring factor num: 0")
    explore_factor_num = 0

    diff_probs = plot_perturved_accuracy_resnet(
        resnet18, 
        corex, 
        val_loader, 
        indexes,
        device, 
        base_accuracies=base_accuracies, 
        base_probs=base_probs, 
        factor_num=explore_factor_num, 
        num_drop=100,
        hidden_dim=512,
        save_path= base_dir + f'/resnet_cluster_{partition}_factor_{explore_factor_num}_perturbed_accuracies.png'
    )

    run_deepdream(
        corex, 
        dreamer,
        ds,
        steps=25,
        step_size=0.02,
        jitter=16,
        octaves=4,
        octave_scale=1.4,
        show_original_above=True,
        factor_num=explore_factor_num, 
        latent_dim=512, 
        node_num=100, 
        image_ids=[3, 10, 25, 40],
        save_path= base_dir + f'/resnet_cluster_{partition}_factor_{explore_factor_num}_deepdream.png'
    )

    plot_logit_effects(
        torch.mean(diff_probs[indexes[partition]], dim=0), 
        class_names, 
        bottom_vals=2,
        top_vals=8,
        save_path= base_dir + f'/resnet_cluster_{partition}_factor_{explore_factor_num}_{partition}_logit_effects.png'
    )

    plot_logit_effects(
        torch.mean(diff_probs[indexes[7]], dim=0), 
        class_names, 
        bottom_vals=5,
        top_vals=5,
        save_path= base_dir + f'/resnet_cluster_{partition}_factor_{explore_factor_num}_{7}_logit_effects.png'
    )

    plot_random_cluster_samples(ds, indexes, 7, class_names, sample_count=36, seed=148, save_path= base_dir + f'/resnet_cluster_{7}_samples.png')

    # plot_random_class_samples(
    #     ds, 
    #     class_label=83, 
    #     class_names=class_names, 
    #     sample_count=4, 
    #     seed=124,
    #     save_path= base_dir + f'/resnet_class_{83}_samples.png'
    # )

    # plot_random_class_samples(
    #     ds, 
    #     class_label=138, 
    #     class_names=class_names, 
    #     sample_count=4, 
    #     seed=124,
    #     save_path= base_dir + f'/resnet_class_{138}_samples.png'
    # )

    print("Exploring factor num: 1")
    explore_factor_num = 1

    diff_probs = plot_perturved_accuracy_resnet(
        resnet18, 
        corex, 
        val_loader, 
        indexes,
        device, 
        base_accuracies=base_accuracies, 
        base_probs=base_probs, 
        factor_num=1, 
        num_drop=100,
        hidden_dim=512,
        save_path= base_dir + f'/resnet_cluster_{partition}_factor_{explore_factor_num}_perturbed_accuracies.png'
    )

    plot_logit_effects(
        torch.mean(diff_probs[indexes[partition]], dim=0), 
        class_names, 
        bottom_vals=2,
        top_vals=8,
        save_path= base_dir + f'/resnet_cluster_{partition}_factor_{explore_factor_num}_{partition}_logit_effects.png'
    )

    plot_logit_effects(
        torch.mean(diff_probs[indexes[15]], dim=0), 
        class_names, 
        bottom_vals=5,
        top_vals=5,
        save_path= base_dir + f'/resnet_cluster_{partition}_factor_{explore_factor_num}_{15}_logit_effects.png'
    )

    plot_random_cluster_samples(ds, indexes, 15, class_names, sample_count=36, seed=148, save_path= base_dir + f'/resnet_cluster_{15}_samples.png')

    plot_logit_effects(
        torch.mean(diff_probs[~indexes[partition]], dim=0), 
        class_names, 
        bottom_vals=5, 
        top_vals=5,
        save_path= base_dir + f'/resnet_cluster_{partition}_factor_{explore_factor_num}_complement_of_{partition}_logit_effects.png'
    )

    run_deepdream(
        corex, 
        dreamer,
        ds,
        steps=25,
        step_size=0.02,
        jitter=16,
        octaves=4,
        octave_scale=1.4,
        show_original_above=True,
        factor_num=1, 
        latent_dim=512, 
        node_num=100, 
        image_ids=[3, 10, 25, 40],
        save_path= base_dir + f'/resnet_cluster_{partition}_factor_{explore_factor_num}_deepdream.png'
    )

if __name__ == "__main__":    
    main()