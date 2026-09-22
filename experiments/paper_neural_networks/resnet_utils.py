import torch
import torchvision.models as models
from torchvision import transforms, models
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import random
from deepdream import TargetSpec, dream_images_and_plot
from tqdm.auto import tqdm

def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32  # derive a seed unique per worker
        np.random.seed(worker_seed)
        random.seed(worker_seed)

class ImageNetDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = item['label']
        return image, label

def load_data_model():
    
    ds = load_dataset('imagenet-1k')
   
    # Create a generator with a fixed seed
    g = torch.Generator()
    g.manual_seed(42)

    test_transforms = transforms.Compose([
        transforms.Resize(256),          # Resize the shorter side to 256 pixels
        transforms.CenterCrop(224),      # Crop the center 224x224 patch
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    imagenet_val = ImageNetDataset(ds['validation'], transform=test_transforms)
    val_loader = DataLoader(imagenet_val, batch_size=64, shuffle=False, num_workers=0, worker_init_fn=seed_worker, generator=g)

    resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet18 = resnet18.to(device)
    return resnet18, val_loader, ds, device


def generate_data_representations(model, dl, feat_name = 'avgpool', device='cpu'):
    # Define hook function
    intermediate_features = {}
    def get_features(name):
        def hook(model, input, output):
            intermediate_features[name] = output.detach()
        return hook

    hook_handle = model.avgpool.register_forward_hook(get_features(feat_name))

    all_inputs = []
    all_features = []
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dl)):
            images = images.to(device).float()
            preds = model(images)
            features = intermediate_features[feat_name].squeeze()
            all_inputs.append(images.cpu())
            all_features.append(features.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels)

    inputs_tensor = torch.cat(all_inputs, dim=0)
    features_tensor = torch.cat(all_features, dim=0)
    preds_tensor = torch.cat(all_preds, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    hook_handle.remove()
    return inputs_tensor, features_tensor, preds_tensor, labels_tensor

def plot_random_cluster_samples(ds, indexes, partition, class_names, sample_count=8, seed=None, fontsize=16, save_path=None):
    """
    Plots randomly selected images from the cluster defined by indexes[partition] in a grid.
    
    Parameters:
        ds (dict): Dataset containing the images and labels (e.g., ds['validation'][i]['image'] and ['label']).
        indexes (dict or array-like): Data structure where indexes[partition] is a boolean array
                                      indicating membership of each sample in the cluster.
        partition: Key/index for the desired cluster in the indexes structure.
        class_names (list or dict): Mapping from label indices to class names.
        sample_count (int): Number of random images to select and plot. Defaults to 8.
        seed (int, optional): Seed for random number generator (for reproducibility).
    """
    
    # Optionally set random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
    
    # Get the indices of images in the desired cluster
    cluster_indices = np.where(indexes[partition] == True)[0]
    if len(cluster_indices) == 0:
        print("No images found in this cluster!")
        return
    
    # Adjust sample_count if the cluster has fewer than sample_count images
    actual_sample_count = min(sample_count, len(cluster_indices))
    
    # Randomly select the desired number of indices from the cluster
    selected_indices = np.random.choice(cluster_indices, size=actual_sample_count, replace=False)
    
    # Determine grid size: here we use 4 columns
    cols = 4
    rows = int(np.ceil(actual_sample_count / cols))
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    # In case there's only one row, wrap axes in a list
    if rows == 1:
        axes = np.array(axes).reshape(-1)
    else:
        axes = axes.flatten()
    
    # Plot each randomly selected image with its label
    for ax, idx in zip(axes, selected_indices):
        idx = int(idx)
        image = ds['validation'][idx]['image']
        label = ds['validation'][idx]['label']
        title_str = f"{class_names[label]}"
        ax.imshow(image)
        ax.set_title(title_str, fontsize=fontsize)
        ax.axis("off")
    
    # Turn off any extra axes (if any)
    for ax in axes[actual_sample_count:]:
        ax.axis("off")
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

def plot_random_class_samples(ds, class_label, class_names, sample_count=8, seed=None, fontsize=16, split='validation', save_path=None):
    """
    Plots randomly selected images from a single class in a grid.

    Parameters:
        ds (dict): Dataset containing images and labels, e.g. ds['validation'][i]['image'].
        class_label (int): Integer class id to sample from.
        class_names (list or dict): Mapping from label indices to class names.
        sample_count (int): Number of random images to select and plot. Defaults to 8.
        seed (int, optional): Seed for random number generator (for reproducibility).
        fontsize (int): Font size for image titles.
        split (str): Dataset split key, default is 'validation'.
    """

    if seed is not None:
        np.random.seed(seed)

    labels = np.array(ds[split]['label'])
    class_indices = np.where(labels == class_label)[0]

    if len(class_indices) == 0:
        print(f"No images found for class {class_label} ({class_names[class_label]})")
        return

    actual_sample_count = min(sample_count, len(class_indices))
    selected_indices = np.random.choice(class_indices, size=actual_sample_count, replace=False)

    cols = 4
    rows = int(np.ceil(actual_sample_count / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = np.array(axes).reshape(-1)
    else:
        axes = axes.flatten()

    for ax, idx in zip(axes, selected_indices):
        idx = int(idx)
        image = ds[split][idx]['image']
        label = ds[split][idx]['label']
        ax.imshow(image)
        ax.set_title(f"{class_names[label]}", fontsize=fontsize)
        ax.axis("off")

    for ax in axes[actual_sample_count:]:
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()



def run_deepdream(
        corex_model, 
        dreamer,
        ds,
        steps=25,
        step_size=0.02,
        jitter=16,
        octaves=4,
        octave_scale=1.4,
        show_original_above=True,
        factor_num=0, 
        latent_dim=512, 
        node_num=100, 
        image_ids=[3, 10, 25, 40],
        save_path=None,
    ):

    top_k_ind = np.argsort(corex_model.mis[factor_num][:latent_dim])[::-1][:node_num].astype(np.int64).copy()
    top_k_ind = top_k_ind.tolist()
    top_k_weights = corex_model.mis[factor_num][:latent_dim][top_k_ind].tolist()

    specs = [
        TargetSpec(
            layer="avgpool",
            weight=1.0,
            channels=top_k_ind,
            node_weights=top_k_weights,
            objective="l2",
        )
    ]

    images_to_dream = [
        ds['validation'][image_id]['image'] for image_id in image_ids
    ]

    dreamed_images = dream_images_and_plot(
        images=images_to_dream,
        dreamer=dreamer,
        specs=specs,
        show_original_above=show_original_above,
        steps=steps,
        step_size=0.02,
        jitter=jitter,
        octaves=octaves,
        octave_scale=octave_scale
        ,save_path=save_path
    )