"""
Food-101 dataset loading, preprocessing, and augmentation pipeline.
Uses torchvision's built-in Food101 dataset which handles downloading automatically.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import Food101


# ImageNet normalization statistics (used for ResNet-50 pre-trained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224


def get_train_transform(augment: bool = True) -> transforms.Compose:
    """
    Returns the transform pipeline for training images.
    Applies data augmentation when augment=True (Section 4.1 of proposal).
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return get_eval_transform()


def get_eval_transform() -> transforms.Compose:
    """Returns the transform pipeline for validation/test images (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_raw_transform() -> transforms.Compose:
    """
    Returns a transform that resizes and converts to tensor without ImageNet normalization.
    Used for HOG/color histogram feature extraction (SVM/RF pipelines).
    Pixel values in [0, 1] as per Section 4.1.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),  # scales to [0, 1]
    ])


def load_food101(
    data_root: str = "./data",
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
    val_fraction: float = 0.10,
    subset_fraction: float = 1.0,
    seed: int = 42,
):
    """
    Loads the Food-101 dataset. Downloads automatically on first run (~4.6 GB).

    Args:
        data_root:        Directory to store/load the dataset.
        batch_size:       DataLoader batch size.
        num_workers:      Number of parallel data loading workers.
        augment:          Whether to apply training augmentations.
        val_fraction:     Fraction of training data to use as validation set (default 10%).
        subset_fraction:  Fraction of each split to use (1.0 = full dataset).
                          Useful for quick smoke-testing.
        seed:             Random seed for reproducibility.

    Returns:
        dict with keys: train_loader, val_loader, test_loader, class_names,
                        num_classes, train_dataset, val_dataset, test_dataset
    """
    os.makedirs(data_root, exist_ok=True)

    # Download / load the official train and test splits
    train_full = Food101(root=data_root, split="train", transform=get_train_transform(augment), download=True)
    test_ds    = Food101(root=data_root, split="test",  transform=get_eval_transform(),          download=True)

    # Carve out a validation set from the training split
    rng = np.random.default_rng(seed)
    n_train_full = len(train_full)
    all_idx = np.arange(n_train_full)
    rng.shuffle(all_idx)

    n_val   = int(n_train_full * val_fraction)
    val_idx   = all_idx[:n_val]
    train_idx = all_idx[n_val:]

    # Optional subset (for quick experiments)
    if subset_fraction < 1.0:
        train_idx = train_idx[:int(len(train_idx) * subset_fraction)]
        val_idx   = val_idx[:int(len(val_idx)   * subset_fraction)]
        test_idx  = rng.choice(len(test_ds), int(len(test_ds) * subset_fraction), replace=False)
    else:
        test_idx = np.arange(len(test_ds))

    # Build validation dataset with no augmentation
    val_full = Food101(root=data_root, split="train", transform=get_eval_transform(), download=False)

    train_ds = Subset(train_full, train_idx)
    val_ds   = Subset(val_full,   val_idx)
    test_ds  = Subset(test_ds,    test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    class_names = train_full.classes

    return {
        "train_loader":   train_loader,
        "val_loader":     val_loader,
        "test_loader":    test_loader,
        "class_names":    class_names,
        "num_classes":    len(class_names),
        "train_dataset":  train_ds,
        "val_dataset":    val_ds,
        "test_dataset":   test_ds,
    }


def load_food101_raw(
    data_root: str = "./data",
    batch_size: int = 64,
    num_workers: int = 4,
    val_fraction: float = 0.10,
    subset_fraction: float = 1.0,
    seed: int = 42,
):
    """
    Loads Food-101 with raw pixel transforms [0,1] (no ImageNet normalization).
    Used for extracting HOG and color histogram features for SVM and Random Forest.

    Returns same dict structure as load_food101().
    """
    os.makedirs(data_root, exist_ok=True)

    raw_transform = get_raw_transform()

    train_full = Food101(root=data_root, split="train", transform=raw_transform, download=True)
    test_ds    = Food101(root=data_root, split="test",  transform=raw_transform, download=True)

    rng = np.random.default_rng(seed)
    n_train_full = len(train_full)
    all_idx = np.arange(n_train_full)
    rng.shuffle(all_idx)

    n_val     = int(n_train_full * val_fraction)
    val_idx   = all_idx[:n_val]
    train_idx = all_idx[n_val:]

    if subset_fraction < 1.0:
        train_idx = train_idx[:int(len(train_idx) * subset_fraction)]
        val_idx   = val_idx[:int(len(val_idx)   * subset_fraction)]
        test_idx  = rng.choice(len(test_ds), int(len(test_ds) * subset_fraction), replace=False)
    else:
        test_idx = np.arange(len(test_ds))

    train_ds = Subset(train_full, train_idx)
    val_ds   = Subset(train_full, val_idx)
    test_ds  = Subset(test_ds,   test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    class_names = train_full.classes

    return {
        "train_loader":   train_loader,
        "val_loader":     val_loader,
        "test_loader":    test_loader,
        "class_names":    class_names,
        "num_classes":    len(class_names),
        "train_dataset":  train_ds,
        "val_dataset":    val_ds,
        "test_dataset":   test_ds,
    }
