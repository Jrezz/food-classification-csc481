"""
ResNet-50 CNN with configurable fine-tuning depth.
Supports two modes (Section 4.3 / Experiment 2):
  - 'head_only'   : Only the final FC layer is trained (all ResNet layers frozen).
  - 'two_blocks'  : The final FC layer + the last 2 residual blocks (layer3, layer4) are trained.
"""

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm


FINE_TUNE_MODES = ("head_only", "two_blocks")


def build_resnet50(num_classes: int = 101, fine_tune_mode: str = "two_blocks") -> nn.Module:
    """
    Constructs a ResNet-50 model pre-trained on ImageNet with a replaced head.

    Args:
        num_classes:     Number of output classes (101 for Food-101).
        fine_tune_mode:  'head_only' or 'two_blocks' (see module docstring).

    Returns:
        nn.Module ready for training.
    """
    if fine_tune_mode not in FINE_TUNE_MODES:
        raise ValueError(f"fine_tune_mode must be one of {FINE_TUNE_MODES}")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Replace the final FC layer
    in_features = model.fc.in_features  # 2048 for ResNet-50
    model.fc = nn.Linear(in_features, num_classes)

    # Freeze parameters based on fine-tuning strategy
    if fine_tune_mode == "head_only":
        # Freeze everything except the new FC head
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    elif fine_tune_mode == "two_blocks":
        # Freeze all layers up through layer2; unfreeze layer3, layer4, and fc
        freeze_until = {"conv1", "bn1", "layer1", "layer2"}
        for name, param in model.named_parameters():
            layer = name.split(".")[0]
            if layer in freeze_until:
                param.requires_grad = False

    return model


def get_optimizer(model: nn.Module, fine_tune_mode: str, lr: float = 1e-3) -> optim.Optimizer:
    """
    Returns an Adam optimizer.
    Uses a lower learning rate for unfrozen backbone layers vs. the new head,
    which is a standard practice for transfer learning.

    Args:
        model:          The ResNet-50 model.
        fine_tune_mode: Used to decide if differential LR is applied.
        lr:             Base learning rate (default 0.001 per proposal).
    """
    if fine_tune_mode == "head_only":
        # Only FC parameters have requires_grad=True
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # For 'two_blocks': lower LR for backbone, higher for head
    backbone_params = []
    head_params     = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return optim.Adam([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params,     "lr": lr},
    ])


class EarlyStopping:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter   = 0
        self.stop      = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def train_cnn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fine_tune_mode: str = "two_blocks",
    epochs: int = 20,
    lr: float = 1e-3,
    patience: int = 5,
    device: str | None = None,
    save_path: str | None = None,
) -> dict:
    """
    Full training loop for the CNN.

    Args:
        model:          ResNet-50 model (from build_resnet50).
        train_loader:   DataLoader for training split.
        val_loader:     DataLoader for validation split.
        fine_tune_mode: 'head_only' or 'two_blocks'.
        epochs:         Max training epochs (default 20 per proposal).
        lr:             Learning rate (default 0.001 per proposal).
        patience:       Early-stopping patience (in epochs with no val loss improvement).
        device:         'cuda', 'mps', or 'cpu'. Auto-detected if None.
        save_path:      If provided, the best model weights are saved here.

    Returns:
        history dict with keys: train_loss, train_acc, val_loss, val_acc (all lists).
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    print(f"Training on: {device}")

    model = model.to(device)
    criterion    = nn.CrossEntropyLoss()
    optimizer    = get_optimizer(model, fine_tune_mode, lr=lr)
    scheduler    = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    early_stop   = EarlyStopping(patience=patience)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_weights = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Training phase ──────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        correct      = 0
        total        = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds         = outputs.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += imgs.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # ── Validation phase ─────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        val_correct  = 0
        val_total    = 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs      = model(imgs)
                loss         = criterion(outputs, labels)
                val_loss_sum += loss.item() * imgs.size(0)
                preds         = outputs.argmax(dim=1)
                val_correct  += (preds == labels).sum().item()
                val_total    += imgs.size(0)

        val_loss = val_loss_sum / val_total
        val_acc  = val_correct  / val_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save(best_weights, save_path)
                print(f"  Best model saved to {save_path}")

        if early_stop(val_loss):
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_weights)
    return history


@torch.no_grad()
def evaluate_cnn(
    model: nn.Module,
    loader: DataLoader,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs inference on a DataLoader and returns predictions and ground-truth labels.

    Returns:
        all_probs:  (N, num_classes) softmax probability array
        all_preds:  (N,) predicted class indices
        all_labels: (N,) ground-truth class indices
    """
    import numpy as np

    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)

    model = model.to(device)
    model.eval()

    all_probs  = []
    all_preds  = []
    all_labels = []

    softmax = nn.Softmax(dim=1)

    for imgs, labels in tqdm(loader, desc="Evaluating CNN"):
        imgs   = imgs.to(device)
        logits = model(imgs)
        probs  = softmax(logits).cpu().numpy()
        preds  = probs.argmax(axis=1)

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return (
        np.vstack(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


def load_cnn(path: str, num_classes: int = 101, fine_tune_mode: str = "two_blocks") -> nn.Module:
    """Loads a saved ResNet-50 from disk."""
    model = build_resnet50(num_classes=num_classes, fine_tune_mode=fine_tune_mode)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model
