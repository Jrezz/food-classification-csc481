"""
Experiment 2 — Transfer Learning Depth
========================================
Compares two ResNet-50 fine-tuning strategies:
  1. head_only   — only the final FC layer is trained
  2. two_blocks  — the final FC layer + last 2 residual blocks (layer3, layer4)

Matches Section 4.4, Experiment 2 of the proposal.

Results saved to: results/experiment2/
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import load_food101
from src.models.cnn_model import build_resnet50, train_cnn, evaluate_cnn
from src.evaluation.metrics import (
    plot_per_class_accuracy,
    compute_all_metrics,
    plot_training_history,
    plot_model_comparison,
    plot_confusion_matrix,
)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment2")
DATA_ROOT   = os.path.join(os.path.dirname(__file__), "..", "food_data", "food-101")


def run(
    data_root: str = DATA_ROOT,
    results_dir: str = RESULTS_DIR,
    subset_fraction: float = 1.0,
    cnn_epochs: int = 20,
    batch_size: int = 32,
):
    """
    Runs Experiment 2.

    Args:
        data_root:       Directory to download/store Food-101.
        results_dir:     Directory for output files.
        subset_fraction: Use a fraction of the dataset (1.0 = full).
        cnn_epochs:      Max CNN training epochs per configuration.
        batch_size:      DataLoader batch size.
    """
    os.makedirs(results_dir, exist_ok=True)

    data = load_food101(
        data_root=data_root,
        batch_size=batch_size,
        subset_fraction=subset_fraction,
    )
    class_names = data["class_names"]
    all_results = {}

    for mode in ("head_only", "two_blocks"):
        print("\n" + "="*70)
        print(f"  EXPERIMENT 2 — CNN fine_tune_mode='{mode}'")
        print("="*70)

        model = build_resnet50(num_classes=101, fine_tune_mode=mode)

        history = train_cnn(
            model=model,
            train_loader=data["train_loader"],
            val_loader=data["val_loader"],
            fine_tune_mode=mode,
            epochs=cnn_epochs,
            save_path=os.path.join(results_dir, f"cnn_{mode}_best.pth"),
        )

        plot_training_history(
            history,
            save_path=os.path.join(results_dir, f"cnn_{mode}_training_history.png"),
            model_name=f"CNN ResNet-50 ({mode})",
        )
        with open(os.path.join(results_dir, f"cnn_{mode}_history.json"), "w") as f:
            json.dump(history, f, indent=2)

        probs, preds, labels = evaluate_cnn(model, data["test_loader"])
        metrics = compute_all_metrics(
            labels, preds, probs,
            model_name=f"CNN ({mode})",
        )
        plot_confusion_matrix(
            labels, preds, class_names,
            save_path=os.path.join(results_dir, f"cnn_{mode}_confusion_matrix.png"),
            model_name=f"CNN ({mode})",
        )
        plot_per_class_accuracy(
            labels, preds, class_names,
            save_path=os.path.join(results_dir, f"cnn_{mode}_per_class_accuracy.png"),
            model_name=f"CNN ({mode})",
        )

        all_results[f"CNN ({mode})"] = metrics

    # Side-by-side training curves comparison
    _plot_combined_history(results_dir)

    # Bar chart comparison
    plot_model_comparison(
        all_results,
        save_path=os.path.join(results_dir, "transfer_learning_comparison.png"),
    )

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nExperiment 2 results saved to: {results_dir}")
    return all_results


def _plot_combined_history(results_dir: str):
    """Overlays val accuracy for head_only vs two_blocks in a single plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"head_only": "steelblue", "two_blocks": "darkorange"}

    for mode in ("head_only", "two_blocks"):
        history_path = os.path.join(results_dir, f"cnn_{mode}_history.json")
        if not os.path.exists(history_path):
            continue
        with open(history_path) as f:
            h = json.load(f)
        epochs = range(1, len(h["val_acc"]) + 1)
        ax.plot(epochs, [a * 100 for a in h["val_acc"]],
                label=f"{mode} (val)", color=colors[mode])
        ax.plot(epochs, [a * 100 for a in h["train_acc"]],
                linestyle="--", label=f"{mode} (train)", color=colors[mode], alpha=0.5)

    ax.set_title("Experiment 2 — Transfer Learning Depth: Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "combined_accuracy_curves.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 2: Transfer Learning Depth (head_only vs two_blocks)")
    parser.add_argument("--data-root",   default=DATA_ROOT,   help="Food-101 data directory")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Output directory")
    parser.add_argument("--subset",      type=float, default=1.0, help="Fraction of dataset (0–1)")
    parser.add_argument("--epochs",      type=int,   default=20,  help="Max CNN epochs per mode")
    parser.add_argument("--batch-size",  type=int,   default=32,  help="Batch size")
    args = parser.parse_args()

    run(
        data_root=args.data_root,
        results_dir=args.results_dir,
        subset_fraction=args.subset,
        cnn_epochs=args.epochs,
        batch_size=args.batch_size,
    )
