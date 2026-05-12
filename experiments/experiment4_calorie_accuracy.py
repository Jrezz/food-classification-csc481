"""
Experiment 4 — Calorie Estimation Accuracy
============================================
Uses the best-performing classifier (CNN) to evaluate end-to-end calorie
estimation accuracy by comparing predicted calorie ranges to USDA ground-truth.

Metrics reported:
  - MAE (kcal) between predicted and true average calorie values
  - Calorie accuracy (% within ±50 kcal tolerance)
  - Per-class calorie lookup table
  - Distribution of calorie errors

Matches Section 4.4, Experiment 4 of the proposal.

Results saved to: results/experiment4/
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import load_food101
from src.models.cnn_model import build_resnet50, train_cnn, evaluate_cnn, load_cnn
from src.calorie.estimator import CalorieEstimator
from src.evaluation.metrics import compute_all_metrics, plot_training_history


RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results", "experiment4")
DATA_ROOT    = os.path.join(os.path.dirname(__file__), "..", "food_data", "food-101")
# Reuse the best CNN from Experiment 1 if it exists
EXP1_CNN_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "experiment1", "cnn_best.pth")


def run(
    data_root: str = DATA_ROOT,
    results_dir: str = RESULTS_DIR,
    subset_fraction: float = 1.0,
    cnn_epochs: int = 20,
    batch_size: int = 32,
    cnn_model_path: str | None = None,
    tolerance_kcal: float = 50.0,
):
    """
    Runs Experiment 4.

    Args:
        data_root:       Directory to download/store Food-101.
        results_dir:     Directory for output files.
        subset_fraction: Use a fraction of the dataset (1.0 = full).
        cnn_epochs:      Max CNN training epochs (only if no saved model found).
        batch_size:      DataLoader batch size.
        cnn_model_path:  Path to a pre-trained CNN .pth file (trains from scratch if None).
        tolerance_kcal:  Tolerance window for calorie accuracy (default ±50 kcal).
    """
    os.makedirs(results_dir, exist_ok=True)

    data        = load_food101(data_root=data_root, batch_size=batch_size, subset_fraction=subset_fraction)
    class_names = data["class_names"]
    estimator   = CalorieEstimator()

    # ─── Load or train CNN ────────────────────────────────────────────────────
    best_path = cnn_model_path or EXP1_CNN_PATH
    if best_path and os.path.exists(best_path):
        print(f"Loading pre-trained CNN from {best_path}")
        model = load_cnn(best_path, num_classes=101, fine_tune_mode="two_blocks")
    else:
        print("No pre-trained CNN found. Training from scratch...")
        model = build_resnet50(num_classes=101, fine_tune_mode="two_blocks")
        history = train_cnn(
            model=model,
            train_loader=data["train_loader"],
            val_loader=data["val_loader"],
            fine_tune_mode="two_blocks",
            epochs=cnn_epochs,
            save_path=os.path.join(results_dir, "cnn_best.pth"),
        )
        plot_training_history(
            history,
            save_path=os.path.join(results_dir, "cnn_training_history.png"),
            model_name="CNN (Experiment 4)",
        )

    # ─── Classification evaluation ────────────────────────────────────────────
    print("\nRunning CNN inference on test set...")
    probs, preds, labels = evaluate_cnn(model, data["test_loader"])
    clf_metrics = compute_all_metrics(labels, preds, probs, model_name="CNN (Experiment 4)")

    # ─── Calorie estimation ───────────────────────────────────────────────────
    print("\nComputing calorie estimation metrics...")
    mae = estimator.compute_mae(preds, labels, class_names)
    cal_acc = estimator.compute_calorie_accuracy(preds, labels, class_names, tolerance_kcal=tolerance_kcal)

    print(f"\n{'='*60}")
    print("  Calorie Estimation Results")
    print(f"{'='*60}")
    print(f"  MAE (kcal):                   {mae:.2f}")
    print(f"  Calorie Accuracy (±{tolerance_kcal:.0f} kcal):  {cal_acc:.4f}  ({cal_acc*100:.2f}%)")
    print(f"{'='*60}")

    # ─── Per-class calorie table ──────────────────────────────────────────────
    calorie_table = _build_calorie_table(class_names, estimator)
    calorie_table_path = os.path.join(results_dir, "calorie_table.json")
    with open(calorie_table_path, "w") as f:
        json.dump(calorie_table, f, indent=2)
    print(f"\nCalorie table (all 101 classes) saved to: {calorie_table_path}")

    # ─── Error distribution plot ──────────────────────────────────────────────
    _plot_calorie_error_distribution(preds, labels, class_names, estimator, results_dir)

    # ─── Per-class calorie error ──────────────────────────────────────────────
    _plot_top_calorie_errors(preds, labels, class_names, estimator, results_dir)

    # ─── Save summary ─────────────────────────────────────────────────────────
    summary = {
        **clf_metrics,
        "calorie_mae_kcal":          mae,
        "calorie_accuracy_pct":      cal_acc * 100,
        "tolerance_kcal":            tolerance_kcal,
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nExperiment 4 results saved to: {results_dir}")
    return summary


def _build_calorie_table(class_names: list[str], estimator: CalorieEstimator) -> list[dict]:
    """Builds a sorted list of all Food-101 classes with their calorie info."""
    table = []
    for cls in class_names:
        info = estimator.estimate(cls)
        table.append({
            "food":       cls.replace("_", " ").title(),
            "class_name": cls,
            **info,
        })
    table.sort(key=lambda x: x["avg_cal"])
    return table


def _plot_calorie_error_distribution(
    preds, labels, class_names, estimator, results_dir
):
    """Histogram of calorie prediction errors (predicted_avg - true_avg)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_cals = estimator.bulk_estimate(preds,  class_names)
    true_cals = estimator.bulk_estimate(labels, class_names)
    errors    = pred_cals - true_cals

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero error")
    ax.set_xlabel("Calorie Error (predicted − true avg kcal)")
    ax.set_ylabel("Count")
    ax.set_title("Experiment 4 — Distribution of Calorie Estimation Errors")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "calorie_error_distribution.png"), dpi=150)
    plt.close(fig)
    print("Calorie error distribution plot saved.")


def _plot_top_calorie_errors(
    preds, labels, class_names, estimator, results_dir, top_n: int = 20
):
    """Bar chart of the top-N food classes with the highest mean calorie error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    rows = []
    for pred_idx, true_idx in zip(preds, labels):
        pred_cal = estimator.estimate(class_names[pred_idx])["avg_cal"]
        true_cal = estimator.estimate(class_names[true_idx])["avg_cal"]
        rows.append({
            "true_class": class_names[true_idx],
            "error":      abs(pred_cal - true_cal),
        })

    df   = pd.DataFrame(rows)
    grp  = df.groupby("true_class")["error"].mean().sort_values(ascending=False).head(top_n)
    labels_plot = [c.replace("_", " ").title() for c in grp.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(labels_plot[::-1], grp.values[::-1], color="salmon")
    ax.set_xlabel("Mean Absolute Calorie Error (kcal)")
    ax.set_title(f"Experiment 4 — Top-{top_n} Classes by Mean Calorie Error")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, "top_calorie_errors_by_class.png"), dpi=150)
    plt.close(fig)
    print("Top calorie errors by class plot saved.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 4: Calorie Estimation Accuracy")
    parser.add_argument("--data-root",    default=DATA_ROOT,    help="Food-101 data directory")
    parser.add_argument("--results-dir",  default=RESULTS_DIR,  help="Output directory")
    parser.add_argument("--subset",       type=float, default=1.0, help="Fraction of dataset (0–1)")
    parser.add_argument("--epochs",       type=int,   default=20,  help="Max CNN epochs")
    parser.add_argument("--batch-size",   type=int,   default=32,  help="Batch size")
    parser.add_argument("--cnn-path",     default=None, help="Path to pre-trained CNN .pth file")
    parser.add_argument("--tolerance",    type=float, default=50.0, help="Calorie tolerance (kcal)")
    args = parser.parse_args()

    run(
        data_root=args.data_root,
        results_dir=args.results_dir,
        subset_fraction=args.subset,
        cnn_epochs=args.epochs,
        batch_size=args.batch_size,
        cnn_model_path=args.cnn_path,
        tolerance_kcal=args.tolerance,
    )
