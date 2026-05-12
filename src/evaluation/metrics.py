"""
Evaluation metrics and plots for all classifiers.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def top5_accuracy(y_true: np.ndarray, probs: np.ndarray) -> float:
    top5_preds = np.argsort(probs, axis=1)[:, -5:]
    correct = sum(
        1 for true, top5 in zip(y_true, top5_preds) if true in top5
    )
    return correct / len(y_true)


def macro_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(precision_score(y_true, y_pred, average="macro", zero_division=0))


def macro_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    model_name: str = "Model",
) -> dict:
    top1  = top1_accuracy(y_true, y_pred)
    top5  = top5_accuracy(y_true, probs)
    prec  = macro_precision(y_true, y_pred)
    rec   = macro_recall(y_true, y_pred)
    f1    = macro_f1(y_true, y_pred)

    metrics = {
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
    }

    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*60}")
    print(f"  Top-1 Accuracy:    {top1:.4f}  ({top1*100:.2f}%)")
    print(f"  Top-5 Accuracy:    {top5:.4f}  ({top5*100:.2f}%)")
    print(f"  Macro Precision:   {prec:.4f}")
    print(f"  Macro Recall:      {rec:.4f}")
    print(f"  Macro F1-Score:    {f1:.4f}")
    print(f"{'='*60}\n")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | None = None,
    top_n: int = 20,
    model_name: str = "Model",
):
    cm = confusion_matrix(y_true, y_pred)
    errors_per_class   = cm.sum(axis=1) - np.diag(cm)
    per_class_accuracy = np.diag(cm) / (cm.sum(axis=1) + 1e-9)

    worst_idx = np.sort(np.argsort(errors_per_class)[-top_n:])
    best_idx  = np.sort(np.argsort(per_class_accuracy)[-top_n:])

    fig, (ax_worst, ax_best) = plt.subplots(1, 2, figsize=(28, 12))

    for ax, idx, title_tag, cmap in [
        (ax_worst, worst_idx, f"Top-{top_n} Most Confused", "Reds"),
        (ax_best,  best_idx,  f"Top-{top_n} Best Classified", "Greens"),
    ]:
        cm_sub = cm[np.ix_(idx, idx)]
        labels = [class_names[i].replace("_", "\n") for i in idx]
        sns.heatmap(
            cm_sub,
            annot=True,
            fmt="d",
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=True,
        )
        ax.set_title(f"{model_name}\n{title_tag}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

    plt.suptitle(f"{model_name} — Confusion Matrices", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.close(fig)


def plot_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | None = None,
    top_n: int = 20,
    model_name: str = "Model",
):
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-9)

    best_idx  = np.argsort(per_class_acc)[-top_n:][::-1]
    worst_idx = np.argsort(per_class_acc)[:top_n]

    fig, (ax_best, ax_worst) = plt.subplots(1, 2, figsize=(18, 8))

    for ax, idx, color, label in [
        (ax_best,  best_idx,  "seagreen",  f"Top-{top_n} Best"),
        (ax_worst, worst_idx, "tomato",    f"Top-{top_n} Worst"),
    ]:
        accs   = per_class_acc[idx] * 100
        names  = [class_names[i].replace("_", " ").title() for i in idx]
        bars   = ax.barh(names[::-1], accs[::-1], color=color, edgecolor="white")
        for bar, val in zip(bars, accs[::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8)
        ax.set_xlim(0, 110)
        ax.set_xlabel("Per-Class Accuracy (%)")
        ax.set_title(f"{label} Classified Classes", fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)

    plt.suptitle(f"{model_name} — Per-Class Accuracy", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Per-class accuracy chart saved to {save_path}")
    plt.close(fig)


def plot_training_history(
    history: dict,
    save_path: str | None = None,
    model_name: str = "CNN",
):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    ax1.set_title(f"{model_name} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],   "r-o", label="Val Acc",   markersize=4)
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Training history saved to {save_path}")
    plt.close(fig)


def plot_model_comparison(
    results: dict[str, dict],
    save_path: str | None = None,
):
    metrics_to_plot = ["top1_accuracy", "top5_accuracy", "macro_f1"]
    labels          = list(results.keys())
    x               = np.arange(len(labels))
    width           = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics_to_plot):
        values = [results[m].get(metric, 0) * 100 for m in labels]
        bars   = ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title())
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — Classification Metrics")
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Model comparison plot saved to {save_path}")
    plt.close(fig)


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    save_path: str | None = None,
):
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report)
        print(f"Classification report saved to {save_path}")
