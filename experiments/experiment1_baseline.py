"""
Experiment 1 — Baseline
Train and evaluate CNN, SVM, and Random Forest on the full Food-101 dataset.
Results saved to: results/experiment1/
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import load_food101, load_food101_raw
from src.features.feature_extractor import FeaturePipeline
from src.models.cnn_model import build_resnet50, train_cnn, evaluate_cnn
from src.models.svm_model import train_svm, evaluate_svm, save_svm
from src.models.rf_model import train_random_forest, evaluate_random_forest, save_rf
from src.evaluation.metrics import (
    plot_per_class_accuracy,
    compute_all_metrics,
    plot_confusion_matrix,
    plot_training_history,
    plot_model_comparison,
    print_classification_report,
)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment1")
DATA_ROOT   = os.path.join(os.path.dirname(__file__), "..", "food_data", "food-101")


def run(
    data_root: str = DATA_ROOT,
    results_dir: str = RESULTS_DIR,
    subset_fraction: float = 1.0,
    cnn_epochs: int = 20,
    batch_size: int = 32,
    pca_components: int = 200,
    skip_svm: bool = False,
    skip_rf: bool = False,
    skip_cnn: bool = False,
):
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}

    if not skip_cnn:
        print("\n" + "="*70)
        print("  EXPERIMENT 1 — CNN (ResNet-50, two_blocks fine-tuning)")
        print("="*70)

        cnn_data = load_food101(
            data_root=data_root,
            batch_size=batch_size,
            subset_fraction=subset_fraction,
        )
        class_names = cnn_data["class_names"]

        model = build_resnet50(num_classes=101, fine_tune_mode="two_blocks")
        history = train_cnn(
            model=model,
            train_loader=cnn_data["train_loader"],
            val_loader=cnn_data["val_loader"],
            fine_tune_mode="two_blocks",
            epochs=cnn_epochs,
            save_path=os.path.join(results_dir, "cnn_best.pth"),
        )

        plot_training_history(
            history,
            save_path=os.path.join(results_dir, "cnn_training_history.png"),
            model_name="CNN (ResNet-50, two_blocks)",
        )

        probs, preds, labels = evaluate_cnn(model, cnn_data["test_loader"])
        cnn_metrics = compute_all_metrics(labels, preds, probs, model_name="CNN (ResNet-50)")

        plot_confusion_matrix(
            labels, preds, class_names,
            save_path=os.path.join(results_dir, "cnn_confusion_matrix.png"),
            model_name="CNN (ResNet-50)",
        )
        plot_per_class_accuracy(
            labels, preds, class_names,
            save_path=os.path.join(results_dir, "cnn_per_class_accuracy.png"),
            model_name="CNN (ResNet-50)",
        )
        print_classification_report(
            labels, preds, class_names,
            save_path=os.path.join(results_dir, "cnn_classification_report.txt"),
        )

        all_results["CNN (ResNet-50)"] = cnn_metrics

        with open(os.path.join(results_dir, "cnn_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    if not skip_svm:
        print("\n" + "="*70)
        print("  EXPERIMENT 1 — SVM (HOG + Color Histogram + PCA)")
        print("="*70)

        raw_data = load_food101_raw(
            data_root=data_root,
            subset_fraction=subset_fraction,
        )
        class_names = raw_data["class_names"]

        pipeline = FeaturePipeline(n_components=pca_components, hog_only=False)

        X_train, y_train = pipeline.fit_transform(raw_data["train_loader"])
        X_test,  y_test  = pipeline.transform(raw_data["test_loader"])

        pipeline.save(os.path.join(results_dir, "feature_pipeline_hog_color.pkl"))

        svm = train_svm(X_train, y_train)
        save_svm(svm, os.path.join(results_dir, "svm_best.pkl"))

        probs, preds, labels = evaluate_svm(svm, X_test, y_test)
        svm_metrics = compute_all_metrics(labels, preds, probs, model_name="SVM (HOG+Color+PCA)")

        plot_confusion_matrix(
            labels, preds, class_names,
            save_path=os.path.join(results_dir, "svm_confusion_matrix.png"),
            model_name="SVM (HOG+Color+PCA)",
        )
        plot_per_class_accuracy(
            labels, preds, class_names,
            save_path=os.path.join(results_dir, "svm_per_class_accuracy.png"),
            model_name="SVM (HOG+Color+PCA)",
        )
        print_classification_report(
            labels, preds, class_names,
            save_path=os.path.join(results_dir, "svm_classification_report.txt"),
        )

        all_results["SVM (HOG+Color+PCA)"] = svm_metrics

    if not skip_rf:
        print("\n" + "="*70)
        print("  EXPERIMENT 1 — Random Forest (HOG + Color Histogram + PCA)")
        print("="*70)

        if skip_svm:
            raw_data = load_food101_raw(
                data_root=data_root,
                subset_fraction=subset_fraction,
            )
            class_names = raw_data["class_names"]
            pipeline    = FeaturePipeline(n_components=pca_components, hog_only=False)
            X_train, y_train = pipeline.fit_transform(raw_data["train_loader"])
            X_test,  y_test  = pipeline.transform(raw_data["test_loader"])

        rf = train_random_forest(X_train, y_train, n_estimators=200)
        save_rf(rf, os.path.join(results_dir, "rf_best.pkl"))

        probs, preds, labels = evaluate_random_forest(rf, X_test, y_test)
        rf_metrics = compute_all_metrics(labels, preds, probs, model_name="Random Forest")

        plot_confusion_matrix(
            labels, preds, class_names,
            save_path=os.path.join(results_dir, "rf_confusion_matrix.png"),
            model_name="Random Forest",
        )
        plot_per_class_accuracy(
            labels, preds, class_names,
            save_path=os.path.join(results_dir, "rf_per_class_accuracy.png"),
            model_name="Random Forest",
        )
        print_classification_report(
            labels, preds, class_names,
            save_path=os.path.join(results_dir, "rf_classification_report.txt"),
        )

        all_results["Random Forest"] = rf_metrics

    if len(all_results) > 1:
        plot_model_comparison(
            all_results,
            save_path=os.path.join(results_dir, "model_comparison.png"),
        )

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nExperiment 1 results saved to: {results_dir}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 1: Baseline comparison of CNN, SVM, RF")
    parser.add_argument("--data-root",      default=DATA_ROOT,    help="Food-101 data directory")
    parser.add_argument("--results-dir",    default=RESULTS_DIR,  help="Output directory")
    parser.add_argument("--subset",         type=float, default=1.0, help="Fraction of dataset to use (0–1)")
    parser.add_argument("--epochs",         type=int,   default=20,  help="Max CNN training epochs")
    parser.add_argument("--batch-size",     type=int,   default=32,  help="CNN batch size")
    parser.add_argument("--pca-components", type=int,   default=200, help="PCA components for SVM/RF")
    parser.add_argument("--skip-svm",  action="store_true", help="Skip SVM training")
    parser.add_argument("--skip-rf",   action="store_true", help="Skip Random Forest training")
    parser.add_argument("--skip-cnn",  action="store_true", help="Skip CNN training")
    args = parser.parse_args()

    run(
        data_root=args.data_root,
        results_dir=args.results_dir,
        subset_fraction=args.subset,
        cnn_epochs=args.epochs,
        batch_size=args.batch_size,
        pca_components=args.pca_components,
        skip_svm=args.skip_svm,
        skip_rf=args.skip_rf,
        skip_cnn=args.skip_cnn,
    )
