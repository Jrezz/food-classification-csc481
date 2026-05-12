"""
Experiment 3 — Feature Ablation
Compares HOG-only vs. HOG + color histograms for SVM and Random Forest.
Results saved to: results/experiment3/
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import load_food101_raw
from src.features.feature_extractor import FeaturePipeline
from src.models.svm_model import train_svm, evaluate_svm, save_svm
from src.models.rf_model import train_random_forest, evaluate_random_forest, save_rf
from src.evaluation.metrics import (
    plot_per_class_accuracy,
    compute_all_metrics,
    plot_model_comparison,
    plot_confusion_matrix,
)


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment3")
DATA_ROOT   = os.path.join(os.path.dirname(__file__), "..", "food_data", "food-101")


def run(
    data_root: str = DATA_ROOT,
    results_dir: str = RESULTS_DIR,
    subset_fraction: float = 1.0,
    pca_components: int = 200,
):
    os.makedirs(results_dir, exist_ok=True)

    raw_data = load_food101_raw(data_root=data_root, subset_fraction=subset_fraction)
    class_names = raw_data["class_names"]

    all_results = {}

    for feature_set, hog_only in [("HOG_only", True), ("HOG_plus_Color", False)]:
        print("\n" + "="*70)
        print(f"  EXPERIMENT 3 — Feature Set: {feature_set}")
        print("="*70)

        pipeline = FeaturePipeline(n_components=pca_components, hog_only=hog_only)

        X_train, y_train = pipeline.fit_transform(raw_data["train_loader"])
        X_test,  y_test  = pipeline.transform(raw_data["test_loader"])

        pipeline.save(os.path.join(results_dir, f"pipeline_{feature_set}.pkl"))

        print(f"\n  SVM [{feature_set}]")
        svm = train_svm(X_train, y_train)
        save_svm(svm, os.path.join(results_dir, f"svm_{feature_set}.pkl"))

        probs, preds, labels = evaluate_svm(svm, X_test, y_test)
        svm_metrics = compute_all_metrics(
            labels, preds, probs,
            model_name=f"SVM [{feature_set}]",
        )
        plot_confusion_matrix(
            labels, preds, class_names,
            save_path=os.path.join(results_dir, f"svm_{feature_set}_confusion.png"),
            model_name=f"SVM [{feature_set}]",
        )
        plot_per_class_accuracy(
            labels, preds, class_names,
            save_path=os.path.join(results_dir, f"svm_{feature_set}_per_class_accuracy.png"),
            model_name=f"SVM [{feature_set}]",
        )
        all_results[f"SVM [{feature_set}]"] = svm_metrics

        print(f"\n  Random Forest [{feature_set}]")
        rf = train_random_forest(X_train, y_train, n_estimators=200)
        save_rf(rf, os.path.join(results_dir, f"rf_{feature_set}.pkl"))

        probs, preds, labels = evaluate_random_forest(rf, X_test, y_test)
        rf_metrics = compute_all_metrics(
            labels, preds, probs,
            model_name=f"RF [{feature_set}]",
        )
        plot_confusion_matrix(
            labels, preds, class_names,
            save_path=os.path.join(results_dir, f"rf_{feature_set}_confusion.png"),
            model_name=f"RF [{feature_set}]",
        )
        plot_per_class_accuracy(
            labels, preds, class_names,
            save_path=os.path.join(results_dir, f"rf_{feature_set}_per_class_accuracy.png"),
            model_name=f"RF [{feature_set}]",
        )
        all_results[f"RF [{feature_set}]"] = rf_metrics

    plot_model_comparison(
        all_results,
        save_path=os.path.join(results_dir, "feature_ablation_comparison.png"),
    )

    _print_ablation_delta(all_results, results_dir)

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nExperiment 3 results saved to: {results_dir}")
    return all_results


def _print_ablation_delta(results: dict, results_dir: str):
    lines = [
        "Feature Ablation Delta (HOG+Color vs HOG-only)\n",
        "=" * 50 + "\n",
    ]
    for clf in ("SVM", "RF"):
        key_hog   = f"{clf} [HOG_only]"
        key_color = f"{clf} [HOG_plus_Color]"
        if key_hog in results and key_color in results:
            for metric in ("top1_accuracy", "top5_accuracy", "macro_f1"):
                delta = results[key_color][metric] - results[key_hog][metric]
                line  = f"  {clf} {metric}: {delta:+.4f} ({delta*100:+.2f}%)\n"
                lines.append(line)
                print(line, end="")

    with open(os.path.join(results_dir, "ablation_delta.txt"), "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experiment 3: Feature Ablation (HOG-only vs HOG+Color)")
    parser.add_argument("--data-root",      default=DATA_ROOT,    help="Food-101 data directory")
    parser.add_argument("--results-dir",    default=RESULTS_DIR,  help="Output directory")
    parser.add_argument("--subset",         type=float, default=1.0, help="Fraction of dataset (0–1)")
    parser.add_argument("--pca-components", type=int,   default=200, help="PCA components")
    args = parser.parse_args()

    run(
        data_root=args.data_root,
        results_dir=args.results_dir,
        subset_fraction=args.subset,
        pca_components=args.pca_components,
    )
