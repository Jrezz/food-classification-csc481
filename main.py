"""
Food Classification and Calorie Estimation Using Deep Learning
==============================================================
CSC 481 — Artificial Intelligence
Team Members: Justin Rzepko, Kathya Reynosa, Tiago Freitas

Main entry point. Runs one or all of the four experiments defined in the proposal.

Usage:
    # Run all experiments (full dataset):
    python main.py --all

    # Run a single experiment:
    python main.py --experiment 1
    python main.py --experiment 2
    python main.py --experiment 3
    python main.py --experiment 4

    # Quick smoke-test with 5% of data:
    python main.py --all --subset 0.05 --epochs 2

    # Predict calories for a single image:
    python main.py --predict path/to/image.jpg --cnn-path results/experiment1/cnn_best.pth
"""

import argparse
import os
import sys
import json

# ─── Quick self-test / demo helpers ──────────────────────────────────────────

def _predict_single(image_path: str, cnn_path: str, data_root: str = "./data"):
    """
    Loads a pre-trained CNN and predicts the food class + calorie estimate
    for a single image file.
    """
    import torch
    from PIL import Image
    from src.data.dataset import get_eval_transform
    from src.models.cnn_model import load_cnn
    from src.calorie.estimator import CalorieEstimator
    from torchvision.datasets import Food101
    import torch.nn as nn

    # Load class names from Food-101 metadata
    ds = Food101(root=data_root, split="train", download=True)
    class_names = ds.classes

    model = load_cnn(cnn_path, num_classes=101, fine_tune_mode="two_blocks")
    model.eval()

    transform = get_eval_transform()
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, tensor = model.to(device), tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = nn.Softmax(dim=1)(logits).cpu().squeeze().numpy()

    top5_idx  = probs.argsort()[-5:][::-1]
    estimator = CalorieEstimator()

    print(f"\nPredictions for: {image_path}")
    print("=" * 55)
    for rank, idx in enumerate(top5_idx, 1):
        food_name = class_names[idx]
        info      = estimator.estimate(food_name)
        print(f"  {rank}. {food_name.replace('_',' ').title():<30} "
              f"({probs[idx]*100:5.1f}%)  "
              f"{info['min_cal']}–{info['max_cal']} kcal {info['unit']}")
    print("=" * 55)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Food Classification and Calorie Estimation — CSC 481",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Experiment selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all",        action="store_true",    help="Run all four experiments sequentially")
    group.add_argument("--experiment", type=int, choices=[1,2,3,4], help="Run a specific experiment (1–4)")
    group.add_argument("--predict",    type=str, metavar="IMAGE", help="Predict food class + calories for IMAGE path")

    # Dataset / training options
    parser.add_argument("--data-root",   default="./food_data/food-101", help="Directory for Food-101 dataset (default: ./food_data/food-101)")
    parser.add_argument("--results-dir", default="./results", help="Root directory for outputs (default: ./results)")
    parser.add_argument("--subset",      type=float, default=1.0,
                        help="Fraction of dataset to use (0–1). Use 0.05 for quick testing.")
    parser.add_argument("--epochs",      type=int, default=20, help="Max CNN training epochs (default: 20)")
    parser.add_argument("--batch-size",  type=int, default=32, help="Batch size for CNN (default: 32)")
    parser.add_argument("--pca",         type=int, default=200, help="PCA components for SVM/RF (default: 200)")
    parser.add_argument("--tolerance",   type=float, default=50.0, help="Calorie tolerance in kcal for Exp 4 (default: 50)")

    # Experiment 1 skip flags
    parser.add_argument("--skip-cnn", action="store_true", help="(Exp 1) Skip CNN")
    parser.add_argument("--skip-svm", action="store_true", help="(Exp 1) Skip SVM")
    parser.add_argument("--skip-rf",  action="store_true", help="(Exp 1) Skip Random Forest")

    # Predict mode
    parser.add_argument("--cnn-path", default=None, help="Path to CNN .pth file (for --predict)")

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.predict:
        if not args.cnn_path:
            # Try default location
            default = os.path.join(args.results_dir, "experiment1", "cnn_best.pth")
            if os.path.exists(default):
                args.cnn_path = default
            else:
                parser.error("--cnn-path is required when using --predict")
        _predict_single(args.predict, args.cnn_path, data_root=args.data_root)
        return

    # ── Experiment runners ────────────────────────────────────────────────────
    from experiments.experiment1_baseline import run as run_exp1
    from experiments.experiment2_transfer_learning import run as run_exp2
    from experiments.experiment3_feature_ablation import run as run_exp3
    from experiments.experiment4_calorie_accuracy import run as run_exp4

    all_summaries = {}

    def _exp1():
        print("\n" + "#"*70)
        print("  EXPERIMENT 1: Baseline (CNN vs SVM vs Random Forest)")
        print("#"*70)
        return run_exp1(
            data_root    = args.data_root,
            results_dir  = os.path.join(args.results_dir, "experiment1"),
            subset_fraction = args.subset,
            cnn_epochs   = args.epochs,
            batch_size   = args.batch_size,
            pca_components = args.pca,
            skip_svm     = args.skip_svm,
            skip_rf      = args.skip_rf,
            skip_cnn     = args.skip_cnn,
        )

    def _exp2():
        print("\n" + "#"*70)
        print("  EXPERIMENT 2: Transfer Learning Depth (head_only vs two_blocks)")
        print("#"*70)
        return run_exp2(
            data_root    = args.data_root,
            results_dir  = os.path.join(args.results_dir, "experiment2"),
            subset_fraction = args.subset,
            cnn_epochs   = args.epochs,
            batch_size   = args.batch_size,
        )

    def _exp3():
        print("\n" + "#"*70)
        print("  EXPERIMENT 3: Feature Ablation (HOG-only vs HOG+Color)")
        print("#"*70)
        return run_exp3(
            data_root    = args.data_root,
            results_dir  = os.path.join(args.results_dir, "experiment3"),
            subset_fraction = args.subset,
            pca_components = args.pca,
        )

    def _exp4():
        print("\n" + "#"*70)
        print("  EXPERIMENT 4: Calorie Estimation Accuracy")
        print("#"*70)
        return run_exp4(
            data_root    = args.data_root,
            results_dir  = os.path.join(args.results_dir, "experiment4"),
            subset_fraction = args.subset,
            cnn_epochs   = args.epochs,
            batch_size   = args.batch_size,
            tolerance_kcal = args.tolerance,
        )

    runners = {1: _exp1, 2: _exp2, 3: _exp3, 4: _exp4}

    if args.all:
        for num, fn in runners.items():
            all_summaries[f"experiment{num}"] = fn()
    else:
        all_summaries[f"experiment{args.experiment}"] = runners[args.experiment]()

    # Write consolidated summary
    summary_path = os.path.join(args.results_dir, "all_results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll results summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
