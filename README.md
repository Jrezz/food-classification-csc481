# Food Classification & Calorie Estimation — CSC 481

**Team:** Justin Rzepko, Kathya Reynosa, Tiago Freitas  
**Course:** CSC 481 — Machine Learning, St. Cloud State University

A comparative study of food image classification on the [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset using a fine-tuned ResNet-50 CNN, an SVM, and a Random Forest, along with a calorie estimation pipeline and an interactive Streamlit dashboard.

---

## Project Structure

```
481Project/
├── main.py                          # Entry point — run experiments or single predictions
├── dashboard.py                     # Streamlit interactive dashboard
├── requirements.txt
├── nutrition.csv                    # Nutritional reference data
├── data/
│   ├── calorie_data.json            # Calorie lookup table
│   └── sample_*.jpg                 # Sample images for quick testing
├── experiments/
│   ├── experiment1_baseline.py      # CNN vs SVM vs Random Forest
│   ├── experiment2_transfer_learning.py  # head_only vs two_blocks fine-tuning
│   ├── experiment3_feature_ablation.py   # HOG-only vs HOG+Color features
│   └── experiment4_calorie_accuracy.py   # End-to-end calorie estimation accuracy
└── src/
    ├── data/dataset.py              # Food-101 loading, transforms, augmentation
    ├── models/
    │   ├── cnn_model.py             # ResNet-50 with configurable fine-tuning
    │   ├── svm_model.py             # SVM classifier
    │   └── rf_model.py              # Random Forest classifier
    ├── features/feature_extractor.py  # HOG + color histogram pipeline
    ├── calorie/estimator.py         # Calorie lookup and estimation
    └── evaluation/metrics.py        # Accuracy, confusion matrix, plots
```

---

## Setup

**Requirements:** Python 3.11+

```bash
# Create and activate virtual environment
python3.11 -m venv .venv311
source .venv311/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The Food-101 dataset (~4.6 GB) downloads automatically on first run.

---

## Usage

### Run all four experiments

```bash
python main.py --all --data-root ./food_data/food-101
```

### Run a specific experiment

```bash
python main.py --experiment 1   # Baseline: CNN vs SVM vs RF
python main.py --experiment 2   # Transfer learning depth
python main.py --experiment 3   # Feature ablation
python main.py --experiment 4   # Calorie estimation accuracy
```

### Quick test (5% subset, 2 epochs)

```bash
python main.py --all --subset 0.05 --epochs 2
```

### Predict a single image

```bash
python main.py --predict data/sample_pizza.jpg --cnn-path results/experiment1/cnn_best.pth
```

### Launch the Streamlit dashboard

```bash
streamlit run dashboard.py
```

---

## Experiments

| # | Name | Key Question |
|---|------|-------------|
| 1 | Baseline | How does a fine-tuned CNN compare to SVM and Random Forest on Food-101? |
| 2 | Transfer Learning Depth | Does unfreezing more ResNet-50 layers (`head_only` vs `two_blocks`) improve accuracy? |
| 3 | Feature Ablation | Does adding color histograms to HOG features improve classical classifier performance? |
| 4 | Calorie Estimation Accuracy | How accurately can the system estimate calories given correct and top-5 food predictions? |

Results (metrics, plots, saved models) are written to `results/experiment{N}/`.

---

## Models

### ResNet-50 CNN

Pretrained on ImageNet. Two fine-tuning modes:

- **`head_only`** — only the final fully-connected layer is trained
- **`two_blocks`** — FC layer + `layer3` and `layer4` residual blocks are trained; differential learning rates applied (backbone: `lr × 0.1`, head: `lr`)

Input images are normalized to ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) to match the pretrained weight distribution.

### SVM / Random Forest

Features are extracted from raw `[0, 1]` pixel values (no ImageNet normalization) using:

- **HOG** (Histogram of Oriented Gradients) — captures shape and texture
- **Color histograms** — 32-bin histograms per RGB channel

PCA is applied before training (default: 200 components). Models are saved as `.joblib` files.

---

## Key Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--data-root` | `./food_data/food-101` | Path to Food-101 dataset |
| `--results-dir` | `./results` | Output directory for results |
| `--subset` | `1.0` | Fraction of dataset to use (0–1) |
| `--epochs` | `20` | Max CNN training epochs |
| `--batch-size` | `32` | CNN batch size |
| `--pca` | `200` | PCA components for SVM/RF |
| `--tolerance` | `50.0` | Calorie tolerance in kcal (Experiment 4) |
