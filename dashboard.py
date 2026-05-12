"""
Food Classification & Calorie Estimation — Dashboard
CSC 481 | Justin Rzepko, Kathya Reynosa, Tiago Freitas
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Food Classification — CSC 481",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Font and background */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
    }
    [data-testid="stSidebar"] * {
        color: #c9d1e0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #c9d1e0 !important;
        font-size: 0.9rem;
        padding: 4px 0;
    }
    [data-testid="stSidebar"] hr {
        border-color: #2e3650;
    }

    /* Main content */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }

    /* Page title */
    h1 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1a1f2e !important;
        border-bottom: 2px solid #e8ecf3;
        padding-bottom: 0.5rem;
        margin-bottom: 1.2rem !important;
    }

    h2 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #2c3450 !important;
        margin-top: 1.5rem !important;
    }

    h3 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #3a4468 !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #f5f7fc;
        border: 1px solid #e0e5f0;
        border-radius: 8px;
        padding: 14px 18px !important;
    }
    [data-testid="metric-container"] label {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        color: #6b7694 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #1a1f2e !important;
    }

    /* Info/success boxes */
    .stAlert {
        border-radius: 8px !important;
        border-left-width: 4px !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f0f3fa;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 6px 16px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #6b7694;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #1a1f2e !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }

    /* Divider */
    hr {
        border-color: #e8ecf3;
        margin: 1.2rem 0;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #c8d0e5;
        border-radius: 10px;
        padding: 1rem;
        background: #f8faff;
    }

    /* Caption text */
    .caption {
        font-size: 0.78rem;
        color: #8a94b0;
        text-align: center;
        margin-top: 0.3rem;
    }

    /* Section label */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8a94b0;
        margin-bottom: 0.4rem;
    }

    /* Prediction row */
    .pred-row {
        background: #f8faff;
        border: 1px solid #e0e8f5;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .pred-rank {
        font-size: 0.8rem;
        font-weight: 700;
        color: #8a94b0;
    }
    .pred-name {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1f2e;
    }
    .pred-conf {
        font-size: 0.85rem;
        color: #4C9BE8;
        font-weight: 600;
    }
    .pred-cal {
        font-size: 0.85rem;
        color: #4CAF82;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

RESULTS  = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
sys.path.insert(0, os.path.dirname(__file__))

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8faff",
    "axes.edgecolor":    "#d0d7e8",
    "axes.labelcolor":   "#3a4468",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.color":       "#6b7694",
    "ytick.color":       "#6b7694",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "grid.color":        "#e5eaf5",
    "grid.linewidth":    0.8,
    "legend.fontsize":   8,
    "font.family":       "sans-serif",
})

BLUE   = "#4C9BE8"
GREEN  = "#4CAF82"
RED    = "#E8645A"
PURPLE = "#9B7FE8"
NAVY   = "#1a1f2e"

@st.cache_data
def load_summary():
    with open(os.path.join(RESULTS, "all_results_summary.json")) as f:
        return json.load(f)

def show_image(path, caption=None):
    if os.path.exists(path):
        st.image(Image.open(path), caption=caption, use_container_width=True)
    else:
        st.caption(f"Image not found: {path}")

def section(text):
    st.markdown(f'<p class="section-label">{text}</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Food Classification")
    st.markdown("**CSC 481 — Artificial Intelligence**")
    st.markdown("Southern Connecticut State University")
    st.divider()
    st.markdown(
        "<p style='font-size:0.8rem;color:#8a94b0;margin-bottom:4px;'>Team Members</p>"
        "<p style='font-size:0.88rem;'>Justin Rzepko<br>Kathya Reynosa<br>Tiago Freitas</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "Navigation",
        ["Overview", "Experiment 1 — Baseline", "Experiment 2 — Transfer Learning",
         "Experiment 3 — Feature Ablation", "Experiment 4 — Calorie Estimation",
         "Live Prediction"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        "<p style='font-size:0.75rem;color:#5a6480;'>Dataset: Food-101<br>"
        "101,000 images · 101 classes<br>"
        "Backbone: ResNet-50 (ImageNet)</p>",
        unsafe_allow_html=True,
    )

if page == "Overview":
    st.title("Project Overview")
    st.markdown(
        "An end-to-end pipeline for food image classification and calorie estimation, "
        "comparing a fine-tuned ResNet-50 CNN against classical machine learning methods "
        "(SVM and Random Forest) on the Food-101 benchmark dataset."
    )

    summary = load_summary()

    section("Key Results — Full Dataset (75,750 train / 25,250 test)")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("CNN Top-1 Accuracy",       f"{summary['experiment1']['CNN (ResNet-50)']['top1_accuracy']*100:.2f}%")
    with c2: st.metric("CNN Top-5 Accuracy",        f"{summary['experiment1']['CNN (ResNet-50)']['top5_accuracy']*100:.2f}%")
    with c3: st.metric("Calorie MAE",               f"{summary['experiment4']['calorie_mae_kcal']:.1f} kcal")
    with c4: st.metric("Calorie Acc. (±50 kcal)",   f"{summary['experiment4']['calorie_accuracy_pct']:.2f}%")

    st.divider()

    col_chart, col_gap, col_dataset = st.columns([5, 0.3, 2])

    with col_chart:
        section("Model Comparison — Full Dataset")
        models  = {
            "CNN (ResNet-50)": summary["experiment1"]["CNN (ResNet-50)"],
            "SVM (HOG+Color)": summary["experiment1"]["SVM (HOG+Color+PCA)"],
            "Random Forest":   summary["experiment1"]["Random Forest"],
        }
        metrics = ["top1_accuracy", "top5_accuracy", "macro_f1"]
        labels  = ["Top-1 Accuracy", "Top-5 Accuracy", "Macro F1"]
        colors  = [BLUE, GREEN, PURPLE]
        x       = np.arange(len(models))
        width   = 0.25

        fig, ax = plt.subplots(figsize=(8, 4))
        for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
            vals = [v[metric] * 100 for v in models.values()]
            bars = ax.bar(x + i * width, vals, width, label=label, color=color,
                          edgecolor="white", linewidth=0.6)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5, color="#3a4468")

        ax.set_xticks(x + width)
        ax.set_xticklabels(list(models.keys()))
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 112)
        ax.set_title("Classifier Performance Comparison", fontweight="bold")
        ax.legend(framealpha=0.9)
        ax.grid(axis="y", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_dataset:
        section("Sample Images — Food-101")
        for path, label in [
            (os.path.join(DATA_DIR, "sample_pizza.jpg"),    "Pizza"),
            (os.path.join(DATA_DIR, "sample_sushi.jpg"),    "Sushi"),
            (os.path.join(DATA_DIR, "sample_hamburger.jpg"),"Hamburger"),
        ]:
            if os.path.exists(path):
                st.image(path, caption=label, use_container_width=True)

    st.divider()
    section("Summary of Experiments")
    r1, r2 = st.columns(2)
    with r1:
        st.info(
            "**Experiment 1 — Baseline:** CNN achieves 82.85% Top-1 vs. 13.89% for SVM "
            "and 5.54% for Random Forest — a 6× gap confirming the necessity of deep features."
        )
        st.info(
            "**Experiment 2 — Transfer Learning:** Fine-tuning last two residual blocks "
            "yields +21.6 pp over head-only fine-tuning (81.83% vs. 61.29%)."
        )
    with r2:
        st.info(
            "**Experiment 3 — Feature Ablation:** Adding color histograms to HOG improves "
            "SVM by +2.83 pp and RF by +0.34 pp — consistent but limited gains."
        )
        st.info(
            "**Experiment 4 — Calorie Estimation:** 96.64% of predictions fall within "
            "±50 kcal of ground truth, with a mean absolute error of only 26.4 kcal."
        )

elif page == "Experiment 1 — Baseline":
    st.title("Experiment 1 — Baseline Classifier Comparison")
    st.markdown(
        "All three classifiers trained and evaluated on the full Food-101 dataset. "
        "The CNN uses ResNet-50 with two-block fine-tuning. SVM and Random Forest use "
        "HOG + color histogram features reduced via PCA (200 components)."
    )

    summary = load_summary()
    exp1    = summary["experiment1"]

    section("Test Set Performance")
    c1, c2, c3 = st.columns(3)
    for col, (name, res) in zip([c1, c2, c3], exp1.items()):
        with col:
            st.markdown(f"**{name}**")
            st.metric("Top-1 Accuracy", f"{res['top1_accuracy']*100:.2f}%")
            st.metric("Top-5 Accuracy", f"{res['top5_accuracy']*100:.2f}%")
            st.metric("Macro F1",       f"{res['macro_f1']*100:.2f}%")

    st.divider()
    tab1, tab2, tab3 = st.tabs(["Model Comparison", "CNN Training Curves", "CNN Confusion Matrix"])

    with tab1:
        show_image(os.path.join(RESULTS, "experiment1", "model_comparison.png"),
                   "Top-1 Accuracy, Top-5 Accuracy, and Macro F1 across all classifiers")
    with tab2:
        show_image(os.path.join(RESULTS, "experiment1", "cnn_training_history.png"),
                   "CNN training and validation loss/accuracy over epochs")
    with tab3:
        show_image(os.path.join(RESULTS, "experiment1", "cnn_confusion_matrix.png"),
                   "Top-20 most confused classes (left) and top-20 best classified classes (right)")

    st.divider()
    gap = exp1["CNN (ResNet-50)"]["top1_accuracy"] / exp1["SVM (HOG+Color+PCA)"]["top1_accuracy"]
    st.info(
        f"The CNN achieves **{exp1['CNN (ResNet-50)']['top1_accuracy']*100:.2f}% Top-1 accuracy**, "
        f"compared to **{exp1['SVM (HOG+Color+PCA)']['top1_accuracy']*100:.2f}%** for SVM and "
        f"**{exp1['Random Forest']['top1_accuracy']*100:.2f}%** for Random Forest — a "
        f"**{gap:.1f}x gap**. Transfer learning with deep features is essential for "
        "fine-grained food recognition across 101 classes."
    )

elif page == "Experiment 2 — Transfer Learning":
    st.title("Experiment 2 — Transfer Learning Depth")
    st.markdown(
        "Comparing two ResNet-50 fine-tuning strategies: "
        "**head-only** (only the final FC layer is trained, all backbone weights frozen) "
        "versus **two-blocks** (FC layer plus the last two residual blocks unfrozen)."
    )

    summary = load_summary()
    exp2    = summary["experiment2"]

    section("Test Set Performance")
    c1, c2 = st.columns(2)
    for col, (name, res) in zip([c1, c2], exp2.items()):
        with col:
            st.markdown(f"**{name}**")
            st.metric("Top-1 Accuracy", f"{res['top1_accuracy']*100:.2f}%")
            st.metric("Top-5 Accuracy", f"{res['top5_accuracy']*100:.2f}%")
            st.metric("Macro F1",       f"{res['macro_f1']*100:.2f}%")

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs([
        "Comparison Chart", "Accuracy Curves",
        "head-only Training", "two-blocks Training",
    ])
    with tab1:
        show_image(os.path.join(RESULTS, "experiment2", "transfer_learning_comparison.png"),
                   "head-only vs. two-blocks — accuracy and F1 comparison")
    with tab2:
        show_image(os.path.join(RESULTS, "experiment2", "combined_accuracy_curves.png"),
                   "Validation accuracy across training epochs")
    with tab3:
        show_image(os.path.join(RESULTS, "experiment2", "cnn_head_only_training_history.png"),
                   "head-only — loss and accuracy curves")
    with tab4:
        show_image(os.path.join(RESULTS, "experiment2", "cnn_two_blocks_training_history.png"),
                   "two-blocks — loss and accuracy curves")

    st.divider()
    delta = (exp2["CNN (two_blocks)"]["top1_accuracy"] - exp2["CNN (head_only)"]["top1_accuracy"]) * 100
    st.info(
        f"Unfreezing the last two residual blocks provides a **+{delta:.2f} pp** Top-1 improvement "
        "over head-only fine-tuning. ImageNet features alone are insufficient — the backbone "
        "requires domain adaptation to food-specific textures and color patterns."
    )

elif page == "Experiment 3 — Feature Ablation":
    st.title("Experiment 3 — Feature Ablation")
    st.markdown(
        "Evaluating the contribution of color histogram features by comparing "
        "**HOG-only** against **HOG + color histograms** for both SVM and Random Forest. "
        "Both variants use PCA reduction to 200 components."
    )

    summary = load_summary()
    exp3    = summary["experiment3"]

    section("Test Set Performance")
    cols = st.columns(4)
    for col, (name, res) in zip(cols, exp3.items()):
        with col:
            st.markdown(f"**{name}**")
            st.metric("Top-1", f"{res['top1_accuracy']*100:.2f}%")
            st.metric("Top-5", f"{res['top5_accuracy']*100:.2f}%")
            st.metric("F1",    f"{res['macro_f1']*100:.2f}%")

    st.divider()
    tab1, tab2, tab3 = st.tabs([
        "Ablation Comparison", "SVM Confusion Matrices", "RF Confusion Matrices"
    ])
    with tab1:
        show_image(os.path.join(RESULTS, "experiment3", "feature_ablation_comparison.png"),
                   "HOG-only vs. HOG+Color — Top-1, Top-5, and Macro F1")
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            show_image(os.path.join(RESULTS, "experiment3", "svm_HOG_only_confusion.png"),
                       "SVM — HOG only")
        with c2:
            show_image(os.path.join(RESULTS, "experiment3", "svm_HOG_plus_Color_confusion.png"),
                       "SVM — HOG + Color")
    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            show_image(os.path.join(RESULTS, "experiment3", "rf_HOG_only_confusion.png"),
                       "Random Forest — HOG only")
        with c2:
            show_image(os.path.join(RESULTS, "experiment3", "rf_HOG_plus_Color_confusion.png"),
                       "Random Forest — HOG + Color")

    st.divider()
    svm_gain = (exp3["SVM [HOG_plus_Color]"]["top1_accuracy"] - exp3["SVM [HOG_only]"]["top1_accuracy"]) * 100
    rf_gain  = (exp3["RF [HOG_plus_Color]"]["top1_accuracy"]  - exp3["RF [HOG_only]"]["top1_accuracy"])  * 100
    st.info(
        f"Adding color features improves SVM by **+{svm_gain:.2f} pp** and Random Forest by "
        f"**+{rf_gain:.2f} pp**. While consistent, the gains are modest — handcrafted features "
        "cannot match deep learned representations for 101-class food recognition."
    )

elif page == "Experiment 4 — Calorie Estimation":
    st.title("Experiment 4 — Calorie Estimation Accuracy")
    st.markdown(
        "Using the best CNN from Experiment 1 to classify food images, then mapping each "
        "predicted label to a calorie range via the USDA-derived nutritional database. "
        "Evaluates end-to-end calorie estimation accuracy."
    )

    summary = load_summary()
    exp4    = summary["experiment4"]

    section("End-to-End Results")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Top-1 Accuracy",             f"{exp4['top1_accuracy']*100:.2f}%")
    with c2: st.metric("Top-5 Accuracy",             f"{exp4['top5_accuracy']*100:.2f}%")
    with c3: st.metric("Calorie MAE",                f"{exp4['calorie_mae_kcal']:.1f} kcal")
    with c4: st.metric("Accuracy within ±50 kcal",   f"{exp4['calorie_accuracy_pct']:.2f}%")

    st.divider()
    tab1, tab2 = st.tabs(["Error Distribution", "Worst Classes by Calorie Error"])
    with tab1:
        show_image(os.path.join(RESULTS, "experiment4", "calorie_error_distribution.png"),
                   "Distribution of absolute calorie prediction errors (kcal)")
    with tab2:
        show_image(os.path.join(RESULTS, "experiment4", "top_calorie_errors_by_class.png"),
                   "Food classes with the highest mean calorie prediction error")

    st.divider()
    st.info(
        f"**{exp4['calorie_accuracy_pct']:.2f}%** of predictions fall within ±50 kcal of the "
        f"true calorie value, with a mean absolute error of **{exp4['calorie_mae_kcal']:.1f} kcal**. "
        "Even when the exact food class is misclassified, the predicted class typically "
        "has a similar caloric density — making this approach practical for dietary tracking."
    )

elif page == "Live Prediction":
    st.title("Live Food Prediction")
    st.markdown(
        "Upload a food photograph and the trained CNN will classify it and estimate "
        "the calorie range from the nutritional database."
    )

    cnn_path = os.path.join(RESULTS, "experiment1", "cnn_best.pth")

    if not os.path.exists(cnn_path):
        st.error("Model file not found at results/experiment1/cnn_best.pth.")
    else:
        uploaded = st.file_uploader("Select a food image (JPG or PNG)", type=["jpg", "jpeg", "png"])

        if uploaded:
            img  = Image.open(uploaded).convert("RGB")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(img, caption="Uploaded image", use_container_width=True)

            with col2:
                with st.spinner("Running inference..."):
                    try:
                        import torch
                        import torch.nn as nn
                        from src.data.dataset import get_eval_transform
                        from src.models.cnn_model import load_cnn
                        from src.calorie.estimator import CalorieEstimator
                        from torchvision.datasets import Food101

                        @st.cache_resource
                        def get_model_and_classes():
                            ds    = Food101(root="./food_data/food-101/food-101",
                                            split="train", download=False)
                            model = load_cnn(cnn_path, num_classes=101,
                                             fine_tune_mode="two_blocks")
                            model.eval()
                            return model, ds.classes

                        model, class_names = get_model_and_classes()
                        estimator = CalorieEstimator()
                        transform = get_eval_transform()
                        tensor    = transform(img).unsqueeze(0)
                        device    = ("mps" if torch.backends.mps.is_available()
                                     else "cuda" if torch.cuda.is_available() else "cpu")
                        model  = model.to(device)
                        tensor = tensor.to(device)

                        with torch.no_grad():
                            logits = model(tensor)
                            probs  = nn.Softmax(dim=1)(logits).cpu().squeeze().numpy()

                        top5_idx = probs.argsort()[-5:][::-1]

                        section("Top-5 Predictions")
                        for rank, idx in enumerate(top5_idx, 1):
                            food  = class_names[idx]
                            info  = estimator.estimate(food)
                            conf  = probs[idx] * 100
                            name  = food.replace("_", " ").title()

                            c1, c2, c3 = st.columns([3, 2, 3])
                            with c1:
                                st.markdown(
                                    f"<span style='color:#8a94b0;font-size:0.75rem;'>"
                                    f"#{rank}</span> &nbsp;"
                                    f"<span style='font-weight:600;color:#1a1f2e;'>{name}</span>",
                                    unsafe_allow_html=True,
                                )
                            with c2:
                                st.markdown(
                                    f"<span style='color:{BLUE};font-weight:600;'>{conf:.1f}%</span> confidence",
                                    unsafe_allow_html=True,
                                )
                            with c3:
                                st.markdown(
                                    f"<span style='color:{GREEN};font-weight:600;'>"
                                    f"{info['min_cal']}–{info['max_cal']} kcal</span> {info['unit']}",
                                    unsafe_allow_html=True,
                                )

                            st.progress(min(int(conf), 100))
                            if rank < 5:
                                st.divider()

                        st.divider()
                        best = class_names[top5_idx[0]]
                        bi   = estimator.estimate(best)
                        st.success(
                            f"**Best prediction:** {best.replace('_', ' ').title()} "
                            f"({probs[top5_idx[0]]*100:.1f}% confidence) — "
                            f"approximately **{bi['avg_cal']} kcal** per serving "
                            f"({bi['min_cal']}–{bi['max_cal']} kcal range)"
                        )

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
        else:
            st.markdown(
                "<div style='text-align:center;padding:3rem;color:#8a94b0;"
                "border:2px dashed #d0d8ee;border-radius:10px;background:#f8faff;'>"
                "Upload an image above to run a prediction."
                "</div>",
                unsafe_allow_html=True,
            )
