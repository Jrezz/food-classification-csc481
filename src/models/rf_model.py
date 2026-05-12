"""
Random Forest classifier with 200 estimators trained on PCA-reduced HOG + color features.
Matches Section 4.3 of the proposal.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    n_jobs: int = -1,
    verbose: int = 1,
) -> RandomForestClassifier:
    """
    Trains a Random Forest classifier.

    Args:
        X_train:      (N, n_components) feature matrix.
        y_train:      (N,) label array.
        n_estimators: Number of trees (default 200 per proposal).
        n_jobs:       Parallel jobs (-1 = all CPUs).
        verbose:      Verbosity level.

    Returns:
        Fitted RandomForestClassifier.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    print(f"Training Random Forest with {n_estimators} estimators...")
    rf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, rf.predict(X_train))
    print(f"Random Forest Train Accuracy: {train_acc:.4f}")
    return rf


def evaluate_random_forest(
    rf: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs inference and returns probabilities, predictions, and labels.

    Returns:
        probs:  (N, num_classes) probability estimates.
        preds:  (N,) predicted class indices.
        labels: (N,) ground-truth class indices (same as y).
    """
    probs = rf.predict_proba(X)
    preds = probs.argmax(axis=1)
    acc   = accuracy_score(y, preds)
    print(f"Random Forest Test Accuracy: {acc:.4f}")
    return probs, preds, y


def save_rf(rf: RandomForestClassifier, path: str):
    """Persists the fitted Random Forest to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(rf, path)
    print(f"Random Forest saved to {path}")


def load_rf(path: str) -> RandomForestClassifier:
    """Loads a previously saved Random Forest from disk."""
    return joblib.load(path)
