"""
SVM classifier with RBF kernel trained on PCA-reduced HOG + color histogram features.
Includes 5-fold cross-validation grid search for C and gamma hyperparameters.
Matches Section 4.3 of the proposal.
"""

import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score


# Hyperparameter search grid (Section 4.3)
SVM_PARAM_GRID = {
    "C":     [0.1, 1.0, 10.0, 100.0],
    "gamma": ["scale", "auto", 0.001, 0.01],
}


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict | None = None,
    cv_folds: int = 5,
    n_jobs: int = -1,
    verbose: int = 1,
) -> SVC:
    """
    Trains an SVM with RBF kernel using 5-fold cross-validated grid search.

    Args:
        X_train:    (N, n_components) feature matrix.
        y_train:    (N,) label array.
        param_grid: Hyperparameter grid. Defaults to SVM_PARAM_GRID.
        cv_folds:   Number of cross-validation folds.
        n_jobs:     Parallel jobs (-1 = all CPUs).
        verbose:    Verbosity level for GridSearchCV.

    Returns:
        Best-fitted SVC estimator.
    """
    if param_grid is None:
        param_grid = SVM_PARAM_GRID

    base_svm = SVC(kernel="rbf", probability=True, random_state=42, class_weight="balanced")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=base_svm,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        scoring="accuracy",
        refit=True,
    )

    print(f"Starting SVM grid search ({cv_folds}-fold CV) over {len(SVM_PARAM_GRID['C']) * len(SVM_PARAM_GRID['gamma'])} param combinations...")
    grid_search.fit(X_train, y_train)

    print(f"Best SVM params:       {grid_search.best_params_}")
    print(f"Best CV accuracy:      {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_svm(
    svm: SVC,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs inference with the SVM and returns probabilities, predictions, and labels.

    Returns:
        probs:  (N, num_classes) probability estimates (requires probability=True in SVC).
        preds:  (N,) predicted class indices.
        labels: (N,) ground-truth class indices (same as y).
    """
    probs  = svm.predict_proba(X)
    preds  = probs.argmax(axis=1)
    acc    = accuracy_score(y, preds)
    print(f"SVM Test Accuracy: {acc:.4f}")
    return probs, preds, y


def save_svm(svm: SVC, path: str):
    """Persists the fitted SVM to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(svm, path)
    print(f"SVM saved to {path}")


def load_svm(path: str) -> SVC:
    """Loads a previously saved SVM from disk."""
    return joblib.load(path)
