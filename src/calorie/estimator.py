"""
Calorie estimation module.
Maps Food-101 predicted class labels to nutrition.csv-derived calorie data.
Each class has 5 portion sizes with full macros (protein, carbs, fat, fiber, sugars, sodium).
Supports per-weight lookup, range lookup, MAE evaluation, and calorie accuracy scoring.
"""

import os
import json
import numpy as np
from pathlib import Path


# Default path to the calorie database JSON (relative to project root)
_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", "..", "data", "calorie_data.json")


class CalorieEstimator:
    """
    Maps food class names (Food-101 format) to nutrition.csv calorie data.

    Each entry contains:
      - min_cal / max_cal / avg_cal : calorie range across the 5 portion sizes
      - per_100g                    : calories per 100g (interpolated from CSV)
      - unit                        : human-readable serving description
      - portions                    : list of {weight_g, calories, protein_g, carbs_g,
                                              fat_g, fiber_g, sugars_g, sodium_mg}

    Usage:
        estimator = CalorieEstimator()
        info = estimator.estimate("pizza")
        detail = estimator.estimate_for_weight("pizza", weight_g=150)
    """

    def __init__(self, db_path: str | None = None):
        """
        Args:
            db_path: Path to calorie_data.json. Uses bundled database by default.
        """
        path = db_path or _DEFAULT_DB
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calorie database not found: {path}")
        with open(path, "r") as f:
            self._db: dict = json.load(f)

    def estimate(self, class_name: str) -> dict:
        """
        Returns calorie info for a given food class name.

        Args:
            class_name: Food-101 class name (e.g. 'pizza', 'apple_pie').

        Returns:
            Dict with keys: min_cal, max_cal, avg_cal, per_100g, unit.
            Returns a fallback dict if the class is not in the database.
        """
        key = class_name.lower().replace(" ", "_")
        if key in self._db:
            return self._db[key]
        # Fuzzy fallback: find closest key by prefix
        for db_key in self._db:
            if db_key.startswith(key[:4]):
                return self._db[db_key]
        return {"min_cal": 0, "max_cal": 0, "avg_cal": 0, "per_100g": 0, "unit": "unknown"}

    def estimate_for_weight(self, class_name: str, weight_g: float) -> dict:
        """
        Interpolates nutrition values for a given portion weight.

        Args:
            class_name: Food-101 class name.
            weight_g:   Portion weight in grams.

        Returns:
            Dict with interpolated calories and all macros at that weight.
        """
        info = self.estimate(class_name)
        portions = info.get("portions", [])
        if not portions:
            # fallback for the 7 classes without portion data
            factor = weight_g / 100.0
            return {
                "weight_g":  weight_g,
                "calories":  round(info["per_100g"] * factor),
                "protein_g": None, "carbs_g": None, "fat_g": None,
                "fiber_g":   None, "sugars_g": None, "sodium_mg": None,
            }
        weights  = np.array([p["weight_g"] for p in portions], dtype=float)
        keys     = ["calories", "protein_g", "carbs_g", "fat_g", "fiber_g", "sugars_g", "sodium_mg"]
        result   = {"weight_g": weight_g}
        for key in keys:
            values      = np.array([p[key] for p in portions], dtype=float)
            result[key] = round(float(np.interp(weight_g, weights, values)), 1)
        return result

    def estimate_from_index(self, class_idx: int, class_names: list[str]) -> dict:
        """
        Convenience wrapper that takes a class index and the class_names list.
        """
        return self.estimate(class_names[class_idx])

    def bulk_estimate(
        self,
        predicted_indices: np.ndarray,
        class_names: list[str],
    ) -> np.ndarray:
        """
        Returns an array of predicted average calories for a batch of predictions.

        Args:
            predicted_indices: (N,) array of predicted class indices.
            class_names:       List of class name strings (101 entries for Food-101).

        Returns:
            (N,) float array of avg_cal values.
        """
        return np.array([
            self.estimate(class_names[idx])["avg_cal"]
            for idx in predicted_indices
        ], dtype=np.float32)

    def compute_mae(
        self,
        predicted_indices: np.ndarray,
        true_indices: np.ndarray,
        class_names: list[str],
    ) -> float:
        """
        Computes the Mean Absolute Error between predicted and ground-truth
        average calorie values (Experiment 4, Section 4.5).

        MAE = mean(|predicted_avg_cal - true_avg_cal|)

        Args:
            predicted_indices: (N,) array of predicted class indices.
            true_indices:      (N,) array of ground-truth class indices.
            class_names:       Food-101 class name list.

        Returns:
            MAE in kcal.
        """
        pred_cals = self.bulk_estimate(predicted_indices, class_names)
        true_cals = self.bulk_estimate(true_indices,      class_names)
        mae = float(np.mean(np.abs(pred_cals - true_cals)))
        return mae

    def compute_calorie_accuracy(
        self,
        predicted_indices: np.ndarray,
        true_indices: np.ndarray,
        class_names: list[str],
        tolerance_kcal: float = 50.0,
    ) -> float:
        """
        Fraction of samples where the predicted calorie range overlaps with the
        true calorie range within a given tolerance.

        Args:
            predicted_indices: (N,) predicted class indices.
            true_indices:      (N,) ground-truth class indices.
            class_names:       Food-101 class name list.
            tolerance_kcal:    Acceptable error margin in kcal.

        Returns:
            Proportion of predictions within tolerance (0–1).
        """
        correct = 0
        for pred_idx, true_idx in zip(predicted_indices, true_indices):
            pred_info = self.estimate(class_names[pred_idx])
            true_info = self.estimate(class_names[true_idx])
            # Check range overlap with tolerance
            pred_min = pred_info["min_cal"] - tolerance_kcal
            pred_max = pred_info["max_cal"] + tolerance_kcal
            true_avg = true_info["avg_cal"]
            if pred_min <= true_avg <= pred_max:
                correct += 1
        return correct / len(predicted_indices)

    def format_result(self, class_name: str) -> str:
        """Human-readable calorie estimate string for display."""
        info = self.estimate(class_name)
        display = class_name.replace("_", " ").title()
        return (
            f"{display}: {info['min_cal']}–{info['max_cal']} kcal "
            f"({info['unit']}) | {info['per_100g']} kcal/100g"
        )

    def all_classes(self) -> list[str]:
        """Returns all food classes in the database."""
        return list(self._db.keys())
