"""
Maps Food-101 class labels to calorie and macro data from the nutritional database.
"""

import os
import json
import numpy as np
from pathlib import Path


# Default path to the calorie database JSON (relative to project root)
_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", "..", "data", "calorie_data.json")


class CalorieEstimator:
    """Looks up calorie and macro data for Food-101 class names."""

    def __init__(self, db_path: str | None = None):
        path = db_path or _DEFAULT_DB
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calorie database not found: {path}")
        with open(path, "r") as f:
            self._db: dict = json.load(f)

    def estimate(self, class_name: str) -> dict:
        key = class_name.lower().replace(" ", "_")
        if key in self._db:
            return self._db[key]
        # prefix fallback for minor name mismatches
        for db_key in self._db:
            if db_key.startswith(key[:4]):
                return self._db[db_key]
        return {"min_cal": 0, "max_cal": 0, "avg_cal": 0, "per_100g": 0, "unit": "unknown"}

    def estimate_for_weight(self, class_name: str, weight_g: float) -> dict:
        """Interpolates nutrition values for a given weight in grams."""
        info = self.estimate(class_name)
        portions = info.get("portions", [])
        if not portions:
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
        return self.estimate(class_names[class_idx])

    def bulk_estimate(
        self,
        predicted_indices: np.ndarray,
        class_names: list[str],
    ) -> np.ndarray:
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
        """Returns mean absolute calorie error in kcal."""
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
        """Fraction of predictions whose calorie range overlaps true avg within tolerance."""
        correct = 0
        for pred_idx, true_idx in zip(predicted_indices, true_indices):
            pred_info = self.estimate(class_names[pred_idx])
            true_info = self.estimate(class_names[true_idx])
            pred_min = pred_info["min_cal"] - tolerance_kcal
            pred_max = pred_info["max_cal"] + tolerance_kcal
            true_avg = true_info["avg_cal"]
            if pred_min <= true_avg <= pred_max:
                correct += 1
        return correct / len(predicted_indices)

    def format_result(self, class_name: str) -> str:
        info = self.estimate(class_name)
        display = class_name.replace("_", " ").title()
        return (
            f"{display}: {info['min_cal']}–{info['max_cal']} kcal "
            f"({info['unit']}) | {info['per_100g']} kcal/100g"
        )

    def all_classes(self) -> list[str]:
        return list(self._db.keys())
