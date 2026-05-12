"""
Feature extraction pipeline: HOG features, RGB/HSV color histograms, PCA reduction.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.feature import hog
from skimage import color as skcolor
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm


HOG_ORIENTATIONS  = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

COLOR_BINS = 32


def extract_hog_features(image_np: np.ndarray) -> np.ndarray:
    gray = skcolor.rgb2gray(image_np)
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features


def extract_color_histogram(image_np: np.ndarray) -> np.ndarray:
    """Returns concatenated L2-normalized RGB + HSV histograms."""
    rgb_hist = []
    for ch in range(3):
        hist, _ = np.histogram(image_np[:, :, ch], bins=COLOR_BINS, range=(0, 256))
        rgb_hist.append(hist.astype(np.float32))
    rgb_hist = np.concatenate(rgb_hist)

    bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hsv_hist = []
    ranges = [(0, 180), (0, 256), (0, 256)]
    for ch, (lo, hi) in enumerate(ranges):
        hist, _ = np.histogram(hsv[:, :, ch], bins=COLOR_BINS, range=(lo, hi))
        hsv_hist.append(hist.astype(np.float32))
    hsv_hist = np.concatenate(hsv_hist)

    combined = np.concatenate([rgb_hist, hsv_hist])
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined = combined / norm
    return combined


def tensor_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """Converts a CxHxW float tensor in [0,1] to uint8 HxWxC numpy array."""
    img = img_tensor.permute(1, 2, 0).numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def extract_features_from_loader(
    loader: DataLoader,
    hog_only: bool = False,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    all_features = []
    all_labels   = []

    iterator = tqdm(loader, desc="Extracting features", unit="batch") if verbose else loader

    for imgs, labels in iterator:
        for img_tensor, label in zip(imgs, labels):
            img_np = tensor_to_uint8(img_tensor)

            hog_feat   = extract_hog_features(img_np)
            if hog_only:
                feat = hog_feat
            else:
                color_feat = extract_color_histogram(img_np)
                feat = np.concatenate([hog_feat, color_feat])

            all_features.append(feat)
            all_labels.append(label.item())

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels,   dtype=np.int64)
    return X, y


class FeaturePipeline:
    """Wraps feature extraction, StandardScaler, and PCA into a single reusable pipeline."""

    def __init__(self, n_components: int = 200, hog_only: bool = False):
        self.n_components = n_components
        self.hog_only     = hog_only
        self.scaler       = StandardScaler()
        self.pca          = PCA(n_components=n_components, random_state=42)
        self._fitted      = False

    def fit_transform(self, loader: DataLoader, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        X, y = extract_features_from_loader(loader, hog_only=self.hog_only, verbose=verbose)
        X_scaled = self.scaler.fit_transform(X)
        X_pca    = self.pca.fit_transform(X_scaled)
        self._fitted = True
        explained = self.pca.explained_variance_ratio_.sum()
        print(f"PCA: {self.n_components} components explain {explained:.1%} of variance.")
        return X_pca, y

    def transform(self, loader: DataLoader, verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Call fit_transform() on training data before transform().")
        X, y = extract_features_from_loader(loader, hog_only=self.hog_only, verbose=verbose)
        X_scaled = self.scaler.transform(X)
        X_pca    = self.pca.transform(X_scaled)
        return X_pca, y

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"scaler": self.scaler, "pca": self.pca, "hog_only": self.hog_only}, path)
        print(f"Feature pipeline saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        data = joblib.load(path)
        pipeline = cls(n_components=data["pca"].n_components, hog_only=data["hog_only"])
        pipeline.scaler  = data["scaler"]
        pipeline.pca     = data["pca"]
        pipeline._fitted = True
        return pipeline
