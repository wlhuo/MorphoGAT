"""Barcode-aligned RGB patch descriptors from the manuscript."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

FEATURE_NAMES = (
    [f"{c}_{stat}" for stat in ("mean", "std") for c in "rgb"]
    + [f"{c}_p{q}" for q in (10, 50, 90) for c in "rgb"]
    + ["gray_gradient_mean", "gray_dark_fraction"]
    + [f"{c}_hist_{b:02d}" for c in "rgb" for b in range(16)]
)


def describe_patch(patch):
    rgb = np.asarray(patch, dtype=np.float32) / 255.0
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2:
        raise ValueError("Expected an RGB patch with both dimensions >= 2")
    pixels = rgb.reshape(-1, 3)
    gray = rgb.mean(axis=2)
    dy, dx = np.gradient(gray)
    summary = [pixels.mean(axis=0), pixels.std(axis=0)]
    summary += [np.percentile(pixels, q, axis=0) for q in (10, 50, 90)]
    summary += [np.array([np.hypot(dx, dy).mean(), (gray < 0.92).mean()])]
    summary += [np.histogram(pixels[:, c], bins=16, range=(0, 1))[0] / len(pixels)
                for c in range(3)]
    return np.concatenate(summary).astype(np.float32)


def read_positions(path):
    """Read headered Visium/ordinary positions or legacy six-column Visium CSV."""
    frame = pd.read_csv(path, dtype={"barcode": str})
    if "barcode" not in frame.columns:
        frame = pd.read_csv(path, header=None, dtype={"barcode": str},
                            names=["barcode", "in_tissue", "array_row", "array_col",
                                   "pxl_row_in_fullres", "pxl_col_in_fullres"])
    if frame.barcode.isna().any() or frame.barcode.duplicated().any():
        raise ValueError(f"Missing or duplicate position barcodes: {path}")
    frame.barcode = frame.barcode.astype(str)
    frame = frame.set_index("barcode")
    if {"pxl_col_in_fullres", "pxl_row_in_fullres"} <= set(frame.columns):
        return frame[["pxl_col_in_fullres", "pxl_row_in_fullres"]].rename(
            columns={"pxl_col_in_fullres": "x", "pxl_row_in_fullres": "y"})
    if {"x", "y"} <= set(frame.columns):
        return frame[["x", "y"]]
    raise ValueError(f"Position table needs pixel x,y or Visium full-resolution columns: {path}")


def extract_descriptors(image_path, centers, patch_size=64):
    """Centers are (x, y) in the selected image's pixels; invalid rows are masked."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("Image centers must have shape (spots, 2)")
    with Image.open(Path(image_path)) as source:
        image = source.convert("RGB")
    width, height = image.size
    valid = (np.isfinite(centers).all(axis=1) & (centers[:, 0] >= 0)
             & (centers[:, 0] < width) & (centers[:, 1] >= 0)
             & (centers[:, 1] < height))
    features = np.zeros((len(centers), 65), dtype=np.float32)
    for i in np.flatnonzero(valid):
        x, y = np.floor(centers[i] + 0.5).astype(int)
        left, top = x - patch_size // 2, y - patch_size // 2
        right, bottom = left + patch_size, top + patch_size
        patch = Image.new("RGB", (patch_size, patch_size), "white")
        box = (max(0, left), max(0, top), min(width, right), min(height, bottom))
        patch.paste(image.crop(box), (box[0] - left, box[1] - top))
        features[i] = describe_patch(patch)
    return features, valid
