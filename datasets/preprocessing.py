"""Image/mask preprocessing and paired augmentation for ARS-Net."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale image at {path}, got {arr.shape}")
    return arr.astype(np.float32)


def read_mask(path: str | Path) -> np.ndarray:
    """Read a categorical mask where 0=background, 1=ACL, 2=PCL."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise FileNotFoundError(f"Could not read mask: {path}")
        if arr.ndim == 3:
            # Do not silently interpret color-coded masks as class IDs.
            raise ValueError(
                f"Mask {path} is multi-channel. Convert it to integer labels "
                "0=background, 1=ACL, 2=PCL before use."
            )
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D categorical mask at {path}, got {arr.shape}")
    return arr.astype(np.int64)


def resize_pair(
    image: np.ndarray,
    mask: np.ndarray,
    size: Tuple[int, int] = (320, 320),
) -> tuple[np.ndarray, np.ndarray]:
    """Resize image bilinearly and categorical mask with nearest-neighbor."""
    h, w = size
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return image.astype(np.float32), mask.astype(np.int64)


def zscore_normalize(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-slice z-score normalization"""
    image = image.astype(np.float32)
    mean = float(image.mean())
    std = float(image.std())
    if std < eps:
        return np.zeros_like(image, dtype=np.float32)
    return (image - mean) / (std + eps)


def paired_augment(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    rotation_degrees: float = 10.0,
    translation_fraction: float = 0.05,
    scale_range: tuple[float, float] = (0.9, 1.1),
    brightness_fraction: float = 0.1,
    contrast_fraction: float = 0.1,
    gamma_range: tuple[float, float] = (0.9, 1.1),
    gaussian_noise_std: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply paired geometric and image-only intensity augmentations."""
    h, w = image.shape
    angle = float(rng.uniform(-rotation_degrees, rotation_degrees))
    scale = float(rng.uniform(*scale_range))
    tx = float(rng.uniform(-translation_fraction, translation_fraction) * w)
    ty = float(rng.uniform(-translation_fraction, translation_fraction) * h)
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty

    image = cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    mask = cv2.warpAffine(
        mask.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    ).astype(np.int64)

    # Intensity transforms are applied after geometry and before z-score.
    contrast = float(rng.uniform(1.0 - contrast_fraction, 1.0 + contrast_fraction))
    brightness = float(rng.uniform(-brightness_fraction, brightness_fraction))
    image = image * contrast + brightness * (float(np.std(image)) + 1e-6)
    gamma = float(rng.uniform(*gamma_range))
    lo, hi = float(image.min()), float(image.max())
    scaled = (image - lo) / (hi - lo + 1e-6)
    image = np.power(np.clip(scaled, 0.0, 1.0), gamma).astype(np.float32)
    if gaussian_noise_std > 0:
        image = image + rng.normal(0.0, gaussian_noise_std, image.shape).astype(np.float32)
    return image.astype(np.float32), mask
