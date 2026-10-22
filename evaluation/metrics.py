"""Overlap metrics for binary or two-channel ligament segmentation."""
from __future__ import annotations

import numpy as np
import torch


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def segmentation_metrics(
    prediction,
    target,
    threshold: float = 0.5,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute per-class DSC and IoU for one image.

    prediction and target may be [C,H,W] or [H,W] categorical labels.
    """
    pred = _as_numpy(prediction)
    gt = _as_numpy(target)
    if pred.ndim == 2 and gt.ndim == 2:
        labels = set(np.unique(gt).tolist()) | set(np.unique(pred).tolist())
        if labels.issubset({0, 1}):
            return {
                "dsc": _binary_dsc(pred > 0, gt > 0, eps),
                "iou": _binary_iou(pred > 0, gt > 0, eps),
            }
        classes = sorted(c for c in labels if c != 0)
        if not classes:
            return {"dsc": 1.0, "iou": 1.0}
        out = {}
        for c in classes:
            p, g = pred == c, gt == c
            out[f"dsc_class_{c}"] = _binary_dsc(p, g, eps)
            out[f"iou_class_{c}"] = _binary_iou(p, g, eps)
        return out
    if pred.ndim == 2:
        pred = pred[None]
    if gt.ndim == 2:
        gt = gt[None]
    if pred.shape != gt.shape:
        raise ValueError(f"Metric shape mismatch: {pred.shape} vs {gt.shape}")
    if pred.min(initial=0) < 0 or pred.max(initial=0) > 1:
        pred = 1.0 / (1.0 + np.exp(-pred))
    result = {}
    for c in range(pred.shape[0]):
        p = pred[c] >= threshold
        g = gt[c] >= 0.5
        result[f"dsc_{c}"] = _binary_dsc(p, g, eps)
        result[f"iou_{c}"] = _binary_iou(p, g, eps)
    return result


def _binary_dsc(p: np.ndarray, g: np.ndarray, eps: float) -> float:
    p, g = p.astype(bool), g.astype(bool)
    if np.array_equal(p, g):
        return 1.0
    if not p.any() and not g.any():
        return 1.0
    return float(2 * np.logical_and(p, g).sum() / (p.sum() + g.sum() + eps))


def _binary_iou(p: np.ndarray, g: np.ndarray, eps: float) -> float:
    p, g = p.astype(bool), g.astype(bool)
    if np.array_equal(p, g):
        return 1.0
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / (union + eps))
