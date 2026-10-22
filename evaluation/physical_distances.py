"""ASD and HD95 in physical units using original-image spacing."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    return np.logical_xor(mask, eroded)


def surface_distances_mm(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_y_mm: float,
    spacing_x_mm: float,
) -> dict[str, float]:
    """Compute symmetric ASD and HD95 for 2-D masks. """
    if spacing_y_mm <= 0 or spacing_x_mm <= 0:
        raise ValueError("Pixel spacing must be positive.")
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(
            f"Physical distance masks must share the same grid: {pred.shape} vs {gt.shape}"
        )
    if not pred.any() and not gt.any():
        return {"asd_mm": 0.0, "hd95_mm": 0.0}
    if not pred.any() or not gt.any():
        return {"asd_mm": float("nan"), "hd95_mm": float("nan")}

    pred_surface = _surface(pred)
    gt_surface = _surface(gt)
    distance_to_gt = ndimage.distance_transform_edt(~gt_surface, sampling=(spacing_y_mm, spacing_x_mm))
    distance_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=(spacing_y_mm, spacing_x_mm))
    d_pred_gt = distance_to_gt[pred_surface]
    d_gt_pred = distance_to_pred[gt_surface]
    both = np.concatenate((d_pred_gt, d_gt_pred))
    return {
        "asd_mm": float((d_pred_gt.sum() + d_gt_pred.sum()) / (len(d_pred_gt) + len(d_gt_pred))),
        "hd95_mm": float(np.percentile(both, 95)),
    }
