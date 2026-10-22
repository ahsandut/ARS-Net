"""Generate auxiliary structural targets from categorical ACL/PCL masks."""
from __future__ import annotations
import numpy as np
import cv2
from skimage.morphology import skeletonize


def make_structural_targets(
    categorical_mask: np.ndarray,
    output_size: tuple[int, int] = (40, 40),
) -> tuple[np.ndarray, np.ndarray]:
    """Return centerline and distance targets shaped [2,H,W], float32.

    categorical_mask: [H,W], labels 0=background, 1=ACL, 2=PCL.
    Centerlines are skeletonized after nearest-neighbor resizing to the target
    resolution. Distances are interior EDTs normalized independently to [0,1]
    for each ligament.
    """
    out_h, out_w = output_size
    small = cv2.resize(
        categorical_mask.astype(np.uint8), (out_w, out_h),
        interpolation=cv2.INTER_NEAREST
    )
    centerlines, distances = [], []
    from scipy import ndimage
    for label in (1, 2):
        region = small == label
        skeleton = skeletonize(region)
        centerlines.append(skeleton.astype(np.float32))
        if region.any():
            d = ndimage.distance_transform_edt(region).astype(np.float32)
            maximum = float(d.max())
            if maximum > 0:
                d /= maximum
        else:
            d = np.zeros_like(region, dtype=np.float32)
        distances.append(d)
    return np.stack(centerlines, axis=0), np.stack(distances, axis=0)
