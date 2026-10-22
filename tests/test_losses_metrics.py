import numpy as np
import torch
from losses.segmentation_losses import segmentation_loss, weighted_centerline_loss, distance_loss
from evaluation.metrics import segmentation_metrics
from evaluation.physical_distances import surface_distances_mm
from evaluation.statistical_tests import holm_adjust


def test_segmentation_loss_finite():
    logits = torch.zeros(2, 2, 32, 32)
    target = torch.zeros_like(logits)
    target[:, 0, 8:12, 8:12] = 1
    result = segmentation_loss(logits, target)
    assert torch.isfinite(result["segmentation"])


def test_centerline_and_distance_losses_finite():
    logits = torch.zeros(1, 2, 16, 16)
    ctr = torch.zeros_like(logits)
    ctr[:, :, 7:9, 7:9] = 1
    assert torch.isfinite(weighted_centerline_loss(logits, ctr)["centerline"])
    assert torch.isfinite(distance_loss(torch.zeros_like(ctr), torch.ones_like(ctr)))


def test_overlap_and_surface_metrics():
    a = np.zeros((32, 32), dtype=bool)
    a[10:15, 10:15] = True
    m = segmentation_metrics(a.astype(np.uint8), a.astype(np.uint8))
    assert m["dsc"] == 1.0 and m["iou"] == 1.0
    d = surface_distances_mm(a, a, 0.5, 0.5)
    assert d["asd_mm"] == 0.0 and d["hd95_mm"] == 0.0


def test_holm_adjustment():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.all(adjusted >= np.array([0.01, 0.04, 0.03]))
    assert np.all(adjusted <= 1.0)


def test_structural_target_generation_shapes():
    from datasets.target_generation import make_structural_targets
    mask = np.zeros((320, 320), dtype=np.uint8)
    mask[80:180, 120:130] = 1
    mask[100:220, 180:190] = 2
    ctr, dist = make_structural_targets(mask, (40, 40))
    assert ctr.shape == (2, 40, 40)
    assert dist.shape == (2, 40, 40)
    assert np.isfinite(ctr).all() and np.isfinite(dist).all()
    assert dist.min() >= 0 and dist.max() <= 1

