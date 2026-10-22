"""ARS-Net segmentation, centerline, and distance losses.

Manuscript-supported objective:
    L_seg = L_Dice + L_BCE
    L_ctr = L_WBCE + L_Dice
    L_dist = SmoothL1(D, D_hat)
    L_total = L_seg + lambda_c * L_ctr + lambda_d * L_dist
with lambda_c=0.3 and lambda_d=0.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import torch
from torch import nn
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    include_background: bool = True,
) -> torch.Tensor:
    """Mean soft Dice loss over batch and channels.
    Args:
        logits: raw logits [B,C,H,W].
        targets: binary targets [B,C,H,W].
    """
    if logits.shape != targets.shape:
        raise ValueError(f"Dice shape mismatch: {logits.shape} vs {targets.shape}")
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def segmentation_loss(
    segmentation_logits: torch.Tensor,
    segmentation_targets: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Dice + BCE segmentation objective for the two ligament channels."""
    if segmentation_logits.shape != segmentation_targets.shape:
        raise ValueError(
            f"Segmentation shape mismatch: {segmentation_logits.shape} "
            f"vs {segmentation_targets.shape}"
        )
    bce = F.binary_cross_entropy_with_logits(
        segmentation_logits, segmentation_targets.float()
    )
    dice = soft_dice_loss(segmentation_logits, segmentation_targets.float())
    total = bce_weight * bce + dice_weight * dice
    return {"segmentation": total, "seg_bce": bce, "seg_dice": dice}


def weighted_centerline_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: float = 5.0,
) -> Dict[str, torch.Tensor]:
    """Weighted BCE + Dice for sparse centerline targets."""
    if logits.shape != targets.shape:
        raise ValueError(f"Centerline shape mismatch: {logits.shape} vs {targets.shape}")
    pos_weight = torch.as_tensor(
        positive_weight, device=logits.device, dtype=logits.dtype
    )
    wbce = F.binary_cross_entropy_with_logits(
        logits, targets.float(), pos_weight=pos_weight
    )
    dice = soft_dice_loss(logits, targets.float())
    return {"centerline": wbce + dice, "centerline_wbce": wbce, "centerline_dice": dice}


def distance_loss(
    predicted_distance: torch.Tensor,
    target_distance: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth L1 loss for normalized continuous distance maps."""
    if predicted_distance.shape != target_distance.shape:
        raise ValueError(
            f"Distance shape mismatch: {predicted_distance.shape} "
            f"vs {target_distance.shape}"
        )
    return F.smooth_l1_loss(
        predicted_distance, target_distance.float(), beta=beta, reduction="mean"
    )


@dataclass
class LossWeights:
    centerline: float = 0.3
    distance: float = 0.2
    segmentation_bce: float = 1.0
    segmentation_dice: float = 1.0
    centerline_positive_weight: float = 5.0


class ARSNetLoss(nn.Module):
    """Composite objective for ARS-Net.

    model output keys:
      segmentation_logits: [B,2,H,W]
      acl_centerline_logits / pcl_centerline_logits: [B,1,h,w]
      acl_distance / pcl_distance: [B,1,h,w]

    target keys:
      mask: [B,2,H,W]
      centerline: [B,2,h,w]
      distance: [B,2,h,w]
    """
    def __init__(self, weights: LossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or LossWeights()

    def forward(
        self,
        outputs: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        seg = segmentation_loss(
            outputs["segmentation_logits"],
            targets["mask"],
            bce_weight=self.weights.segmentation_bce,
            dice_weight=self.weights.segmentation_dice,
        )
        ctr_logits = torch.cat(
            (outputs["acl_centerline_logits"], outputs["pcl_centerline_logits"]), dim=1
        )
        dist_pred = torch.cat((outputs["acl_distance"], outputs["pcl_distance"]), dim=1)

        ctr = weighted_centerline_loss(
            ctr_logits, targets["centerline"],
            positive_weight=self.weights.centerline_positive_weight,
        )
        dist = distance_loss(dist_pred, targets["distance"])

        total = (
            seg["segmentation"]
            + self.weights.centerline * ctr["centerline"]
            + self.weights.distance * dist
        )
        return {
            "total": total,
            **seg,
            **ctr,
            "distance": dist,
        }
