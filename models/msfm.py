"""Morphology-guided Structural Feedback Module (MSFM)."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ConvBNGELU, ConvGNBlock


class MorphologyGuidedStructuralFeedbackModule(nn.Module):
    """Predict region/centerline/distance cues and feed them into features.
    The six structural maps are produced at E3 resolution (40x40 for 320x320
    inputs), then concatenated with the structural feature and refined through
    a residual feedback block. Auxiliary outputs are logits except distance
    outputs, which are sigmoid-bounded to [0,1].
    """

    def __init__(
        self,
        relational_channels: int = 128,
        encoder_channels: int = 128,
        structural_channels: int = 128,
    ) -> None:
        super().__init__()
        self.structural_projection = ConvBNGELU(
            relational_channels + encoder_channels,
            structural_channels,
            kernel_size=1,
        )
        # Per ligament: region, centerline, and normalized distance prediction.
        self.acl_region = nn.Conv2d(structural_channels, 1, kernel_size=1)
        self.acl_centerline = nn.Conv2d(structural_channels, 1, kernel_size=1)
        self.acl_distance = nn.Conv2d(structural_channels, 1, kernel_size=1)
        self.pcl_region = nn.Conv2d(structural_channels, 1, kernel_size=1)
        self.pcl_centerline = nn.Conv2d(structural_channels, 1, kernel_size=1)
        self.pcl_distance = nn.Conv2d(structural_channels, 1, kernel_size=1)

        self.feedback_projection = ConvGNBlock(
            structural_channels + 6, structural_channels, kernel_size=1
        )
        self.feedback_local1 = ConvGNBlock(
            structural_channels, structural_channels, kernel_size=3
        )
        self.feedback_local2 = ConvGNBlock(
            structural_channels, structural_channels, kernel_size=3
        )

    def forward(
        self,
        relational_features: torch.Tensor,
        e3: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        relational_up = F.interpolate(
            relational_features, size=e3.shape[-2:], mode="bilinear",
            align_corners=False
        )
        structural = self.structural_projection(
            torch.cat((relational_up, e3), dim=1)
        )

        acl_region_logits = self.acl_region(structural)
        acl_centerline_logits = self.acl_centerline(structural)
        acl_distance = torch.sigmoid(self.acl_distance(structural))
        pcl_region_logits = self.pcl_region(structural)
        pcl_centerline_logits = self.pcl_centerline(structural)
        pcl_distance = torch.sigmoid(self.pcl_distance(structural))

        # Use probabilities for region/centerline feedback and bounded values
        # for distance feedback; preserve logits separately for supervision.
        structural_maps = torch.cat(
            (
                torch.sigmoid(acl_region_logits),
                torch.sigmoid(acl_centerline_logits),
                acl_distance,
                torch.sigmoid(pcl_region_logits),
                torch.sigmoid(pcl_centerline_logits),
                pcl_distance,
            ),
            dim=1,
        )
        feedback_input = torch.cat((structural, structural_maps), dim=1)
        update = self.feedback_local2(
            self.feedback_local1(self.feedback_projection(feedback_input))
        )
        refined = structural + update

        aux = {
            "acl_region_logits": acl_region_logits,
            "acl_centerline_logits": acl_centerline_logits,
            "acl_distance": acl_distance,
            "pcl_region_logits": pcl_region_logits,
            "pcl_centerline_logits": pcl_centerline_logits,
            "pcl_distance": pcl_distance,
        }
        return refined, aux
