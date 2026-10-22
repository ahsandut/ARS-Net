"""Complete ARS-Net model."""
from __future__ import annotations

import torch
from torch import nn

from .encoder import MultiScaleFeatureEncoder
from .arrm import AnatomicalRelationalReasoningModule
from .msfm import MorphologyGuidedStructuralFeedbackModule
from .decoder import ProgressiveDecoder


class ARSNet(nn.Module):
    """ARS-Net for joint ACL/PCL segmentation of sagittal knee MRI.

    Input:
        image: [B, 1, 320, 320]
    Output dictionary:
        segmentation_logits: [B, 2, 320, 320]
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 2,
        input_size: int = 320,
        dropout: float = 0.1,
        num_heads: int = 4,
        structural_channels: int = 128,
        architecture_variant: str = "full",
    ) -> None:
        super().__init__()
        if input_channels != 1:
            raise ValueError("single-channel MRI input.")
        if num_classes != 2:
            raise ValueError("ARS-Net outputs two channels: ACL and PCL.")
        if input_size % 16 != 0:
            raise ValueError("input_size must be divisible by 16 for the specified encoder hierarchy.")

        valid_variants = {"full", "no_arrm", "no_msfm", "no_arrm_no_msfm"}
        if architecture_variant not in valid_variants:
            raise ValueError(f"architecture_variant must be one of {sorted(valid_variants)}")
        self.input_size = input_size
        self.architecture_variant = architecture_variant
        self.encoder = MultiScaleFeatureEncoder(input_channels)
        self.arrm = AnatomicalRelationalReasoningModule(
            in_channels=256, channels=128, num_heads=num_heads, dropout=dropout
        ) if architecture_variant not in {"no_arrm", "no_arrm_no_msfm"} else None
        self.e4_projection = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128), nn.GELU(),
        )
        self.msfm = MorphologyGuidedStructuralFeedbackModule(
            relational_channels=128,
            encoder_channels=128,
            structural_channels=structural_channels,
        ) if architecture_variant not in {"no_msfm", "no_arrm_no_msfm"} else None
        # Bypass adapter used when MSFM is removed. It fuses the relational/E4
        # representation with E3 but does not predict or feed back structural maps.
        self.no_msfm_fusion = nn.Sequential(
            nn.Conv2d(256, structural_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(structural_channels), nn.GELU(),
            nn.Conv2d(structural_channels, structural_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(structural_channels), nn.GELU(),
        )
        # Auxiliary heads are retained for compatible output dictionaries in
        # no-M SFM variants; they do not feed their predictions back into features.
        self.aux_heads = nn.ModuleDict({
            name: nn.Conv2d(structural_channels, 1, kernel_size=1)
            for name in ("acl_region_logits", "acl_centerline_logits", "acl_distance",
                         "pcl_region_logits", "pcl_centerline_logits", "pcl_distance")
        })
        self.decoder = ProgressiveDecoder(
            structural_channels=structural_channels,
            e2_channels=64,
            e1_channels=32,
            output_channels=num_classes,
        )

    def forward(self, image: torch.Tensor) -> dict[str, object]:
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(
                f"Expected input [B,1,H,W], received {tuple(image.shape)}"
            )
        if image.shape[-2] != self.input_size or image.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size}x{self.input_size} input, "
                f"received {image.shape[-2]}x{image.shape[-1]}"
            )

        features = self.encoder(image)
        if self.arrm is None:
            relational = self.e4_projection(features["E4"])
        else:
            relational = self.arrm(features["E4"])

        if self.msfm is not None:
            refined, auxiliary = self.msfm(relational, features["E3"])
        else:
            relational_up = torch.nn.functional.interpolate(
                relational, size=features["E3"].shape[-2:], mode="bilinear", align_corners=False
            )
            refined = self.no_msfm_fusion(torch.cat((relational_up, features["E3"]), dim=1))
            auxiliary = {}
            for name, head in self.aux_heads.items():
                value = head(refined)
                auxiliary[name] = torch.sigmoid(value) if name.endswith("_distance") else value

        segmentation_logits = self.decoder(
            refined, features["E2"], features["E1"],
            output_size=(self.input_size, self.input_size),
        )
        return {
            "segmentation_logits": segmentation_logits,
            **auxiliary,
            "encoder_features": features,
            "relational_features": relational,
            "refined_structural_features": refined,
        }
