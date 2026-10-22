"""Progressive decoder for ARS-Net."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ConvBNGELU


class ProgressiveDecoder(nn.Module):
    """Bilinear upsampling with encoder skip connections."""

    def __init__(
        self,
        structural_channels: int = 128,
        e2_channels: int = 64,
        e1_channels: int = 32,
        output_channels: int = 2,
    ) -> None:
        super().__init__()
        self.decode_e2 = nn.Sequential(
            ConvBNGELU(structural_channels + e2_channels, e2_channels, 3),
            ConvBNGELU(e2_channels, e2_channels, 3),
        )
        self.decode_e1 = nn.Sequential(
            ConvBNGELU(e2_channels + e1_channels, e1_channels, 3),
            ConvBNGELU(e1_channels, e1_channels, 3),
        )
        self.final_refine = ConvBNGELU(e1_channels, e1_channels, 3)
        self.segmentation_head = nn.Conv2d(e1_channels, output_channels, kernel_size=1)

    def forward(
        self,
        fsr: torch.Tensor,
        e2: torch.Tensor,
        e1: torch.Tensor,
        output_size: tuple[int, int] = (320, 320),
    ) -> torch.Tensor:
        x = F.interpolate(fsr, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode_e2(torch.cat((x, e2), dim=1))
        x = F.interpolate(x, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decode_e1(torch.cat((x, e1), dim=1))
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        x = self.final_refine(x)
        return self.segmentation_head(x)
