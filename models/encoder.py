"""Multi-scale Feature Encoder for ARS-Net."""
from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvBNGELU, ResidualEncoderStage


class MultiScaleFeatureEncoder(nn.Module):
    """Extract E1-E4 features from a single-channel 320x320 MRI slice.

    feature hierarchy for a 320x320 input:
        E1: [B,  32, 160, 160]
        E2: [B,  64,  80,  80]
        E3: [B, 128,  40,  40]
        E4: [B, 256,  20,  20]
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.stem = ConvBNGELU(in_channels, 32, kernel_size=3, stride=2)
        self.stage1 = ResidualEncoderStage(32, 32, stride=1)
        self.stage2 = ResidualEncoderStage(32, 64, stride=2)
        self.stage3 = ResidualEncoderStage(64, 128, stride=2)
        self.stage4 = ResidualEncoderStage(128, 256, stride=2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"input [B,C,H,W], got shape {tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"ARS-Net expects a single-channel MRI input; got {x.shape[1]} channels")
        x = self.stem(x)
        e1 = self.stage1(x)
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)
        return {"E1": e1, "E2": e2, "E3": e3, "E4": e4}
