"""Reusable convolutional blocks for ARS-Net.
This file keeps those blocks separate so their implementation can be audited and tested.
"""
from __future__ import annotations

import torch
from torch import nn


class ConvBNGELU(nn.Sequential):
    """2-D convolution followed by batch normalization and GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class ResidualEncoderStage(nn.Module):
    """Two Conv-BN-GELU layers with a residual projection when needed."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBNGELU(
            in_channels, out_channels, kernel_size=3, stride=stride
        )
        self.conv2 = ConvBNGELU(
            out_channels, out_channels, kernel_size=3, stride=1
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.out_activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_activation(self.conv2(self.conv1(x)) + self.skip(x))


class ConvGNBlock(nn.Module):
    """Convolution, GroupNorm, and GELU used in structural refinement."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        groups: int = 8,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        actual_groups = min(groups, out_channels)
        while out_channels % actual_groups != 0:
            actual_groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding=padding, bias=False
            ),
            nn.GroupNorm(actual_groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
