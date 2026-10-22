"""Anatomical Relational Reasoning Module (ARRM)."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GeometryBias(nn.Module):
    """Map relative (dx, dy, distance) descriptors to per-head attention bias."""

    def __init__(self, hidden_dim: int = 32, num_heads: int = 4) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

    @staticmethod
    def relative_geometry(
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Coordinates are normalized to [0, 1].
        ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
        delta = coords[:, None, :] - coords[None, :, :]
        distance = torch.sqrt(torch.clamp((delta ** 2).sum(dim=-1), min=1e-12))
        return torch.cat((delta, distance.unsqueeze(-1)), dim=-1)

    def forward(self, height: int, width: int, device: torch.device, dtype: torch.dtype):
        rel = self.relative_geometry(height, width, device, dtype)
        # [N, N, heads] -> [heads, N, N]
        return self.mlp(rel).permute(2, 0, 1).contiguous()


class DirectedCrossAttention(nn.Module):
    """Multi-head cross-attention with a geometry-dependent additive bias."""

    def __init__(
        self,
        channels: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        geometry_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(channels, channels)
        self.k = nn.Linear(channels, channels)
        self.v = nn.Linear(channels, channels)
        self.out = nn.Linear(channels, channels)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.geometry_bias = GeometryBias(geometry_hidden_dim, num_heads)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query_map: torch.Tensor,
        key_value_map: torch.Tensor,
        reverse_geometry: bool = False,
    ) -> torch.Tensor:
        batch, channels, height, width = query_map.shape
        q_tokens = query_map.flatten(2).transpose(1, 2)
        kv_tokens = key_value_map.flatten(2).transpose(1, 2)

        q = self._split_heads(self.q(q_tokens))
        k = self._split_heads(self.k(kv_tokens))
        v = self._split_heads(self.v(kv_tokens))

        bias = self.geometry_bias(height, width, query_map.device, q.dtype)
        if reverse_geometry:
            # Reverse query/key order: signed offsets change sign, distance does not.
            bias = self.geometry_bias(height, width, query_map.device, q.dtype)
            # Recompute using the reversed descriptor, not merely transposing a
            # precomputed bias, because the learned MLP need not be sign-symmetric.
            rel = GeometryBias.relative_geometry(
                height, width, query_map.device, q.dtype
            )
            rel = rel.transpose(0, 1)
            bias = self.geometry_bias.mlp(rel).permute(2, 0, 1).contiguous()

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        logits = logits + bias.unsqueeze(0)
        weights = self.attn_dropout(torch.softmax(logits, dim=-1))
        response = torch.matmul(weights, v)
        response = response.transpose(1, 2).reshape(batch, -1, channels)
        response = self.proj_dropout(self.out(response))
        return response.transpose(1, 2).reshape(batch, channels, height, width)


class RelationalBranch(nn.Module):
    """Residual relational update followed by a feed-forward refinement."""

    def __init__(self, channels: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attention = DirectedCrossAttention(
            channels=channels, num_heads=num_heads, dropout=dropout
        )
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm1 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(channels)

    @staticmethod
    def _layer_norm_map(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(
        self,
        own_features: torch.Tensor,
        other_features: torch.Tensor,
        reverse_geometry: bool = False,
    ) -> torch.Tensor:
        relation = self.cross_attention(
            own_features, other_features, reverse_geometry=reverse_geometry
        )
        x = self._layer_norm_map(self.norm1, own_features + self.gamma * relation)
        tokens = x.flatten(2).transpose(1, 2)
        x = x + self.ffn(tokens).transpose(1, 2).reshape_as(x)
        return self._layer_norm_map(self.norm2, x)


class AnatomicalRelationalReasoningModule(nn.Module):
    """Bidirectional ACL/PCL cross-attention with continuous geometric bias.

      input: E4 [B,256,20,20]
      output: FR [B,128,20,20]
    """

    def __init__(
        self,
        in_channels: int = 256,
        channels: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.acl_projection = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.pcl_projection = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.acl_branch = RelationalBranch(channels, num_heads, dropout)
        self.pcl_branch = RelationalBranch(channels, num_heads, dropout)
        self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, e4: torch.Tensor) -> torch.Tensor:
        acl = self.acl_projection(e4)
        pcl = self.pcl_projection(e4)
        # ACL queries PCL keys/values; reverse branch exchanges query/key roles.
        acl_refined = self.acl_branch(acl, pcl, reverse_geometry=False)
        pcl_refined = self.pcl_branch(pcl, acl, reverse_geometry=True)
        return self.fusion(torch.cat((acl_refined, pcl_refined), dim=1))
