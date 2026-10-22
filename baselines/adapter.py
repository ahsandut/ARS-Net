"""Adapter utilities for integrating external baseline segmentation models.

"""
from __future__ import annotations
import importlib
from typing import Any
import torch
from torch import nn


def load_factory(spec: str, kwargs: dict[str, Any] | None = None) -> nn.Module:
    """Load a factory from 'python.module:callable' and instantiate it."""
    if ":" not in spec:
        raise ValueError("Factory must use 'python.module:callable' syntax")
    module_name, attr = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, attr)
    model = factory(**(kwargs or {}))
    if not isinstance(model, nn.Module):
        raise TypeError(f"Factory {spec!r} did not return torch.nn.Module")
    return model


def segmentation_logits(output: Any) -> torch.Tensor:
    """Normalize supported model outputs to [B,2,H,W] logits."""
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, dict):
        logits = output.get("segmentation_logits", output.get("logits"))
        if logits is None:
            raise KeyError("Model output dict needs 'segmentation_logits' or 'logits'")
    elif isinstance(output, (tuple, list)) and output:
        logits = output[0]
    else:
        raise TypeError(f"Unsupported model output type: {type(output).__name__}")
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("Baseline must return logits with shape [B,2,H,W]")
    return logits
