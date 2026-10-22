"""Parameter count, FLOPs hook, and inference latency helpers."""
from __future__ import annotations
import time
import torch


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters": int(total), "trainable_parameters": int(trainable)}


@torch.no_grad()
def measure_latency_ms(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    warmup_iterations: int = 50,
    measured_iterations: int = 200,
) -> dict[str, float]:
    """Measure batch-level forward latency; reports mean and median milliseconds."""
    if warmup_iterations < 0 or measured_iterations < 1:
        raise ValueError("Invalid iteration counts.")
    model.eval()
    device = input_tensor.device
    for _ in range(warmup_iterations):
        _ = model(input_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    for _ in range(measured_iterations):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(input_tensor)
            end.record()
            torch.cuda.synchronize(device)
            timings.append(float(start.elapsed_time(end)))
        else:
            start_t = time.perf_counter()
            _ = model(input_tensor)
            timings.append((time.perf_counter() - start_t) * 1000.0)
    import numpy as np
    return {
        "mean_latency_ms": float(np.mean(timings)),
        "median_latency_ms": float(np.median(timings)),
        "batch_size": int(input_tensor.shape[0]),
        "warmup_iterations": int(warmup_iterations),
        "measured_iterations": int(measured_iterations),
    }


def estimate_flops_macs(model: torch.nn.Module, input_tensor: torch.Tensor) -> dict[str, int]:
    """Estimate MACs/FLOPs using optional thop package. Install thop separately. """
    try:
        from thop import profile
    except ImportError as exc:
        raise ImportError("Install optional dependency 'thop' to estimate MACs/FLOPs.") from exc
    model.eval()
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    return {"macs": int(macs), "flops_if_2_per_mac": int(2 * macs), "thop_parameters": int(params)}
