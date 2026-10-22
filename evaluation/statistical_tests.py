"""Subject-level bootstrap confidence intervals and paired tests."""
from __future__ import annotations

from typing import Iterable
import numpy as np
from scipy.stats import wilcoxon


def bootstrap_mean_ci(
    values: Iterable[float],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return observed mean and percentile bootstrap CI."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("No finite observations supplied.")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_bootstrap, x.size))
    boot_means = x[idx].mean(axis=1)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_wilcoxon(
    proposed: Iterable[float],
    comparator: Iterable[float],
) -> dict[str, float]:
    """Two-sided paired Wilcoxon signed-rank test and median paired difference."""
    a = np.asarray(list(proposed), dtype=float)
    b = np.asarray(list(comparator), dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired arrays must have the same shape and ordering.")
    keep = np.isfinite(a) & np.isfinite(b)
    delta = a[keep] - b[keep]
    if delta.size == 0:
        raise ValueError("No finite paired observations.")
    if np.allclose(delta, 0):
        return {"median_delta": 0.0, "p_raw": 1.0, "n": int(delta.size)}
    test = wilcoxon(a[keep], b[keep], alternative="two-sided", zero_method="wilcox")
    return {
        "median_delta": float(np.median(delta)),
        "p_raw": float(test.pvalue),
        "n": int(delta.size),
    }


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values in original input order."""
    p = np.asarray(list(p_values), dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("All p-values must be in [0,1].")
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p[idx]
        running = max(running, candidate)
        adjusted_sorted[rank] = min(running, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted
