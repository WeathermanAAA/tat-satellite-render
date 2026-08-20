"""Small deterministic fixtures for M0 validator-discrimination tests."""

from __future__ import annotations

import numpy as np


def alias_stress_columns(size: int = 1600) -> np.ndarray:
    """A bounded chirp plus Nyquist component that exposes point decimation."""

    if size < 8:
        raise ValueError("alias stress field needs at least eight samples")
    x = np.arange(size, dtype=np.float64) / size
    phase = 2.0 * np.pi * size * (0.08 * x + 0.5 * (0.49 - 0.08) * x * x)
    nyquist = np.where(np.arange(size) % 2 == 0, 1.0, -1.0)
    signal = 0.50 + 0.38 * np.sin(phase) + 0.10 * nyquist
    return np.clip(signal, 0.0, 1.0)


def neutral_edge(
    *, width: int = 64, height: int = 16, edge_column: int = 32, low: float = 0.2, high: float = 0.8
) -> np.ndarray:
    if not 0 < edge_column < width:
        raise ValueError("edge column must lie inside the raster")
    scalar = np.full((height, width), low, dtype=np.float32)
    scalar[:, edge_column:] = high
    return np.repeat(scalar[..., np.newaxis], 3, axis=2)


def ringing_edge_mutant(*, width: int = 64, edge_column: int = 32) -> np.ndarray:
    """A deliberately invalid edge with deterministic under/overshoot."""

    edge = neutral_edge(width=width, height=1, edge_column=edge_column, low=0.0, high=1.0)[
        0, :, 0
    ].astype(np.float64)
    edge[edge_column - 1] = -0.08
    edge[edge_column] = 1.11
    return edge


def deterministic_flat_noise(*, size: int = 4096, seed: int = 20260820) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(loc=0.5, scale=0.025, size=size).astype(np.float64)


def equal_weight_contrast_pair(contrast_ratio: float) -> np.ndarray:
    if contrast_ratio <= 0:
        raise ValueError("contrast ratio must be positive")
    return np.asarray([1.0, float(contrast_ratio)], dtype=np.float64)
