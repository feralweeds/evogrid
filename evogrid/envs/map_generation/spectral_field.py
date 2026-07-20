"""Finite correlated random fields for fractal/percolation maps."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def generate_rank_normalized_field(
    shape: tuple[int, int] | Sequence[int],
    hurst: float,
    seed: int,
) -> np.ndarray:
    """Generate a reproducible fBm-like 2D field rank-normalized to [0, 1]."""

    height, width = _shape(shape)
    hurst = _hurst(hurst)
    rng = np.random.default_rng(int(seed))
    white = rng.standard_normal((height, width), dtype=np.float64)
    spectrum = np.fft.rfft2(white)

    ky = np.fft.fftfreq(height)[:, np.newaxis]
    kx = np.fft.rfftfreq(width)[np.newaxis, :]
    q2 = kx * kx + ky * ky
    amplitude = np.zeros_like(q2, dtype=np.float64)
    nonzero = q2 > 0.0
    amplitude[nonzero] = q2[nonzero] ** (-(hurst + 1.0) / 2.0)
    spectrum *= amplitude
    spectrum[0, 0] = 0.0

    field = np.fft.irfft2(spectrum, s=(height, width)).real
    return _rank_normalize(field)


def _rank_normalize(field: np.ndarray) -> np.ndarray:
    flat = np.asarray(field, dtype=np.float64).reshape(-1)
    order = np.lexsort((np.arange(flat.size), flat))
    ranks = np.empty(flat.size, dtype=np.float64)
    ranks[order] = np.arange(flat.size, dtype=np.float64)
    if flat.size > 1:
        ranks /= float(flat.size - 1)
    return ranks.reshape(field.shape)


def _shape(shape: tuple[int, int] | Sequence[int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("shape: expected (height, width)")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("shape: dimensions must be positive")
    return height, width


def _hurst(value: float) -> float:
    number = float(value)
    if not (0.0 < number < 1.0):
        raise ValueError("hurst: expected 0 < H < 1")
    return number
