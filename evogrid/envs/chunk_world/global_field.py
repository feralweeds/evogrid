"""Order-independent global-coordinate random fields."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from evogrid.envs.map_generation.seeding import derive_seed


class GlobalCoordinateField:
    """Sample smooth deterministic fields at arbitrary integer world cells."""

    def __init__(self, root_seed: int, base_period: float = 32.0, octaves: int = 5):
        if base_period <= 0:
            raise ValueError("base_period must be positive")
        if octaves <= 0:
            raise ValueError("octaves must be positive")
        self.root_seed = int(root_seed)
        self.base_period = float(base_period)
        self.octaves = int(octaves)

    def sample_grid(self, xs: Iterable[int], ys: Iterable[int], *, channel: str, hurst: float) -> np.ndarray:
        xs_array = np.asarray(list(xs), dtype=np.float64)
        ys_array = np.asarray(list(ys), dtype=np.float64)
        out = np.zeros((ys_array.size, xs_array.size), dtype=np.float64)
        weight_total = 0.0
        for octave in range(self.octaves):
            period = self.base_period / (2**octave)
            period = max(1.0, period)
            amplitude = (2.0 ** (-float(hurst) * octave))
            out += amplitude * self._value_noise(xs_array, ys_array, channel, octave, period)
            weight_total += amplitude
        if weight_total:
            out /= weight_total
        return np.clip((out + 1.0) / 2.0, 0.0, 1.0)

    def sample_points(self, points: Iterable[tuple[int, int]], *, channel: str, hurst: float) -> dict[tuple[int, int], float]:
        return {
            (int(x), int(y)): float(self.sample_grid([int(x)], [int(y)], channel=channel, hurst=hurst)[0, 0])
            for x, y in points
        }

    def _value_noise(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        channel: str,
        octave: int,
        period: float,
    ) -> np.ndarray:
        gx = xs / period
        gy = ys / period
        x0 = np.floor(gx).astype(np.int64)
        y0 = np.floor(gy).astype(np.int64)
        x1 = x0 + 1
        y1 = y0 + 1
        tx = _smoothstep(gx - x0)
        ty = _smoothstep(gy - y0)

        n00 = self._lattice_values(x0, y0, channel, octave)
        n10 = self._lattice_values(x1, y0, channel, octave)
        n01 = self._lattice_values(x0, y1, channel, octave)
        n11 = self._lattice_values(x1, y1, channel, octave)
        nx0 = n00 * (1.0 - tx) + n10 * tx
        nx1 = n01 * (1.0 - tx) + n11 * tx
        return nx0 * (1.0 - ty[:, np.newaxis]) + nx1 * ty[:, np.newaxis]

    def _lattice_values(self, xs: np.ndarray, ys: np.ndarray, channel: str, octave: int) -> np.ndarray:
        values = np.empty((ys.size, xs.size), dtype=np.float64)
        for row, y in enumerate(ys):
            for col, x in enumerate(xs):
                values[row, col] = _seed_to_unit_signed(derive_seed(self.root_seed, "global_field", channel, octave, int(x), int(y)))
        return values


def _smoothstep(value: np.ndarray) -> np.ndarray:
    return value * value * (3.0 - 2.0 * value)


def _seed_to_unit_signed(seed: int) -> float:
    unit = (int(seed) & ((1 << 53) - 1)) / float((1 << 53) - 1)
    if math.isclose(unit, 0.0):
        return -1.0
    return unit * 2.0 - 1.0
