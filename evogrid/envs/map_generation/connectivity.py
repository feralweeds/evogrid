"""Topology masks and four-neighbor connectivity diagnostics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from evogrid.envs.map_generation.schemas import MapGenerationConfig, TopologyConfig
from evogrid.envs.map_generation.spectral_field import generate_rank_normalized_field


@dataclass(frozen=True)
class ComponentIndex:
    labels: np.ndarray
    component_sizes: dict[int, int]
    spans_horizontal: set[int]
    spans_vertical: set[int]

    @property
    def component_count(self) -> int:
        return len(self.component_sizes)

    @property
    def largest_component_id(self) -> int | None:
        if not self.component_sizes:
            return None
        return max(self.component_sizes, key=lambda key: self.component_sizes[key])

    @property
    def largest_component_size(self) -> int:
        largest = self.largest_component_id
        return 0 if largest is None else self.component_sizes[largest]


def generate_open_mask(config: MapGenerationConfig | TopologyConfig | dict[str, Any], seed: int) -> np.ndarray:
    topology, shape = _topology_and_shape(config)
    rng = np.random.default_rng(int(seed))
    if topology.model == "iid_site":
        return rng.random(shape) < topology.p_open
    if topology.model == "correlated_site":
        field = generate_rank_normalized_field(shape, topology.hurst or 0.5, seed)
        return _top_rank_mask(field, topology.p_open)
    raise ValueError(f"topology.model: unsupported value {topology.model!r}")


def label_components(open_mask: np.ndarray) -> ComponentIndex:
    mask = np.asarray(open_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("open_mask: expected a 2D array")
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    component_sizes: dict[int, int] = {}
    spans_horizontal: set[int] = set()
    spans_vertical: set[int] = set()
    next_id = 1

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or labels[row, col] != 0:
                continue
            size, touches = _flood_fill(mask, labels, (row, col), next_id)
            component_sizes[next_id] = size
            if touches["left"] and touches["right"]:
                spans_horizontal.add(next_id)
            if touches["top"] and touches["bottom"]:
                spans_vertical.add(next_id)
            next_id += 1

    return ComponentIndex(
        labels=labels,
        component_sizes=component_sizes,
        spans_horizontal=spans_horizontal,
        spans_vertical=spans_vertical,
    )


def _top_rank_mask(field: np.ndarray, p_open: float) -> np.ndarray:
    flat = np.asarray(field).reshape(-1)
    target_count = int(round(float(p_open) * flat.size))
    if target_count <= 0:
        return np.zeros(field.shape, dtype=bool)
    if target_count >= flat.size:
        return np.ones(field.shape, dtype=bool)
    order = np.lexsort((np.arange(flat.size), -flat))
    mask = np.zeros(flat.size, dtype=bool)
    mask[order[:target_count]] = True
    return mask.reshape(field.shape)


def _flood_fill(
    mask: np.ndarray,
    labels: np.ndarray,
    start: tuple[int, int],
    component_id: int,
) -> tuple[int, dict[str, bool]]:
    height, width = mask.shape
    queue = deque([start])
    labels[start] = component_id
    size = 0
    touches = {"top": False, "bottom": False, "left": False, "right": False}
    while queue:
        row, col = queue.popleft()
        size += 1
        touches["top"] = touches["top"] or row == 0
        touches["bottom"] = touches["bottom"] or row == height - 1
        touches["left"] = touches["left"] or col == 0
        touches["right"] = touches["right"] or col == width - 1
        for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            if not mask[next_row, next_col] or labels[next_row, next_col] != 0:
                continue
            labels[next_row, next_col] = component_id
            queue.append((next_row, next_col))
    return size, touches


def _topology_and_shape(config: MapGenerationConfig | TopologyConfig | dict[str, Any]) -> tuple[TopologyConfig, tuple[int, int]]:
    if isinstance(config, MapGenerationConfig):
        return config.world.topology, config.grid_size
    if isinstance(config, TopologyConfig):
        return config, (32, 32)
    parsed = MapGenerationConfig.from_config({"env": config} if "env" not in config else config)
    return parsed.world.topology, parsed.grid_size
