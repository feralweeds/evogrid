"""Static diagnostics for generated maps."""

from __future__ import annotations

import heapq
import math
from typing import Any

import numpy as np

from evogrid.envs.map_generation.connectivity import ComponentIndex, label_components
from evogrid.envs.map_generation.schemas import MapGenerationConfig

Position = tuple[int, int]


def compute_map_diagnostics(
    open_mask: np.ndarray,
    roughness: np.ndarray,
    base_pos: Position | None = None,
    ore_positions: set[Position] | None = None,
    config: MapGenerationConfig | None = None,
    map_id: str = "",
    placement_status: str = "ok",
) -> dict[str, Any]:
    """Compute reset-time/evaluator-only map diagnostics without mutating inputs."""

    mask = np.asarray(open_mask, dtype=bool)
    rough = np.asarray(roughness, dtype=np.float64)
    if mask.ndim != 2:
        raise ValueError("open_mask: expected a 2D array")
    if rough.shape != mask.shape:
        raise ValueError("roughness: expected same shape as open_mask")

    index = label_components(mask)
    total_cells = int(mask.size)
    open_count = int(mask.sum())
    ore_positions = set(ore_positions or set())
    base_ore_fraction = _base_ore_reachable_fraction(index, base_pos, ore_positions)
    shortest = _shortest_path(mask, base_pos, ore_positions)
    min_cost = _minimum_cost_path(mask, rough, base_pos, ore_positions, config)
    rough_patch_index = label_components(_rough_patch_mask(rough, config))
    terrain_axis = _axis_neighbor_statistics(rough)
    topology_axis = _axis_neighbor_statistics(mask.astype(np.float64))

    return {
        "schema_version": 1,
        "map_id": map_id,
        "target_p_open": None if config is None else config.world.topology.p_open,
        "realized_p_open": open_count / total_cells if total_cells else 0.0,
        "component_count": index.component_count,
        "largest_component_size": index.largest_component_size,
        "largest_component_fraction": index.largest_component_size / total_cells if total_cells else 0.0,
        "spans_horizontal": bool(index.spans_horizontal),
        "spans_vertical": bool(index.spans_vertical),
        "base_ore_reachable_fraction": base_ore_fraction,
        "shortest_path_length": None if shortest is None else len(shortest) - 1,
        "minimum_cost_path_cost": min_cost,
        "path_tortuosity": _path_tortuosity(shortest, base_pos),
        "articulation_point_count": _articulation_point_count(mask, index, base_pos),
        "roughness_mean": float(rough.mean()) if rough.size else 0.0,
        "roughness_std": float(rough.std()) if rough.size else 0.0,
        "rough_patch_count": rough_patch_index.component_count,
        "largest_rough_patch_fraction": (
            rough_patch_index.largest_component_size / total_cells if total_cells else 0.0
        ),
        "estimated_terrain_hurst": _estimate_hurst_like(rough),
        "estimated_topology_hurst": _estimate_hurst_like(mask.astype(np.float64)),
        "terrain_neighbor_correlation": terrain_axis["mean_correlation"],
        "terrain_axis_corr_horizontal": terrain_axis["horizontal_correlation"],
        "terrain_axis_corr_vertical": terrain_axis["vertical_correlation"],
        "terrain_axis_corr_abs_diff": terrain_axis["correlation_abs_diff"],
        "terrain_lag1_semivariance_horizontal": terrain_axis["horizontal_semivariance"],
        "terrain_lag1_semivariance_vertical": terrain_axis["vertical_semivariance"],
        "terrain_lag1_semivariance_abs_diff": terrain_axis["semivariance_abs_diff"],
        "topology_neighbor_correlation": topology_axis["mean_correlation"],
        "topology_axis_corr_horizontal": topology_axis["horizontal_correlation"],
        "topology_axis_corr_vertical": topology_axis["vertical_correlation"],
        "topology_axis_corr_abs_diff": topology_axis["correlation_abs_diff"],
        "topology_lag1_semivariance_horizontal": topology_axis["horizontal_semivariance"],
        "topology_lag1_semivariance_vertical": topology_axis["vertical_semivariance"],
        "topology_lag1_semivariance_abs_diff": topology_axis["semivariance_abs_diff"],
        "valid_for_percolation_analysis": _valid_for_percolation_analysis(config),
        "placement_status": placement_status,
    }


def _base_ore_reachable_fraction(
    index: ComponentIndex,
    base_pos: Position | None,
    ore_positions: set[Position],
) -> float:
    if base_pos is None or not ore_positions:
        return 0.0
    base_label = int(index.labels[base_pos])
    if base_label == 0:
        return 0.0
    reachable = sum(1 for ore_pos in ore_positions if int(index.labels[ore_pos]) == base_label)
    return reachable / len(ore_positions)


def _shortest_path(
    mask: np.ndarray,
    base_pos: Position | None,
    ore_positions: set[Position],
) -> list[Position] | None:
    if base_pos is None or not ore_positions or not mask[base_pos]:
        return None
    if base_pos in ore_positions:
        return [base_pos]
    queue: list[Position] = [base_pos]
    parents: dict[Position, Position | None] = {base_pos: None}
    cursor = 0
    while cursor < len(queue):
        current = queue[cursor]
        cursor += 1
        for nxt in _neighbors(current, mask.shape):
            if nxt in parents or not mask[nxt]:
                continue
            parents[nxt] = current
            if nxt in ore_positions:
                return _reconstruct_path(parents, nxt)
            queue.append(nxt)
    return None


def _minimum_cost_path(
    mask: np.ndarray,
    roughness: np.ndarray,
    base_pos: Position | None,
    ore_positions: set[Position],
    config: MapGenerationConfig | None,
) -> float | None:
    if base_pos is None or not ore_positions or not mask[base_pos]:
        return None
    if base_pos in ore_positions:
        return 0.0
    terrain = None if config is None else config.world.terrain
    base_cost = 0.01 if terrain is None else terrain.base_move_cost
    strength = 0.04 if terrain is None else terrain.roughness_strength
    exponent = 1.0 if terrain is None else terrain.cost_exponent
    distances: dict[Position, float] = {base_pos: 0.0}
    heap: list[tuple[float, Position]] = [(0.0, base_pos)]
    while heap:
        cost, current = heapq.heappop(heap)
        if cost != distances[current]:
            continue
        if current in ore_positions:
            return cost
        for nxt in _neighbors(current, mask.shape):
            if not mask[nxt]:
                continue
            step_cost = base_cost + strength * float(roughness[nxt]) ** exponent
            next_cost = cost + step_cost
            if next_cost < distances.get(nxt, math.inf):
                distances[nxt] = next_cost
                heapq.heappush(heap, (next_cost, nxt))
    return None


def _path_tortuosity(path: list[Position] | None, base_pos: Position | None) -> float | None:
    if path is None or base_pos is None:
        return None
    path_length = len(path) - 1
    manhattan = abs(path[-1][0] - base_pos[0]) + abs(path[-1][1] - base_pos[1])
    if manhattan == 0:
        return 0.0
    return path_length / manhattan


def _rough_patch_mask(roughness: np.ndarray, config: MapGenerationConfig | None) -> np.ndarray:
    threshold = 0.75
    if config is not None and config.world.terrain.observation_bins:
        threshold = config.world.terrain.observation_bins[-1]
    return roughness >= threshold


def _articulation_point_count(mask: np.ndarray, index: ComponentIndex, base_pos: Position | None) -> int:
    if base_pos is None or not mask[base_pos]:
        component_id = index.largest_component_id
    else:
        component_id = int(index.labels[base_pos])
    if component_id is None or component_id == 0:
        return 0
    nodes = [tuple(map(int, pos)) for pos in np.argwhere(index.labels == component_id)]
    node_set = set(nodes)
    if len(nodes) < 3:
        return 0

    discovery: dict[Position, int] = {}
    low: dict[Position, int] = {}
    parent: dict[Position, Position | None] = {}
    child_count: dict[Position, int] = {}
    articulation: set[Position] = set()
    time = 0

    root = nodes[0]
    parent[root] = None
    discovery[root] = time
    low[root] = time
    child_count[root] = 0
    time += 1
    stack: list[tuple[Position, Any]] = [(root, iter(_neighbors(root, mask.shape)))]

    while stack:
        node, iterator = stack[-1]
        try:
            nxt = next(iterator)
        except StopIteration:
            stack.pop()
            parent_node = parent.get(node)
            if parent_node is None:
                if child_count.get(node, 0) > 1:
                    articulation.add(node)
            else:
                low[parent_node] = min(low[parent_node], low[node])
                if parent.get(parent_node) is not None and low[node] >= discovery[parent_node]:
                    articulation.add(parent_node)
            continue

        if nxt not in node_set:
            continue
        if nxt not in discovery:
            parent[nxt] = node
            child_count[node] = child_count.get(node, 0) + 1
            child_count[nxt] = 0
            discovery[nxt] = time
            low[nxt] = time
            time += 1
            stack.append((nxt, iter(_neighbors(nxt, mask.shape))))
        elif nxt != parent.get(node):
            low[node] = min(low[node], discovery[nxt])
    return len(articulation)


def _estimate_hurst_like(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 4:
        return None
    variances: list[tuple[float, float]] = []
    for lag in (1, 2, 4, 8):
        if lag >= array.shape[0] or lag >= array.shape[1]:
            continue
        diffs = np.concatenate(
            [
                (array[:, lag:] - array[:, :-lag]).reshape(-1),
                (array[lag:, :] - array[:-lag, :]).reshape(-1),
            ]
        )
        semivariance = float(np.mean(diffs * diffs))
        if semivariance > 0.0:
            variances.append((math.log(float(lag)), math.log(semivariance)))
    if len(variances) < 2:
        return None
    xs = np.array([item[0] for item in variances])
    ys = np.array([item[1] for item in variances])
    slope = float(np.polyfit(xs, ys, 1)[0])
    return max(0.0, min(1.0, slope / 2.0))


def _axis_neighbor_statistics(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 2:
        return {
            "horizontal_correlation": None,
            "vertical_correlation": None,
            "mean_correlation": None,
            "correlation_abs_diff": None,
            "horizontal_semivariance": None,
            "vertical_semivariance": None,
            "semivariance_abs_diff": None,
        }
    horizontal_left = array[:, :-1].reshape(-1)
    horizontal_right = array[:, 1:].reshape(-1)
    vertical_top = array[:-1, :].reshape(-1)
    vertical_bottom = array[1:, :].reshape(-1)
    horizontal_corr = _pearson(horizontal_left, horizontal_right)
    vertical_corr = _pearson(vertical_top, vertical_bottom)
    horizontal_semivar = float(np.mean((horizontal_right - horizontal_left) ** 2) / 2.0)
    vertical_semivar = float(np.mean((vertical_bottom - vertical_top) ** 2) / 2.0)
    correlations = [value for value in (horizontal_corr, vertical_corr) if value is not None]
    return {
        "horizontal_correlation": horizontal_corr,
        "vertical_correlation": vertical_corr,
        "mean_correlation": float(np.mean(correlations)) if correlations else None,
        "correlation_abs_diff": (
            None if horizontal_corr is None or vertical_corr is None else abs(horizontal_corr - vertical_corr)
        ),
        "horizontal_semivariance": horizontal_semivar,
        "vertical_semivariance": vertical_semivar,
        "semivariance_abs_diff": abs(horizontal_semivar - vertical_semivar),
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0 or right.size == 0:
        return None
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std == 0.0 or right_std == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _valid_for_percolation_analysis(config: MapGenerationConfig | None) -> bool:
    if config is None:
        return True
    return config.world.topology.solvability_mode != "legacy_carved"


def _neighbors(pos: Position, shape: tuple[int, int]) -> tuple[Position, ...]:
    row, col = pos
    height, width = shape
    neighbors = []
    for nxt in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
        next_row, next_col = nxt
        if 0 <= next_row < height and 0 <= next_col < width:
            neighbors.append(nxt)
    return tuple(neighbors)


def _reconstruct_path(parents: dict[Position, Position | None], goal: Position) -> list[Position]:
    path: list[Position] = []
    current: Position | None = goal
    while current is not None:
        path.append(current)
        current = parents[current]
    path.reverse()
    return path
