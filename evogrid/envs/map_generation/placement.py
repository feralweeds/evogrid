"""Base and resource placement for generated maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evogrid.envs.map_generation.connectivity import ComponentIndex, label_components
from evogrid.envs.map_generation.schemas import MapGenerationConfig
from evogrid.envs.map_generation.seeding import derive_seed
from evogrid.envs.map_generation.spectral_field import generate_rank_normalized_field

Position = tuple[int, int]


@dataclass(frozen=True)
class PlacementResult:
    base_pos: Position | None
    ore_positions: set[Position]
    placement_status: str
    diagnostics: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.placement_status == "ok"


def place_base_and_resources(
    open_mask: np.ndarray,
    config: MapGenerationConfig,
    seed: int,
    resource_seed: int | None = None,
) -> PlacementResult:
    """Place base and resources without modifying topology."""

    mask = np.asarray(open_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("open_mask: expected a 2D array")
    topology = config.world.topology
    index = label_components(mask)
    rng = np.random.default_rng(int(seed))

    if topology.solvability_mode == "raw":
        allowed_label = None
        allowed_positions = _positions(mask)
    elif topology.solvability_mode == "conditioned_same_component":
        allowed_label = _choose_component(index, config, rng)
        if allowed_label is None:
            return _failed("task_placement_failed", index, "no_component_meets_threshold")
        allowed_positions = _positions(index.labels == allowed_label)
    elif topology.solvability_mode == "legacy_carved":
        allowed_label = None
        allowed_positions = _positions(mask)
    else:
        raise ValueError(f"env.world.topology.solvability_mode: unsupported value {topology.solvability_mode!r}")

    if not allowed_positions:
        return _failed("task_placement_failed", index, "no_open_candidate")

    base_candidates = _apply_margin(allowed_positions, mask.shape, config.world.placement.base_margin)
    if not base_candidates:
        return _failed("task_placement_failed", index, "no_base_candidate")

    for _ in range(config.world.placement.max_attempts):
        base_pos = base_candidates[int(rng.integers(0, len(base_candidates)))]
        ore_positions = _select_ores(
            allowed_positions=allowed_positions,
            base_pos=base_pos,
            shape=mask.shape,
            config=config,
            seed=derive_seed(seed, "resources") if resource_seed is None else resource_seed,
        )
        if len(ore_positions) == config.world.resources.count:
            reachable_fraction = _reachable_ore_fraction(index, base_pos, ore_positions)
            return PlacementResult(
                base_pos=base_pos,
                ore_positions=ore_positions,
                placement_status="ok",
                diagnostics={
                    "component_id": allowed_label,
                    "component_count": index.component_count,
                    "largest_component_size": index.largest_component_size,
                    "base_ore_reachable_fraction": reachable_fraction,
                    "placement_failure_reason": None,
                },
            )

    return _failed("task_placement_failed", index, "resource_constraints_unsatisfied")


def _select_ores(
    allowed_positions: list[Position],
    base_pos: Position,
    shape: tuple[int, int],
    config: MapGenerationConfig,
    seed: int,
) -> set[Position]:
    resources = config.world.resources
    candidates = [
        pos
        for pos in allowed_positions
        if pos != base_pos and _manhattan(pos, base_pos) >= resources.min_base_distance
    ]
    if not candidates:
        return set()
    if resources.distribution == "clustered":
        field = generate_rank_normalized_field(shape, resources.hurst or 0.5, seed)
        candidates = sorted(candidates, key=lambda pos: (-float(field[pos]), pos[0], pos[1]))
    elif resources.distribution == "uniform":
        rng = np.random.default_rng(int(seed))
        candidates = [candidates[index] for index in rng.permutation(len(candidates))]
    else:
        raise ValueError(f"env.world.resources.distribution: unsupported value {resources.distribution!r}")

    selected: list[Position] = []
    for pos in candidates:
        if all(_manhattan(pos, chosen) >= resources.min_pair_distance for chosen in selected):
            selected.append(pos)
            if len(selected) == resources.count:
                break
    return set(selected)


def _choose_component(
    index: ComponentIndex,
    config: MapGenerationConfig,
    rng: np.random.Generator,
) -> int | None:
    if not index.component_sizes:
        return None
    min_size = int(np.ceil(config.world.topology.min_task_component_fraction * np.prod(config.grid_size)))
    candidates = [component_id for component_id, size in index.component_sizes.items() if size >= min_size]
    if not candidates:
        return None
    if config.world.topology.task_component == "largest":
        return max(candidates, key=lambda component_id: index.component_sizes[component_id])
    if config.world.topology.task_component == "sampled":
        return candidates[int(rng.integers(0, len(candidates)))]
    raise ValueError("env.world.topology.task_component: expected 'largest' or 'sampled'")


def _failed(status: str, index: ComponentIndex, reason: str) -> PlacementResult:
    return PlacementResult(
        base_pos=None,
        ore_positions=set(),
        placement_status=status,
        diagnostics={
            "component_id": None,
            "component_count": index.component_count,
            "largest_component_size": index.largest_component_size,
            "base_ore_reachable_fraction": 0.0,
            "placement_failure_reason": reason,
        },
    )


def _positions(mask: np.ndarray) -> list[Position]:
    rows, cols = np.nonzero(mask)
    return [(int(row), int(col)) for row, col in zip(rows, cols)]


def _apply_margin(positions: list[Position], shape: tuple[int, int], margin: int) -> list[Position]:
    height, width = shape
    return [
        (row, col)
        for row, col in positions
        if margin <= row < height - margin and margin <= col < width - margin
    ]


def _reachable_ore_fraction(index: ComponentIndex, base_pos: Position, ore_positions: set[Position]) -> float:
    if not ore_positions:
        return 0.0
    base_label = int(index.labels[base_pos])
    if base_label == 0:
        return 0.0
    reachable = sum(1 for pos in ore_positions if int(index.labels[pos]) == base_label)
    return reachable / len(ore_positions)


def _manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])
