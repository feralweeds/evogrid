"""Map construction utilities."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict
from typing import Iterable, List, Sequence, Tuple

from evogrid.constants import Tile
from evogrid.envs.map_generation.fractal_percolation import FractalPercolationMapGenerator
from evogrid.envs.map_generation.schemas import (
    MapBuildResult,
    MapGenerationConfig,
    stable_map_id,
)

Position = Tuple[int, int]


def _as_pos(value: Sequence[int]) -> Position:
    return int(value[0]), int(value[1])


def _set_many(grid: List[List[int]], positions: Iterable[Position], tile: Tile) -> None:
    height = len(grid)
    width = len(grid[0])
    for row, col in positions:
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = int(tile)


def build_map(config: dict | None = None, seed: int | None = None) -> MapBuildResult:
    generation_config = MapGenerationConfig.from_config(config, seed=seed)
    if generation_config.map_mode == "fractal_percolation":
        return FractalPercolationMapGenerator().build(
            generation_config,
            generation_config.world.world_seed,
        )

    grid, base_pos, ore_positions = _build_legacy_map(config, seed=seed)
    payload = {
        "grid": grid,
        "base_pos": base_pos,
        "ore_positions": ore_positions,
        "map_mode": generation_config.map_mode,
        "world_seed": generation_config.world.world_seed,
    }
    map_id = stable_map_id(payload)
    diagnostics = _legacy_diagnostics(grid, map_id, generation_config.map_mode)
    provenance = {
        "schema_version": 1,
        "map_mode": generation_config.map_mode,
        "world_seed": generation_config.world.world_seed,
        "generator_version": generation_config.world.generator_version,
        "config": asdict(generation_config),
    }
    return MapBuildResult(
        schema_version=1,
        map_id=map_id,
        grid=grid,
        roughness=None,
        base_pos=base_pos,
        ore_positions=ore_positions,
        diagnostics=diagnostics,
        provenance=provenance,
    )


def build_fixed_map(config: dict | None = None, seed: int | None = None) -> tuple[List[List[int]], Position, set[Position]]:
    result = build_map(config, seed=seed)
    return result.grid, result.base_pos, result.ore_positions


def _build_legacy_map(config: dict | None = None, seed: int | None = None) -> tuple[List[List[int]], Position, set[Position]]:
    config = config or {}
    env_config = config.get("env", config)
    if str(env_config.get("map_mode", "fixed")) == "random_curriculum":
        return build_random_curriculum_map(env_config, seed=seed)
    if str(env_config.get("map_mode", "fixed")) in {
        "controlled_corridor_curriculum",
        "controlled_random_curriculum",
    }:
        return build_controlled_corridor_curriculum_map(env_config, seed=seed)

    height, width = env_config.get("grid_size", [32, 32])
    base_pos = _as_pos(env_config.get("base_pos", [2, 2]))
    ore_positions = {_as_pos(pos) for pos in env_config.get("ore_positions", [[26, 26]])}

    grid = [[int(Tile.GROUND) for _ in range(width)] for _ in range(height)]

    explicit_obstacles = env_config.get("obstacles")
    explicit_rough = env_config.get("rough_terrain")

    if explicit_rough is not None:
        _set_many(grid, (_as_pos(pos) for pos in explicit_rough), Tile.ROUGH)
    elif env_config.get("enable_rough_terrain", True):
        rough_cells = []
        for row in range(8, min(25, height)):
            for col in range(8, min(25, width)):
                if (row + col) % 3 != 0:
                    rough_cells.append((row, col))
        _set_many(grid, rough_cells, Tile.ROUGH)

    if explicit_obstacles is not None:
        _set_many(grid, (_as_pos(pos) for pos in explicit_obstacles), Tile.OBSTACLE)
    elif env_config.get("enable_obstacles", True):
        wall_col = min(14, width - 2)
        gap_row = min(28, height - 2)
        wall_cells = [(row, wall_col) for row in range(0, min(28, height)) if row != gap_row]
        _set_many(grid, wall_cells, Tile.OBSTACLE)

    row, col = base_pos
    grid[row][col] = int(Tile.BASE)
    for ore_pos in ore_positions:
        row, col = ore_pos
        grid[row][col] = int(Tile.ORE)

    return grid, base_pos, ore_positions


def _legacy_diagnostics(grid: list[list[int]], map_id: str, map_mode: str) -> dict:
    tile_counts: dict[str, int] = {}
    for row in grid:
        for value in row:
            name = Tile(value).name
            tile_counts[name] = tile_counts.get(name, 0) + 1
    total_cells = len(grid) * len(grid[0]) if grid else 0
    open_count = total_cells - tile_counts.get("OBSTACLE", 0)
    return {
        "schema_version": 1,
        "map_id": map_id,
        "map_mode": map_mode,
        "tile_counts": tile_counts,
        "target_p_open": None,
        "realized_p_open": (open_count / total_cells) if total_cells else 0.0,
        "valid_for_percolation_analysis": False,
        "placement_status": "legacy_carved",
    }


def build_random_curriculum_map(
    env_config: dict,
    seed: int | None = None,
) -> tuple[List[List[int]], Position, set[Position]]:
    """Build a seeded random map while preserving at least one base-to-ore path."""

    height, width = env_config.get("grid_size", [16, 16])
    base_pos = _as_pos(env_config.get("base_pos", [2, 2]))
    random_config = env_config.get("random_map", {})
    rng = random.Random(seed if seed is not None else int(random_config.get("seed", 0)))

    ore_count = int(random_config.get("ore_count", 1))
    min_distance = int(random_config.get("min_base_ore_distance", max(2, min(height, width) // 2)))
    obstacle_density = float(random_config.get("obstacle_density", 0.12))
    rough_density = float(random_config.get("rough_density", 0.25))
    rough_corridor_bias = float(random_config.get("rough_corridor_bias", 0.4))
    ensure_reachable = bool(random_config.get("ensure_reachable", True))
    max_attempts = int(random_config.get("max_generation_attempts", 100))

    if not _in_bounds(base_pos, height, width):
        raise ValueError(f"base_pos {base_pos} is outside grid_size {(height, width)}")
    if ore_count < 1:
        raise ValueError("random_map.ore_count must be at least 1")

    for _ in range(max_attempts):
        ore_positions = _sample_ore_positions(
            rng=rng,
            height=height,
            width=width,
            base_pos=base_pos,
            ore_count=ore_count,
            min_distance=min_distance,
        )
        protected_paths = set()
        for ore_pos in ore_positions:
            protected_paths.update(_random_manhattan_path(rng, base_pos, ore_pos))

        grid = [[int(Tile.GROUND) for _ in range(width)] for _ in range(height)]
        special_cells = {base_pos, *ore_positions}

        for row in range(height):
            for col in range(width):
                pos = (row, col)
                if pos in special_cells or pos in protected_paths:
                    continue
                if rng.random() < obstacle_density:
                    grid[row][col] = int(Tile.OBSTACLE)

        corridor_cells = _corridor_neighborhood(protected_paths, height, width)
        for row in range(height):
            for col in range(width):
                pos = (row, col)
                if pos in special_cells or grid[row][col] == int(Tile.OBSTACLE):
                    continue
                probability = rough_density
                if pos in corridor_cells:
                    probability = max(probability, rough_corridor_bias)
                if rng.random() < probability:
                    grid[row][col] = int(Tile.ROUGH)

        base_row, base_col = base_pos
        grid[base_row][base_col] = int(Tile.BASE)
        for ore_row, ore_col in ore_positions:
            grid[ore_row][ore_col] = int(Tile.ORE)

        if not ensure_reachable or all(_is_reachable(grid, base_pos, ore_pos) for ore_pos in ore_positions):
            return grid, base_pos, ore_positions

    raise RuntimeError(f"Failed to generate a reachable random map after {max_attempts} attempts.")


def build_controlled_corridor_curriculum_map(
    env_config: dict,
    seed: int | None = None,
) -> tuple[List[List[int]], Position, set[Position]]:
    """Build randomized road-learning maps with controlled corridor pressure.

    The generator creates three kinds of maps without exposing the type to the
    agent: positive maps with rough transport corridors, negative maps with
    mostly off-route rough distractors, and mixed maps containing both.
    """

    height, width = env_config.get("grid_size", [16, 16])
    random_config = env_config.get("random_map", {})
    rng = random.Random(seed if seed is not None else int(random_config.get("seed", 0)))

    max_attempts = int(random_config.get("max_generation_attempts", 100))
    min_distance = int(random_config.get("min_base_ore_distance", max(4, min(height, width) // 2)))
    ore_count = int(random_config.get("ore_count", 1))
    obstacle_density = float(random_config.get("obstacle_density", 0.08))
    extra_rough_density = float(random_config.get("extra_rough_density", 0.08))
    corridor = random_config.get("controlled_corridor", {})
    base_margin = int(corridor.get("base_margin", 2))
    route_rough_probability = {
        "positive": float(corridor.get("positive_route_rough_probability", 0.70)),
        "mixed": float(corridor.get("mixed_route_rough_probability", 0.38)),
        "negative": float(corridor.get("negative_route_rough_probability", 0.06)),
    }
    off_route_rough_probability = {
        "positive": float(corridor.get("positive_off_route_rough_probability", 0.08)),
        "mixed": float(corridor.get("mixed_off_route_rough_probability", 0.24)),
        "negative": float(corridor.get("negative_off_route_rough_probability", 0.34)),
    }
    min_route_rough = {
        "positive": int(corridor.get("positive_min_route_rough", 4)),
        "mixed": int(corridor.get("mixed_min_route_rough", 2)),
        "negative": int(corridor.get("negative_min_route_rough", 0)),
    }
    rough_band_probability = {
        "positive": float(corridor.get("positive_transport_band_probability", 0.70)),
        "mixed": float(corridor.get("mixed_transport_band_probability", 0.35)),
        "negative": float(corridor.get("negative_transport_band_probability", 0.0)),
    }
    scenario_weights = {
        "positive": float(corridor.get("positive_weight", 0.60)),
        "mixed": float(corridor.get("mixed_weight", 0.20)),
        "negative": float(corridor.get("negative_weight", 0.20)),
    }

    if ore_count < 1:
        raise ValueError("random_map.ore_count must be at least 1")

    for _ in range(max_attempts):
        scenario = _weighted_choice(rng, scenario_weights)
        base_pos = _sample_base_position(
            rng=rng,
            height=height,
            width=width,
            margin=base_margin,
            fallback=_as_pos(env_config.get("base_pos", [2, 2])),
        )
        ore_positions = _sample_ore_positions(
            rng=rng,
            height=height,
            width=width,
            base_pos=base_pos,
            ore_count=ore_count,
            min_distance=min_distance,
        )
        primary_ore = max(ore_positions, key=lambda pos: _manhattan(base_pos, pos))
        route = _random_manhattan_path(rng, base_pos, primary_ore)
        route_cells = set(route)
        protected_cells = set(route_cells) | {base_pos, *ore_positions}
        buildable_route_cells = [pos for pos in route[1:-1] if pos not in ore_positions]

        grid = [[int(Tile.GROUND) for _ in range(width)] for _ in range(height)]
        _apply_obstacles(
            rng=rng,
            grid=grid,
            protected_cells=protected_cells,
            obstacle_density=obstacle_density,
        )
        _apply_route_rough(
            rng=rng,
            grid=grid,
            route_cells=buildable_route_cells,
            probability=route_rough_probability[scenario],
            minimum=min_route_rough[scenario],
        )
        _apply_transport_rough_band(
            rng=rng,
            grid=grid,
            base_pos=base_pos,
            ore_pos=primary_ore,
            protected_cells={base_pos, *ore_positions},
            probability=rough_band_probability[scenario],
        )
        _apply_off_route_rough(
            rng=rng,
            grid=grid,
            protected_cells={base_pos, *ore_positions},
            route_cells=route_cells,
            probability=off_route_rough_probability[scenario],
            extra_probability=extra_rough_density,
        )

        base_row, base_col = base_pos
        grid[base_row][base_col] = int(Tile.BASE)
        for ore_row, ore_col in ore_positions:
            grid[ore_row][ore_col] = int(Tile.ORE)

        if all(_is_reachable(grid, base_pos, ore_pos) for ore_pos in ore_positions):
            return grid, base_pos, ore_positions

    raise RuntimeError(f"Failed to generate a controlled corridor map after {max_attempts} attempts.")


def _sample_base_position(
    rng: random.Random,
    height: int,
    width: int,
    margin: int,
    fallback: Position,
) -> Position:
    candidates = [
        (row, col)
        for row in range(max(0, margin), max(0, height - margin))
        for col in range(max(0, margin), max(0, width - margin))
    ]
    if not candidates:
        return fallback if _in_bounds(fallback, height, width) else (0, 0)
    return rng.choice(candidates)


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 0.0:
        return next(iter(weights))
    roll = rng.random() * total
    cumulative = 0.0
    for key, value in weights.items():
        cumulative += max(0.0, float(value))
        if roll <= cumulative:
            return key
    return next(reversed(weights))


def _apply_obstacles(
    rng: random.Random,
    grid: list[list[int]],
    protected_cells: set[Position],
    obstacle_density: float,
) -> None:
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            pos = (row, col)
            if pos in protected_cells:
                continue
            if rng.random() < obstacle_density:
                grid[row][col] = int(Tile.OBSTACLE)


def _apply_route_rough(
    rng: random.Random,
    grid: list[list[int]],
    route_cells: list[Position],
    probability: float,
    minimum: int,
) -> None:
    selected = [pos for pos in route_cells if rng.random() < probability]
    remaining = [pos for pos in route_cells if pos not in selected]
    while len(selected) < minimum and remaining:
        selected.append(remaining.pop(rng.randrange(len(remaining))))
    _set_many(grid, selected, Tile.ROUGH)


def _apply_off_route_rough(
    rng: random.Random,
    grid: list[list[int]],
    protected_cells: set[Position],
    route_cells: set[Position],
    probability: float,
    extra_probability: float,
) -> None:
    corridor_neighbors = _corridor_neighborhood(route_cells, len(grid), len(grid[0]))
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            pos = (row, col)
            if pos in protected_cells or pos in route_cells or grid[row][col] == int(Tile.OBSTACLE):
                continue
            rough_probability = probability if pos in corridor_neighbors else extra_probability
            if rng.random() < rough_probability:
                grid[row][col] = int(Tile.ROUGH)


def _apply_transport_rough_band(
    rng: random.Random,
    grid: list[list[int]],
    base_pos: Position,
    ore_pos: Position,
    protected_cells: set[Position],
    probability: float,
) -> None:
    if probability <= 0.0:
        return
    height = len(grid)
    width = len(grid[0])
    row_delta = abs(base_pos[0] - ore_pos[0])
    col_delta = abs(base_pos[1] - ore_pos[1])
    thickness = 1 + int(rng.random() < 0.35)
    if col_delta >= row_delta:
        center_col = (base_pos[1] + ore_pos[1]) // 2
        cols = range(max(0, center_col - thickness + 1), min(width, center_col + thickness + 1))
        cells = [(row, col) for col in cols for row in range(height)]
    else:
        center_row = (base_pos[0] + ore_pos[0]) // 2
        rows = range(max(0, center_row - thickness + 1), min(height, center_row + thickness + 1))
        cells = [(row, col) for row in rows for col in range(width)]

    for row, col in cells:
        pos = (row, col)
        if pos in protected_cells or grid[row][col] == int(Tile.OBSTACLE):
            continue
        if rng.random() < probability:
            grid[row][col] = int(Tile.ROUGH)


def _sample_ore_positions(
    rng: random.Random,
    height: int,
    width: int,
    base_pos: Position,
    ore_count: int,
    min_distance: int,
) -> set[Position]:
    candidates = [
        (row, col)
        for row in range(height)
        for col in range(width)
        if (row, col) != base_pos and _manhattan((row, col), base_pos) >= min_distance
    ]
    if len(candidates) < ore_count:
        candidates = [
            (row, col)
            for row in range(height)
            for col in range(width)
            if (row, col) != base_pos
        ]
    if len(candidates) < ore_count:
        raise ValueError("Grid is too small for requested random_map.ore_count")
    return set(rng.sample(candidates, ore_count))


def _random_manhattan_path(rng: random.Random, start: Position, goal: Position) -> list[Position]:
    current = start
    path = [current]
    while current != goal:
        row, col = current
        goal_row, goal_col = goal
        choices = []
        if row < goal_row:
            choices.append((row + 1, col))
        elif row > goal_row:
            choices.append((row - 1, col))
        if col < goal_col:
            choices.append((row, col + 1))
        elif col > goal_col:
            choices.append((row, col - 1))
        current = rng.choice(choices)
        path.append(current)
    return path


def _corridor_neighborhood(path_cells: set[Position], height: int, width: int) -> set[Position]:
    cells = set(path_cells)
    for row, col in list(path_cells):
        for pos in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if _in_bounds(pos, height, width):
                cells.add(pos)
    return cells


def _is_reachable(grid: list[list[int]], start: Position, goal: Position) -> bool:
    queue = deque([start])
    seen = {start}
    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        row, col = current
        for nxt in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if nxt in seen or not _in_bounds(nxt, len(grid), len(grid[0])):
                continue
            next_row, next_col = nxt
            if Tile(grid[next_row][next_col]) == Tile.OBSTACLE:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return False


def _in_bounds(pos: Position, height: int, width: int) -> bool:
    row, col = pos
    return 0 <= row < height and 0 <= col < width


def _manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])
