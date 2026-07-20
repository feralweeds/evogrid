"""Schemas for parameterized map generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any


ALLOWED_MAP_MODES = {
    "fixed",
    "random_curriculum",
    "controlled_corridor_curriculum",
    "controlled_random_curriculum",
    "fractal_percolation",
}
ALLOWED_TOPOLOGY_MODELS = {"iid_site", "correlated_site"}
ALLOWED_SOLVABILITY_MODES = {"raw", "conditioned_same_component", "legacy_carved"}
ALLOWED_RESOURCE_DISTRIBUTIONS = {"uniform", "clustered"}


@dataclass(frozen=True)
class TopologyConfig:
    model: str = "correlated_site"
    p_open: float = 0.65
    hurst: float | None = 0.5
    solvability_mode: str = "conditioned_same_component"
    task_component: str = "largest"
    min_task_component_fraction: float = 0.2
    boundary_mode: str = "finite_4_neighbor"


@dataclass(frozen=True)
class TerrainConfig:
    model: str = "fractional_gaussian_surface"
    hurst: float = 0.7
    base_move_cost: float = 0.01
    roughness_strength: float = 0.04
    cost_exponent: float = 1.0
    road_move_cost: float = 0.0
    observation_bins: tuple[float, ...] = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class ResourcesConfig:
    distribution: str = "clustered"
    count: int = 1
    hurst: float | None = 0.7
    renewable: bool = True
    min_base_distance: int = 0
    min_pair_distance: int = 0


@dataclass(frozen=True)
class PlacementConfig:
    base_margin: int = 0
    max_attempts: int = 100


@dataclass(frozen=True)
class DiagnosticsConfig:
    enabled: bool = True
    estimate_hurst: bool = False
    articulation_points: bool = False


@dataclass(frozen=True)
class WorldConfig:
    schema_version: int = 1
    generator_version: str = "legacy_v1"
    world_seed: int = 0
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    placement: PlacementConfig = field(default_factory=PlacementConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


@dataclass(frozen=True)
class MapGenerationConfig:
    map_mode: str = "fixed"
    grid_size: tuple[int, int] = (32, 32)
    world: WorldConfig = field(default_factory=WorldConfig)

    @classmethod
    def from_config(cls, config: dict | None, seed: int | None = None) -> "MapGenerationConfig":
        root = config or {}
        env = root.get("env", root)
        map_mode = str(env.get("map_mode", "fixed"))
        if map_mode not in ALLOWED_MAP_MODES:
            raise ValueError(f"env.map_mode: unknown map mode {map_mode!r}")

        grid_size = _grid_size(
            env.get("grid_size", [32, 32]),
            "env.grid_size",
            require_minimum=map_mode == "fractal_percolation",
        )
        raw_world = dict(env.get("world", {}))
        if seed is not None:
            raw_world["world_seed"] = int(seed)
        world = _world_config(raw_world, map_mode)
        return cls(map_mode=map_mode, grid_size=grid_size, world=world)


@dataclass
class MapBuildResult:
    schema_version: int
    map_id: str
    grid: list[list[int]]
    roughness: list[list[float]] | None
    base_pos: tuple[int, int]
    ore_positions: set[tuple[int, int]]
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "map_id": self.map_id,
            "grid": [row[:] for row in self.grid],
            "roughness": None if self.roughness is None else [row[:] for row in self.roughness],
            "base_pos": list(self.base_pos),
            "ore_positions": [list(pos) for pos in sorted(self.ore_positions)],
            "diagnostics": _json_ready(self.diagnostics),
            "provenance": _json_ready(self.provenance),
        }


def stable_map_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _world_config(raw_world: dict[str, Any], map_mode: str) -> WorldConfig:
    generator_version = str(raw_world.get("generator_version", "legacy_v1"))
    if map_mode == "fractal_percolation" and generator_version == "legacy_v1":
        generator_version = "spectral_fbm_v1"
    topology = _topology_config(raw_world.get("topology", {}), map_mode)
    terrain = _terrain_config(raw_world.get("terrain", {}))
    resources = _resources_config(raw_world.get("resources", {}))
    placement = _placement_config(raw_world.get("placement", {}))
    diagnostics = _diagnostics_config(raw_world.get("diagnostics", {}))
    return WorldConfig(
        schema_version=int(raw_world.get("schema_version", 1)),
        generator_version=generator_version,
        world_seed=int(raw_world.get("world_seed", 0)),
        topology=topology,
        terrain=terrain,
        resources=resources,
        placement=placement,
        diagnostics=diagnostics,
    )


def _topology_config(raw: dict[str, Any], map_mode: str) -> TopologyConfig:
    model = str(raw.get("model", "correlated_site"))
    if model not in ALLOWED_TOPOLOGY_MODELS:
        raise ValueError(f"env.world.topology.model: expected one of {sorted(ALLOWED_TOPOLOGY_MODELS)}")
    p_open = _float(raw.get("p_open", 0.65), "env.world.topology.p_open")
    if not (0.0 < p_open <= 1.0):
        raise ValueError("env.world.topology.p_open: expected 0 < p_open <= 1")
    hurst = raw.get("hurst", 0.5)
    hurst_value = None if hurst is None else _hurst(hurst, "env.world.topology.hurst")
    if model == "correlated_site" and hurst_value is None:
        raise ValueError("env.world.topology.hurst: required for correlated_site")
    solvability_mode = str(raw.get("solvability_mode", "conditioned_same_component"))
    if map_mode != "fractal_percolation":
        solvability_mode = str(raw.get("solvability_mode", "legacy_carved"))
    if solvability_mode not in ALLOWED_SOLVABILITY_MODES:
        raise ValueError("env.world.topology.solvability_mode: unsupported value")
    min_fraction = _float(
        raw.get("min_task_component_fraction", 0.2),
        "env.world.topology.min_task_component_fraction",
    )
    if min_fraction < 0.0:
        raise ValueError("env.world.topology.min_task_component_fraction: expected non-negative")
    return TopologyConfig(
        model=model,
        p_open=p_open,
        hurst=hurst_value,
        solvability_mode=solvability_mode,
        task_component=str(raw.get("task_component", "largest")),
        min_task_component_fraction=min_fraction,
        boundary_mode=str(raw.get("boundary_mode", "finite_4_neighbor")),
    )


def _terrain_config(raw: dict[str, Any]) -> TerrainConfig:
    base_move_cost = _float(raw.get("base_move_cost", 0.01), "env.world.terrain.base_move_cost")
    roughness_strength = _float(
        raw.get("roughness_strength", 0.04),
        "env.world.terrain.roughness_strength",
    )
    cost_exponent = _float(raw.get("cost_exponent", 1.0), "env.world.terrain.cost_exponent")
    road_move_cost = _float(raw.get("road_move_cost", 0.0), "env.world.terrain.road_move_cost")
    if base_move_cost < 0.0:
        raise ValueError("env.world.terrain.base_move_cost: expected non-negative")
    if roughness_strength < 0.0:
        raise ValueError("env.world.terrain.roughness_strength: expected non-negative")
    if cost_exponent <= 0.0:
        raise ValueError("env.world.terrain.cost_exponent: expected > 0")
    if road_move_cost < 0.0:
        raise ValueError("env.world.terrain.road_move_cost: expected non-negative")
    bins = tuple(_float(value, "env.world.terrain.observation_bins") for value in raw.get("observation_bins", [0.25, 0.5, 0.75]))
    if any(value <= 0.0 or value >= 1.0 for value in bins) or any(left >= right for left, right in zip(bins, bins[1:])):
        raise ValueError("env.world.terrain.observation_bins: expected strictly increasing values in (0, 1)")
    return TerrainConfig(
        model=str(raw.get("model", "fractional_gaussian_surface")),
        hurst=_hurst(raw.get("hurst", 0.7), "env.world.terrain.hurst"),
        base_move_cost=base_move_cost,
        roughness_strength=roughness_strength,
        cost_exponent=cost_exponent,
        road_move_cost=road_move_cost,
        observation_bins=bins,
    )


def _resources_config(raw: dict[str, Any]) -> ResourcesConfig:
    distribution = str(raw.get("distribution", "clustered"))
    if distribution not in ALLOWED_RESOURCE_DISTRIBUTIONS:
        raise ValueError("env.world.resources.distribution: unsupported value")
    count = int(raw.get("count", 1))
    if count < 1:
        raise ValueError("env.world.resources.count: expected >= 1")
    hurst = raw.get("hurst", 0.7)
    hurst_value = None if hurst is None else _hurst(hurst, "env.world.resources.hurst")
    if distribution == "clustered" and hurst_value is None:
        raise ValueError("env.world.resources.hurst: required for clustered resources")
    min_base_distance = int(raw.get("min_base_distance", 0))
    min_pair_distance = int(raw.get("min_pair_distance", 0))
    if min_base_distance < 0:
        raise ValueError("env.world.resources.min_base_distance: expected non-negative")
    if min_pair_distance < 0:
        raise ValueError("env.world.resources.min_pair_distance: expected non-negative")
    return ResourcesConfig(
        distribution=distribution,
        count=count,
        hurst=hurst_value,
        renewable=bool(raw.get("renewable", True)),
        min_base_distance=min_base_distance,
        min_pair_distance=min_pair_distance,
    )


def _placement_config(raw: dict[str, Any]) -> PlacementConfig:
    base_margin = int(raw.get("base_margin", 0))
    max_attempts = int(raw.get("max_attempts", 100))
    if base_margin < 0:
        raise ValueError("env.world.placement.base_margin: expected non-negative")
    if max_attempts < 1:
        raise ValueError("env.world.placement.max_attempts: expected >= 1")
    return PlacementConfig(base_margin=base_margin, max_attempts=max_attempts)


def _diagnostics_config(raw: dict[str, Any]) -> DiagnosticsConfig:
    return DiagnosticsConfig(
        enabled=bool(raw.get("enabled", True)),
        estimate_hurst=bool(raw.get("estimate_hurst", False)),
        articulation_points=bool(raw.get("articulation_points", False)),
    )


def _grid_size(value: Any, path: str, require_minimum: bool) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{path}: expected [height, width]")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{path}: both dimensions must be positive")
    if require_minimum and (height < 8 or width < 8):
        raise ValueError(f"{path}: both dimensions must be >= 8")
    return height, width


def _hurst(value: Any, path: str) -> float:
    number = _float(value, path)
    if not (0.0 < number < 1.0):
        raise ValueError(f"{path}: expected 0 < H < 1")
    return number


def _float(value: Any, path: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: expected number") from exc


def _json_ready(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return [_json_ready(item) for item in sorted(value)]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
