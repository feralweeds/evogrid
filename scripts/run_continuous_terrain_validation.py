from __future__ import annotations

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import yaml

from evogrid.constants import Action, Tile
from evogrid.envs import EvoGridMineEnv
from evogrid.evaluation.continuous_terrain_gates import (
    evaluate_continuous_terrain_gates,
    write_continuous_terrain_gate_report,
)
from scripts.calibrate_fractal_maps import _runtime_metadata, _stable_hash


def run_continuous_terrain_validation(config_path: str | Path, out_dir: str | Path, seed: int = 0) -> dict[str, Any]:
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    started_at = datetime.now(timezone.utc).isoformat()
    (out_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )

    metrics = {
        "schema_version": 1,
        "protocol_id": "continuous_terrain_B0_B3_v1",
        "seed": int(seed),
        "numeric_cases": _numeric_cases(config),
        "causal_cases": _causal_cases(config, seed),
        "leakage_check": _leakage_check(config, seed),
        "performance_check": _performance_check(config, seed),
    }
    metrics_path = out_dir / "continuous_terrain_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    gate_report = evaluate_continuous_terrain_gates(metrics)
    gate_report_path = write_continuous_terrain_gate_report(
        gate_report,
        out_dir / "continuous_terrain_gates.json",
    )
    manifest = {
        "schema_version": 1,
        "run_id": out_dir.name,
        "experiment_type": "continuous_terrain",
        "mode": "formal",
        "mock_smoke": False,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "resolved_config_hash": _stable_hash(config),
        "seed": int(seed),
        "runtime": _runtime_metadata(),
        "outputs": {
            "config_resolved": "config_resolved.yaml",
            "metrics": "continuous_terrain_metrics.json",
            "gate_report": "continuous_terrain_gates.json",
        },
        "formal_acceptance": {
            "passed": gate_report.passed,
            "conclusion_level": "E2" if gate_report.passed else "E0",
            "gate_report": str(gate_report_path.relative_to(out_dir)).replace("\\", "/"),
        },
        "completion_status": "completed",
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate continuous terrain causal gates.")
    parser.add_argument("--config", default="configs/env_continuous_terrain_formal.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    manifest = run_continuous_terrain_validation(args.config, args.out, args.seed)
    print(json.dumps({"run_manifest": str(Path(args.out) / "run_manifest.json"), "passed": manifest["formal_acceptance"]["passed"]}))


def _numeric_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    env = EvoGridMineEnv(_with_observation(config, mode="full_obs"))
    env.reset(seed=11)
    rows = []
    for roughness in (0.0, 0.5, 1.0):
        assert env.state is not None
        target = _ground_neighbor(env)
        env.state.roughness[target[0]][target[1]] = roughness
        env.state.grid[target[0]][target[1]] = int(Tile.GROUND)
        expected_move_cost = 0.01 + 0.04 * roughness
        expected_saving = expected_move_cost
        expected_break_even = _ceil_division(0.1, expected_saving)
        observed_move_cost = env._move_cost(Tile.GROUND, target)
        observed_road_cost = env._move_cost(Tile.ROAD, target)
        observed_saving = observed_move_cost - observed_road_cost
        rows.append(
            {
                "schema_version": 1,
                "roughness": roughness,
                "expected_move_cost": expected_move_cost,
                "observed_move_cost": observed_move_cost,
                "expected_saving_per_use": expected_saving,
                "observed_saving_per_use": observed_saving,
                "expected_break_even_uses": expected_break_even,
                "observed_break_even_uses": _ceil_division(0.1, observed_saving),
            }
        )
    return rows


def _ceil_division(numerator: float, denominator: float) -> int:
    return int(math.ceil((float(numerator) - 1e-12) / float(denominator)))


def _causal_cases(config: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "no_road": _no_road_case(config, seed),
        "reused_road": _reused_road_case(config, seed),
        "unused_road": _unused_road_case(config, seed),
        "dig_vs_build_road": _dig_vs_build_road_case(config, seed),
    }


def _no_road_case(config: dict[str, Any], seed: int) -> dict[str, Any]:
    env = _prepared_env(config, seed)
    target = _ground_neighbor(env)
    _set_target_roughness(env, target, 0.5)
    _, reward, _, _, _ = env.step(_move_action_to(env, target))
    return {
        "schema_version": 1,
        "target": list(target),
        "move_onto_target_reward": reward,
        "expected_reward": -0.03,
    }


def _reused_road_case(config: dict[str, Any], seed: int) -> dict[str, Any]:
    env = _prepared_env(config, seed)
    assert env.state is not None
    target = _ground_neighbor(env)
    original_base = env.state.base_pos
    _set_target_roughness(env, target, 0.5)
    env.step(_move_action_to(env, target))
    env.step(Action.BUILD_ROAD)
    record = env.state.road_credit_tracker.records[target]
    rewards = []
    for _ in range(4):
        env.step(_move_action_to(env, original_base))
        _, reward, _, _, _ = env.step(_move_action_to(env, target))
        rewards.append(reward)
    dropoff_before = _dropoff_reward(config, seed)
    dropoff_after = _dropoff_reward(config, seed, build_road=True)
    return {
        "schema_version": 1,
        "target": list(target),
        "move_onto_target_reward_after_road": rewards[-1],
        "road_usage_count": record.usage_count,
        "road_saved_cost": record.saved_cost,
        "road_build_cost": record.build_cost,
        "road_net_payoff": record.net_payoff,
        "dropoff_reward_before": dropoff_before,
        "dropoff_reward_after": dropoff_after,
        "dropoff_reward_changed": abs(dropoff_before - dropoff_after) > 1e-12,
    }


def _unused_road_case(config: dict[str, Any], seed: int) -> dict[str, Any]:
    env = _prepared_env(config, seed)
    target = _ground_neighbor(env)
    _set_target_roughness(env, target, 0.5)
    env.step(_move_action_to(env, target))
    env.step(Action.BUILD_ROAD)
    assert env.state is not None
    record = env.state.road_credit_tracker.records[target]
    return {
        "schema_version": 1,
        "target": list(target),
        "road_usage_count": record.usage_count,
        "road_saved_cost": record.saved_cost,
        "road_build_cost": record.build_cost,
        "road_net_payoff": record.net_payoff,
    }


def _dig_vs_build_road_case(config: dict[str, Any], seed: int) -> dict[str, Any]:
    build_env = _prepared_env(config, seed)
    assert build_env.state is not None
    obstacle = _ground_neighbor(build_env)
    build_env.state.grid[obstacle[0]][obstacle[1]] = int(Tile.OBSTACLE)
    _, _, _, _, build_info = build_env.step(Action.BUILD_ROAD)
    build_road_opened = build_env.state.tile_at(obstacle) != Tile.OBSTACLE

    dig_env = _prepared_env(config, seed)
    assert dig_env.state is not None
    dig_target = _ground_neighbor(dig_env)
    dig_env.state.grid[dig_target[0]][dig_target[1]] = int(Tile.OBSTACLE)
    dig_env.step(Action.DIG)
    dig_opened = dig_env.state.tile_at(dig_target) != Tile.OBSTACLE
    return {
        "schema_version": 1,
        "build_road_attempt_invalid_actions": build_info["invalid_actions"],
        "build_road_opened_obstacle": build_road_opened,
        "dig_opened_obstacle": dig_opened,
    }


def _leakage_check(config: dict[str, Any], seed: int) -> dict[str, Any]:
    env = EvoGridMineEnv(_with_observation(config, mode="partial_obs", radius=4))
    obs, info = env.reset(seed=seed)
    hidden = {
        "route_rough_tile_count",
        "off_route_rough_tile_count",
        "positive_road_opportunity_count",
        "transport_corridor_length",
        "shortest_path_length",
        "largest_component_fraction",
    }
    visible = obs.get("visible_tiles", [])
    return {
        "schema_version": 1,
        "agent_info_hidden_keys": sorted(hidden & set(info)),
        "partial_observation_has_full_grid": "grid" in obs,
        "partial_observation_has_continuous_roughness": any("roughness" in item for item in visible),
        "partial_observation_has_terrain_band": any("terrain_band" in item for item in visible),
    }


def _performance_check(config: dict[str, Any], seed: int) -> dict[str, Any]:
    perf_config = _with_observation(config, mode="partial_obs", radius=4)
    perf_config["env"]["grid_size"] = [128, 128]
    perf_config["env"]["max_steps"] = 250
    env = EvoGridMineEnv(perf_config)
    env.reset(seed=seed)
    calls = 0
    original = env._map_diagnostics

    def counted():
        nonlocal calls
        calls += 1
        return original()

    env._map_diagnostics = counted
    for _ in range(200):
        env.step(Action.NOOP)
    step_calls = calls
    env.get_audit_snapshot()
    return {
        "schema_version": 1,
        "step_count": 200,
        "static_diagnostic_calls_during_steps": step_calls,
        "static_diagnostic_calls_after_audit_snapshot": calls,
    }


def _prepared_env(config: dict[str, Any], seed: int) -> EvoGridMineEnv:
    env = EvoGridMineEnv(_with_observation(config, mode="full_obs"))
    env.reset(seed=seed)
    assert env.state is not None
    row, col = env.state.base_pos
    for pos in ((row, col + 1), (row, col - 1), (row + 1, col), (row - 1, col)):
        if env.state.in_bounds(pos):
            env.state.grid[pos[0]][pos[1]] = int(Tile.GROUND)
            env.state.roughness[pos[0]][pos[1]] = 0.5
    return env


def _ground_neighbor(env: EvoGridMineEnv) -> tuple[int, int]:
    assert env.state is not None
    row, col = env.state.agent_pos
    for pos in ((row, col + 1), (row, col - 1), (row + 1, col), (row - 1, col)):
        if env.state.in_bounds(pos):
            env.state.grid[pos[0]][pos[1]] = int(Tile.GROUND)
            return pos
    raise RuntimeError("no in-bounds neighbor")


def _set_target_roughness(env: EvoGridMineEnv, target: tuple[int, int], roughness: float) -> None:
    assert env.state is not None and env.state.roughness is not None
    env.state.grid[target[0]][target[1]] = int(Tile.GROUND)
    env.state.roughness[target[0]][target[1]] = float(roughness)


def _move_action_to(env: EvoGridMineEnv, target: tuple[int, int]) -> Action:
    assert env.state is not None
    row, col = env.state.agent_pos
    target_row, target_col = target
    delta = (target_row - row, target_col - col)
    actions = {
        (0, 1): Action.MOVE_RIGHT,
        (0, -1): Action.MOVE_LEFT,
        (1, 0): Action.MOVE_DOWN,
        (-1, 0): Action.MOVE_UP,
    }
    if delta not in actions:
        raise RuntimeError(f"target is not adjacent: {env.state.agent_pos} -> {target}")
    return actions[delta]


def _dropoff_reward(config: dict[str, Any], seed: int, build_road: bool = False) -> float:
    env = _prepared_env(config, seed)
    assert env.state is not None
    if build_road:
        target = _ground_neighbor(env)
        env.step(_move_action_to(env, target))
        env.step(Action.BUILD_ROAD)
        env.step(_move_action_to(env, env.state.base_pos))
    env.state.has_ore = True
    _, reward, _, _, _ = env.step(Action.DROPOFF)
    return reward


def _with_observation(config: dict[str, Any], mode: str, radius: int = 4) -> dict[str, Any]:
    updated = deepcopy(config)
    updated.setdefault("env", {})
    updated["env"]["observation"] = {
        "mode": mode,
        "local_view_radius": radius,
        "expose_continuous_roughness": False,
    }
    return updated


if __name__ == "__main__":
    main()
