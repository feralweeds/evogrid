"""Formal gate checks for continuous-terrain causal validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContinuousTerrainGate:
    gate_id: str
    name: str
    passed: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass(frozen=True)
class ContinuousTerrainGateReport:
    schema_version: int
    gates: list[ContinuousTerrainGate]

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": "continuous_terrain_B0_B3_v1",
            "passed": self.passed,
            "conclusion_level_if_passed": "E2",
            "gates": [gate.to_dict() for gate in self.gates],
        }


def evaluate_continuous_terrain_gates(
    payload: dict[str, Any],
    *,
    tolerance: float = 1e-9,
) -> ContinuousTerrainGateReport:
    gates = [
        _gate_b0(payload),
        _gate_b1(payload, tolerance),
        _gate_b2(payload, tolerance),
        _gate_b3(payload),
    ]
    return ContinuousTerrainGateReport(schema_version=1, gates=gates)


def write_continuous_terrain_gate_report(report: ContinuousTerrainGateReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _gate_b0(payload: dict[str, Any]) -> ContinuousTerrainGate:
    required = {"numeric_cases", "causal_cases", "leakage_check", "performance_check"}
    missing = sorted(required - set(payload))
    return ContinuousTerrainGate(
        "B0",
        "data_integrity",
        not missing and payload.get("schema_version") == 1,
        {
            "missing_sections": missing,
            "schema_version": payload.get("schema_version"),
        },
    )


def _gate_b1(payload: dict[str, Any], tolerance: float) -> ContinuousTerrainGate:
    failures = []
    for row in payload.get("numeric_cases", []):
        for key in ("move_cost", "saving_per_use"):
            expected = float(row.get(f"expected_{key}", 0.0))
            observed = float(row.get(f"observed_{key}", 0.0))
            if abs(observed - expected) > tolerance:
                failures.append(
                    {
                        "roughness": row.get("roughness"),
                        "metric": key,
                        "expected": expected,
                        "observed": observed,
                    }
                )
        if int(row.get("observed_break_even_uses", -1)) != int(row.get("expected_break_even_uses", -2)):
            failures.append(
                {
                    "roughness": row.get("roughness"),
                    "metric": "break_even_uses",
                    "expected": row.get("expected_break_even_uses"),
                    "observed": row.get("observed_break_even_uses"),
                }
            )
    return ContinuousTerrainGate(
        "B1",
        "numeric_cost_table",
        bool(payload.get("numeric_cases")) and not failures,
        {"case_count": len(payload.get("numeric_cases", [])), "failures": failures},
    )


def _gate_b2(payload: dict[str, Any], tolerance: float) -> ContinuousTerrainGate:
    cases = payload.get("causal_cases", {})
    reused = cases.get("reused_road", {})
    no_road = cases.get("no_road", {})
    unused = cases.get("unused_road", {})
    dig = cases.get("dig_vs_build_road", {})
    failures = []
    if float(reused.get("road_net_payoff", 0.0)) <= 0.0:
        failures.append("reused road did not produce positive net payoff")
    if float(unused.get("road_net_payoff", 0.0)) >= 0.0:
        failures.append("unused road did not preserve negative build-cost payoff")
    if float(no_road.get("move_onto_target_reward", 0.0)) >= float(reused.get("move_onto_target_reward_after_road", 0.0)) - tolerance:
        failures.append("road did not improve later movement reward")
    if bool(dig.get("build_road_opened_obstacle")):
        failures.append("BUILD_ROAD changed blocked/open topology")
    if not bool(dig.get("dig_opened_obstacle")):
        failures.append("DIG did not open blocked topology")
    if bool(reused.get("dropoff_reward_changed", True)):
        failures.append("road changed dropoff reward")
    return ContinuousTerrainGate(
        "B2",
        "causal_road_payoff",
        not failures,
        {
            "failures": failures,
            "reused_road_net_payoff": reused.get("road_net_payoff"),
            "unused_road_net_payoff": unused.get("road_net_payoff"),
            "no_road_move_reward": no_road.get("move_onto_target_reward"),
            "road_move_reward": reused.get("move_onto_target_reward_after_road"),
        },
    )


def _gate_b3(payload: dict[str, Any]) -> ContinuousTerrainGate:
    leakage = payload.get("leakage_check", {})
    performance = payload.get("performance_check", {})
    failures = []
    if leakage.get("agent_info_hidden_keys"):
        failures.append("agent info contains hidden keys")
    if leakage.get("partial_observation_has_full_grid"):
        failures.append("partial observation contains full grid")
    if leakage.get("partial_observation_has_continuous_roughness"):
        failures.append("partial observation exposes continuous roughness")
    if int(performance.get("static_diagnostic_calls_during_steps", -1)) != 0:
        failures.append("static diagnostics recomputed during partial steps")
    if int(performance.get("step_count", 0)) < 200:
        failures.append("performance check did not execute 200 steps")
    return ContinuousTerrainGate(
        "B3",
        "partial_observation_and_performance",
        not failures,
        {
            "failures": failures,
            "leakage_check": leakage,
            "performance_check": performance,
        },
    )
