from __future__ import annotations

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evogrid.constants import Tile
from evogrid.evaluation.skill_verifier import SkillVerifier, SkillVerificationProtocol
from evogrid.skills.registry import SkillRegistry
from evogrid.skills.schemas import SkillSpec
from scripts.run_skill_verification import (
    _load_rollout_env_config,
    _rollout_stratum,
    _run_episode,
)
from evogrid.agents.route_only_agent import RouteOnlyAgent
from evogrid.agents.skill_agent import SkillAgent
from evogrid.skills.runtime import SkillRuntime


PRIMARY_METRIC = "road_net_payoff"


def run_handcrafted_route_target_rollout(out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = handcrafted_route_target_candidate()
    candidate_path = out_dir / "candidate.json"
    candidate_path.write_text(
        json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    registry = SkillRegistry(out_dir / "registry")
    registry.register_candidate(candidate)
    evaluator, episode_rows, trace_rows = _recording_rollout_evaluator(
        candidate=candidate,
        out_dir=out_dir,
        verification=_verification_config(),
    )
    protocol = _protocol()
    paired_seeds = list(range(5200, 5220))
    report = SkillVerifier(protocol).verify(
        candidate,
        paired_seeds=paired_seeds,
        evaluator=evaluator,
        source_train_seeds=set(),
        environment_strata=["rollout_road_positive", "rollout_road_negative"],
    )
    lease = registry.begin_verification(candidate.skill_id, candidate.version)
    record = registry.apply_verification(report, lease_id=lease.lease_id)
    metrics_path = out_dir / "episodes" / "metrics.csv"
    trace_path = out_dir / "skills" / "skill_trace.jsonl"
    _write_metrics_csv(metrics_path, episode_rows)
    _write_trace_jsonl(trace_path, trace_rows)
    report_ref = record.verification_reports[-1].replace("\\", "/") if record.verification_reports else ""
    rollout_summary = _rollout_summary(episode_rows)
    manifest = {
        "schema_version": 1,
        "experiment_type": "handcrafted_route_target_rollout",
        "mode": "pilot",
        "candidate": str(candidate_path.as_posix()),
        "candidate_hash": candidate.spec_hash,
        "decision": report.decision,
        "promoted_status": record.spec.status,
        "report": report_ref,
        "paired_seeds": paired_seeds,
        "sample_size": report.sample_size,
        "environment_strata": report.environment_strata,
        "metrics": {
            "paired_delta_mean": report.metrics.get("paired_delta_mean", 0.0),
            "success_rate": report.metrics.get("success_rate", 0.0),
            "false_trigger_rate": report.metrics.get("false_trigger_rate", 0.0),
            "activation_rate": report.metrics.get("activation_rate", 0.0),
            "runtime_failure_rate": report.metrics.get("runtime_failure_rate", 0.0),
        },
        "rollout_summary": rollout_summary,
        "outputs": {
            "metrics_csv": str(metrics_path.as_posix()),
            "skill_trace_jsonl": str(trace_path.as_posix()),
            "registry": str((out_dir / "registry").as_posix()),
        },
        "scope": "real EvoGrid rollout pilot; not a final/test conclusion",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def handcrafted_route_target_candidate() -> SkillSpec:
    return SkillSpec.from_dict(
        {
            "schema_version": 1,
            "skill_id": "handcrafted_route_target_road",
            "version": "1.0.0",
            "status": "candidate",
            "name": "Handcrafted route target road",
            "description": "Select a visible rough route tile, move to it, and build one road.",
            "problem_addressed": "Repeated transport over observed rough route tiles.",
            "source": {
                "proposer": "handcrafted",
                "source_episode_ids": ["r5/route_target_rollout_pilot"],
                "partition": "train",
            },
            "applicability": {
                "all": [
                    {"feature": "route.exists", "op": "eq", "value": True},
                    {"feature": "route.is_known_transport_route", "op": "eq", "value": True},
                    {"feature": "route.remaining_length_bucket", "op": "in", "value": ["medium", "long"]},
                ]
            },
            "procedure": [
                {
                    "op": "SELECT_TARGET",
                    "source": "route.observed_tiles",
                    "filters": [
                        {"feature": "candidate.has_road", "op": "eq", "value": False},
                        {"feature": "candidate.tile_type", "op": "eq", "value": int(Tile.ROUGH)},
                    ],
                    "rank_by": [
                        {"feature": "candidate.visit_count_bucket", "direction": "desc"},
                        {"feature": "candidate.route_order", "direction": "asc"},
                    ],
                    "select": "first",
                    "store_as": "target",
                    "episode_store_as": "road_target",
                },
                {
                    "op": "IF",
                    "condition": {"left": {"var": "target"}, "op": "ne", "right": None},
                    "then": [
                        {
                            "op": "PLAN_ROUTE",
                            "target": {"var": "target"},
                            "unknown_cell_policy": "allow",
                            "max_length": 32,
                            "store_as": "route_to_target",
                        },
                        {"op": "FOLLOW_ROUTE", "route_var": "route_to_target", "max_steps": 1},
                        {"op": "ACT", "action": "BUILD_ROAD"},
                    ],
                    "else": [{"op": "RETURN", "result": "no_route_target"}],
                },
            ],
            "budget": {
                "max_runtime_steps": 6,
                "max_environment_actions": 1,
                "max_nested_skill_depth": 0,
                "max_uses_per_episode": 1,
                "episode_use_actions": ["BUILD_ROAD"],
                "stop_after_success": True,
            },
            "objective": {
                "primary_metric": PRIMARY_METRIC,
                "direction": "maximize",
                "negative_context_metric": "false_trigger_rate",
            },
            "dependencies": [],
            "rationale": "R5 pilot fixture using generic route-local target selection DSL.",
        }
    )


def _recording_rollout_evaluator(
    *,
    candidate: SkillSpec,
    out_dir: Path,
    verification: dict[str, Any],
):
    env_config = _load_rollout_env_config(verification, {"verification": verification})
    registry = SkillRegistry(out_dir / "evaluator_registry")
    registry.register_candidate(candidate)
    runtime = SkillRuntime()
    episode_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    def evaluator(seed: int, enabled: bool) -> dict[str, Any]:
        fallback = RouteOnlyAgent()
        agent = SkillAgent(registry, fallback, runtime=runtime, allow_candidates=True) if enabled else fallback
        metrics, trace = _run_episode(env_config, agent, seed)
        steps = max(1, int(metrics.get("steps", 0) or 0))
        skill_traces = [item for item in trace if item.get("source") == "skill"]
        runtime_failures = [
            item
            for item in skill_traces
            if item.get("runtime", {}).get("termination")
            not in {"completed", "completed_no_action", "returned", "not_applicable", "episode_use_limit_reached", "episode_stop_after_success"}
        ]
        road_builds = int(metrics.get("num_build_road", 0) or 0)
        road_net_payoff = float(metrics.get(PRIMARY_METRIC, 0.0) or 0.0)
        stratum = _rollout_stratum(metrics)
        row = {
            "seed": seed,
            "enabled": enabled,
            "stratum": stratum,
            PRIMARY_METRIC: road_net_payoff,
            "episode_reward": float(metrics.get("episode_reward", 0.0) or 0.0),
            "ore_delivered": int(metrics.get("ore_delivered", 0) or 0),
            "num_build_road": road_builds,
            "road_usage_rate": float(metrics.get("road_usage_rate", 0.0) or 0.0),
            "road_total_usage_count": int(metrics.get("road_total_usage_count", 0) or 0),
            "runtime_failure_rate": len(runtime_failures) / max(1, len(skill_traces)),
            "invalid_action_rate": float(metrics.get("invalid_actions", 0) or 0) / steps,
            "activation_rate": 1.0 if skill_traces else 0.0,
            "false_trigger_rate": 1.0 if enabled and road_builds > 0 and road_net_payoff <= 0.0 else 0.0,
        }
        episode_rows.append(row)
        for item in skill_traces:
            trace_rows.append({"seed": seed, "enabled": enabled, **item})
        return {
            PRIMARY_METRIC: road_net_payoff,
            "false_trigger_rate": row["false_trigger_rate"],
            "runtime_failure_rate": row["runtime_failure_rate"],
            "invalid_action_rate": row["invalid_action_rate"],
            "activation_rate": row["activation_rate"],
            "redundancy_score": 0.0,
            "stratum": stratum,
            "episode_reward": row["episode_reward"],
            "ore_delivered": row["ore_delivered"],
            "num_build_road": road_builds,
            "road_usage_rate": row["road_usage_rate"],
        }

    return evaluator, episode_rows, trace_rows


def _protocol() -> SkillVerificationProtocol:
    return SkillVerificationProtocol(
        protocol_id="skill_verification_v1",
        primary_metric=PRIMARY_METRIC,
        direction="maximize",
        min_effect=0.0,
        min_success_rate=0.25,
        max_false_trigger_rate=0.35,
        min_paired_samples=20,
        bootstrap_iterations=300,
        ci_level=0.95,
        max_runtime_failure_rate=0.1,
        max_invalid_action_rate_delta=0.05,
        min_activation_rate=0.05,
        requires_transfer=False,
        max_redundancy_score=1.0,
        bootstrap_seed=51,
    )


def _verification_config() -> dict[str, Any]:
    return {
        "episode_max_steps": 220,
        "env_config": {
            "env": {
                "map_mode": "controlled_corridor_curriculum",
                "grid_size": [16, 16],
                "max_steps": 220,
                "base_pos": [2, 2],
                "random_map": {
                    "ore_count": 1,
                    "min_base_ore_distance": 9,
                    "obstacle_density": 0.04,
                    "extra_rough_density": 0.04,
                    "ensure_reachable": True,
                    "max_generation_attempts": 100,
                    "controlled_corridor": {
                        "positive_weight": 0.7,
                        "mixed_weight": 0.2,
                        "negative_weight": 0.1,
                        "base_margin": 2,
                        "positive_route_rough_probability": 0.8,
                        "mixed_route_rough_probability": 0.38,
                        "negative_route_rough_probability": 0.03,
                        "positive_off_route_rough_probability": 0.05,
                        "mixed_off_route_rough_probability": 0.18,
                        "negative_off_route_rough_probability": 0.30,
                        "positive_min_route_rough": 4,
                        "mixed_min_route_rough": 2,
                        "negative_min_route_rough": 0,
                        "positive_transport_band_probability": 0.75,
                        "mixed_transport_band_probability": 0.35,
                        "negative_transport_band_probability": 0.0,
                    },
                },
                "observation": {"mode": "partial_obs", "local_view_radius": 4},
                "shaping": {"allow_dig": True, "allow_build_road": True, "reset_after_dropoff": False},
                "rewards": {
                    "dropoff": 10.0,
                    "move_ground": -0.01,
                    "move_rough": -0.05,
                    "move_road": 0.0,
                    "dig": -0.2,
                    "build_road": -0.1,
                    "mine": -0.01,
                    "dropoff_action": -0.01,
                    "noop": -0.01,
                    "invalid_action": -0.05,
                },
            }
        },
    }


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "enabled",
        "stratum",
        PRIMARY_METRIC,
        "episode_reward",
        "ore_delivered",
        "num_build_road",
        "road_usage_rate",
        "road_total_usage_count",
        "runtime_failure_rate",
        "invalid_action_rate",
        "activation_rate",
        "false_trigger_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _rollout_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    enabled = [row for row in rows if bool(row.get("enabled"))]
    disabled = [row for row in rows if not bool(row.get("enabled"))]
    enabled_build_rows = [row for row in enabled if int(row.get("num_build_road", 0) or 0) > 0]
    return {
        "enabled_episode_count": len(enabled),
        "disabled_episode_count": len(disabled),
        "enabled_road_builds": sum(int(row.get("num_build_road", 0) or 0) for row in enabled),
        "disabled_road_builds": sum(int(row.get("num_build_road", 0) or 0) for row in disabled),
        "enabled_positive_road_net_episodes": sum(
            1 for row in enabled_build_rows if float(row.get(PRIMARY_METRIC, 0.0) or 0.0) > 0.0
        ),
        "enabled_nonpositive_road_net_episodes": sum(
            1 for row in enabled_build_rows if float(row.get(PRIMARY_METRIC, 0.0) or 0.0) <= 0.0
        ),
        "enabled_road_total_usage_count": sum(int(row.get("road_total_usage_count", 0) or 0) for row in enabled),
        "enabled_road_net_sum": sum(float(row.get(PRIMARY_METRIC, 0.0) or 0.0) for row in enabled),
    }


def _write_trace_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real rollout for the handcrafted R5 route-target road Skill.")
    parser.add_argument("--out", default="outputs/r5_handcrafted_route_target_rollout")
    args = parser.parse_args()
    manifest = run_handcrafted_route_target_rollout(args.out)
    print(json.dumps({"run_manifest": str(Path(args.out) / "run_manifest.json"), "decision": manifest["decision"]}))


if __name__ == "__main__":
    main()
