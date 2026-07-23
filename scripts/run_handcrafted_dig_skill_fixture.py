from __future__ import annotations

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evogrid.constants import Tile
from evogrid.evaluation.skill_verifier import SkillVerifier, SkillVerificationProtocol
from evogrid.skills.context import SkillContext
from evogrid.skills.registry import SkillRegistry
from evogrid.skills.runtime import SkillEpisodeState, SkillRuntime
from evogrid.skills.schemas import SkillSpec


PRIMARY_METRIC = "dig_success"
POSITIVE_STRATUM = "adjacent_obstacle_context"
NEGATIVE_STRATUM = "no_adjacent_obstacle_context"
SAFE_TERMINATIONS = {
    "completed",
    "completed_no_action",
    "returned",
    "not_applicable",
    "episode_use_limit_reached",
    "episode_stop_after_success",
    "episode_intervention_limit_reached",
    "no_progress_detected",
}


def run_handcrafted_dig_skill_fixture(out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = SkillRegistry(out_dir / "registry")
    candidate = handcrafted_dig_candidate()
    candidate_path = out_dir / "candidate.json"
    candidate_path.write_text(
        json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    registered = registry.register_candidate(candidate)
    lease = registry.begin_verification(registered.spec.skill_id, registered.spec.version)
    protocol = SkillVerificationProtocol(
        protocol_id="skill_verification_v1",
        primary_metric=PRIMARY_METRIC,
        direction="maximize",
        min_effect=0.1,
        min_success_rate=0.5,
        max_false_trigger_rate=0.0,
        min_paired_samples=60,
        bootstrap_iterations=300,
        ci_level=0.95,
        max_runtime_failure_rate=0.0,
        max_invalid_action_rate_delta=0.0,
        min_activation_rate=0.5,
        requires_transfer=False,
        max_redundancy_score=0.1,
        bootstrap_seed=71,
    )
    report = SkillVerifier(protocol).verify(
        registered.spec,
        paired_seeds=list(range(60)),
        evaluator=_dig_context_fixture_evaluator(registered.spec),
        source_train_seeds=set(),
        environment_strata=[POSITIVE_STRATUM, NEGATIVE_STRATUM],
    )
    promoted = registry.apply_verification(report, lease_id=lease.lease_id)
    report_ref = promoted.verification_reports[-1].replace("\\", "/") if promoted.verification_reports else ""
    manifest = {
        "schema_version": 1,
        "experiment_type": "handcrafted_dig_skill_fixture",
        "mode": "development_fixture",
        "candidate": str(candidate_path.as_posix()),
        "candidate_hash": candidate.spec_hash,
        "decision": report.decision,
        "promoted_status": promoted.spec.status,
        "report": report_ref,
        "sample_size": report.sample_size,
        "environment_strata": report.environment_strata,
        "metrics": {
            "paired_delta_mean": report.metrics.get("paired_delta_mean", 0.0),
            "success_rate": report.metrics.get("success_rate", 0.0),
            "false_trigger_rate": report.metrics.get("false_trigger_rate", 0.0),
            "activation_rate": report.metrics.get("activation_rate", 0.0),
            "runtime_failure_rate": report.metrics.get("runtime_failure_rate", 0.0),
        },
        "scope": "synthetic DIG fixture for DSL/runtime/verifier plumbing; not real rollout or formal DIG Skill claim",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def handcrafted_dig_candidate() -> SkillSpec:
    return SkillSpec.from_dict(
        {
            "schema_version": 1,
            "skill_id": "handcrafted_adjacent_obstacle_dig",
            "version": "1.0.0",
            "status": "candidate",
            "name": "Handcrafted adjacent obstacle dig",
            "description": "Select an adjacent visible obstacle and dig once.",
            "problem_addressed": "Local topology repair by opening an observed adjacent obstacle.",
            "source": {
                "proposer": "handcrafted",
                "source_episode_ids": ["r5/dig_fixture"],
                "partition": "train",
            },
            "applicability": {
                "all": [
                    {"feature": "local.adjacent_obstacle_count", "op": "gt", "value": 0},
                    {"feature": "episode_budget.steps_remaining", "op": "gt", "value": 1},
                ]
            },
            "procedure": [
                {
                    "op": "SELECT_TARGET",
                    "source": "visible_tiles",
                    "filters": [
                        {"feature": "candidate.tile_type", "op": "eq", "value": int(Tile.OBSTACLE)},
                        {"feature": "candidate.distance_from_agent", "op": "lte", "value": 1},
                    ],
                    "rank_by": [{"feature": "candidate.distance_from_agent", "direction": "asc"}],
                    "select": "first",
                    "store_as": "target",
                    "episode_store_as": "dig_target",
                },
                {
                    "op": "IF",
                    "condition": {"left": {"var": "target"}, "op": "ne", "right": None},
                    "then": [{"op": "ACT", "action": "DIG"}],
                    "else": [{"op": "RETURN", "result": "no_adjacent_obstacle"}],
                },
            ],
            "budget": {
                "max_runtime_steps": 4,
                "max_environment_actions": 1,
                "max_nested_skill_depth": 0,
                "max_uses_per_episode": 1,
                "episode_use_actions": ["DIG"],
                "stop_after_success": True,
                "max_consecutive_interventions": 2,
            },
            "objective": {
                "primary_metric": PRIMARY_METRIC,
                "direction": "maximize",
                "negative_context_metric": "false_trigger_rate",
            },
            "dependencies": [],
            "rationale": "R5 DIG fixture using generic visible target selection and primitive DIG.",
        }
    )


def _dig_context_fixture_evaluator(candidate: SkillSpec):
    runtime = SkillRuntime()

    def evaluator(seed: int, enabled: bool) -> dict[str, Any]:
        positive_context = bool(seed % 2)
        stratum = POSITIVE_STRATUM if positive_context else NEGATIVE_STRATUM
        if not enabled:
            return _row(stratum=stratum, value=0.0, activation=0.0)

        context = _context(positive=positive_context)
        episode_state = SkillEpisodeState()
        first = runtime.execute(candidate, context, allow_candidate=True, episode_state=episode_state)
        second = runtime.execute(candidate, context, allow_candidate=True, episode_state=episode_state)
        dug = first.chosen_action == "DIG"
        guard_blocked = second.termination in {"episode_use_limit_reached", "episode_stop_after_success", "not_applicable"}
        value = 1.0 if positive_context and dug and guard_blocked else 0.0
        false_trigger = 1.0 if not positive_context and dug else 0.0
        runtime_failure = 0.0 if first.termination in SAFE_TERMINATIONS and guard_blocked else 1.0
        activation = 1.0 if first.trace.applicable and first.termination != "not_applicable" else 0.0
        return _row(
            stratum=stratum,
            value=value,
            activation=activation,
            false_trigger=false_trigger,
            runtime_failure=runtime_failure,
            first_termination=first.termination,
            second_termination=second.termination,
            first_action=first.chosen_action,
            selected_target=first.variables.get("target"),
        )

    return evaluator


def _row(
    *,
    stratum: str,
    value: float,
    activation: float,
    false_trigger: float = 0.0,
    runtime_failure: float = 0.0,
    first_termination: str | None = None,
    second_termination: str | None = None,
    first_action: str | None = None,
    selected_target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        PRIMARY_METRIC: value,
        "false_trigger_rate": false_trigger,
        "runtime_failure_rate": runtime_failure,
        "invalid_action_rate": 0.0,
        "activation_rate": activation,
        "redundancy_score": 0.0,
        "stratum": stratum,
        "first_termination": first_termination,
        "second_termination": second_termination,
        "first_action": first_action,
        "selected_target": selected_target,
    }


def _context(*, positive: bool) -> SkillContext:
    visible_tiles = [
        {"pos": [2, 2], "tile": int(Tile.GROUND), "terrain_band": "SMOOTH"},
    ]
    if positive:
        visible_tiles.append({"pos": [2, 3], "tile": int(Tile.OBSTACLE), "terrain_band": "SMOOTH"})
    else:
        visible_tiles.append({"pos": [2, 3], "tile": int(Tile.GROUND), "terrain_band": "SMOOTH"})
    return SkillContext.from_observable_inputs(
        observation={
            "agent_pos": [2, 2],
            "base_pos": [1, 1],
            "has_ore": False,
            "visible_tiles": visible_tiles,
        },
        info={},
        memory_summary={},
        route_plan={"exists": True},
        episode_budget={"steps_remaining": 20},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the handcrafted DIG Skill fixture.")
    parser.add_argument("--out", default="outputs/r5_handcrafted_dig_fixture")
    args = parser.parse_args()
    manifest = run_handcrafted_dig_skill_fixture(args.out)
    print(json.dumps({"run_manifest": str(Path(args.out) / "run_manifest.json"), "decision": manifest["decision"]}))


if __name__ == "__main__":
    main()
