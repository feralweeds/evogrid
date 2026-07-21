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


PRIMARY_METRIC = "road_net_payoff"
POSITIVE_STRATUM = "positive_transport_context"
NEGATIVE_STRATUM = "negative_transport_context"


def run_handcrafted_positive_skill_fixture(out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = SkillRegistry(out_dir / "registry")
    candidate = _handcrafted_candidate()
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
        bootstrap_seed=17,
    )
    report = SkillVerifier(protocol).verify(
        registered.spec,
        paired_seeds=list(range(60)),
        evaluator=_transport_context_fixture_evaluator(registered.spec),
        source_train_seeds=set(),
        environment_strata=[POSITIVE_STRATUM, NEGATIVE_STRATUM],
    )
    promoted = registry.apply_verification(report, lease_id=lease.lease_id)
    manifest = {
        "schema_version": 1,
        "experiment_type": "handcrafted_positive_skill_fixture",
        "candidate_id": candidate.skill_id,
        "candidate_version": candidate.version,
        "candidate_hash": candidate.spec_hash,
        "verified": str((out_dir / "registry" / "verified" / candidate.skill_id / f"{candidate.version}.json").as_posix()),
        "report": promoted.verification_reports[-1].replace("\\", "/"),
        "decision": report.decision,
        "promoted_status": promoted.spec.status,
        "sample_size": report.sample_size,
        "environment_strata": [POSITIVE_STRATUM, NEGATIVE_STRATUM],
        "metrics": {
            "paired_delta_mean": report.metrics["paired_delta_mean"],
            "success_rate": report.metrics["success_rate"],
            "false_trigger_rate": report.metrics["false_trigger_rate"],
            "activation_rate": report.metrics["activation_rate"],
            "runtime_failure_rate": report.metrics["runtime_failure_rate"],
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _transport_context_fixture_evaluator(candidate: SkillSpec):
    runtime = SkillRuntime(
        estimators={
            "future_route_uses": lambda context, variables: 5,
            "road_break_even_uses": lambda context, variables: 3,
        }
    )

    def evaluator(seed: int, enabled: bool) -> dict[str, Any]:
        positive_context = bool(seed % 2)
        stratum = POSITIVE_STRATUM if positive_context else NEGATIVE_STRATUM
        if not enabled:
            return _row(stratum=stratum, value=0.0, activation=0.0)

        context = _context(positive=positive_context)
        episode_state = SkillEpisodeState()
        first = runtime.execute(candidate, context, allow_candidate=True, episode_state=episode_state)
        second = runtime.execute(candidate, context, allow_candidate=True, episode_state=episode_state)
        built = first.chosen_action == "BUILD_ROAD"
        guard_blocked = second.termination in {"episode_use_limit_reached", "episode_stop_after_success", "not_applicable"}
        runtime_failure = 0.0 if first.termination in _SAFE_TERMINATIONS and guard_blocked else 1.0
        value = 1.0 if positive_context and built and guard_blocked else 0.0
        false_trigger = 1.0 if not positive_context and built else 0.0
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
        )

    return evaluator


_SAFE_TERMINATIONS = {
    "completed",
    "completed_no_action",
    "returned",
    "not_applicable",
    "episode_use_limit_reached",
    "episode_stop_after_success",
}


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
    }


def _context(*, positive: bool) -> SkillContext:
    terrain_band = "ROUGH" if positive else "SMOOTH"
    return SkillContext.from_observable_inputs(
        observation={
            "agent_pos": [2, 2],
            "base_pos": [1, 1],
            "has_ore": False,
            "visible_tiles": [
                {"pos": [2, 2], "tile": int(Tile.GROUND), "terrain_band": terrain_band},
            ],
        },
        info={},
        memory_summary={"similar_mean_payoff": 0.5, "similar_outcome_count": 4, "visit_count_bucket": "medium"},
        route_plan={
            "exists": True,
            "is_known_transport_route": True,
            "remaining_length_bucket": "medium",
        },
        episode_budget={"steps_remaining": 50},
    )


def _handcrafted_candidate() -> SkillSpec:
    return SkillSpec.from_dict(
        {
            "schema_version": 1,
            "skill_id": "handcrafted_positive_transport_road",
            "version": "1.0.0",
            "status": "candidate",
            "name": "Handcrafted positive transport road",
            "description": "Fixture candidate that builds one road only in a known positive transport context.",
            "problem_addressed": "Verify that an effective transport-road Skill can be promoted.",
            "source": {
                "proposer": "handcrafted",
                "source_episode_ids": ["r4_postmortem/positive_fixture"],
                "partition": "train",
            },
            "applicability": {
                "all": [
                    {"feature": "current.terrain_band", "op": "in", "value": ["ROUGH", "VERY_ROUGH"]},
                    {"feature": "current.tile_type", "op": "in", "value": [int(Tile.GROUND), int(Tile.ROAD)]},
                    {"feature": "cargo.has_ore", "op": "eq", "value": False},
                    {"feature": "route.exists", "op": "eq", "value": True},
                    {"feature": "route.is_known_transport_route", "op": "eq", "value": True},
                    {"feature": "route.remaining_length_bucket", "op": "in", "value": ["medium", "long"]},
                    {"feature": "memory.similar_outcome_count", "op": "gte", "value": 3},
                ]
            },
            "procedure": [
                {"op": "ESTIMATE", "estimator": "future_route_uses", "store_as": "n_use"},
                {"op": "ESTIMATE", "estimator": "road_break_even_uses", "store_as": "n_break_even"},
                {
                    "op": "IF",
                    "condition": {"left": {"var": "n_use"}, "op": "gte", "right": {"var": "n_break_even"}},
                    "then": [{"op": "ACT", "action": "BUILD_ROAD"}],
                    "else": [{"op": "RETURN", "result": "not_applicable"}],
                },
            ],
            "budget": {
                "max_runtime_steps": 4,
                "max_environment_actions": 1,
                "max_nested_skill_depth": 0,
                "max_uses_per_episode": 1,
                "stop_after_success": True,
            },
            "objective": {
                "primary_metric": PRIMARY_METRIC,
                "direction": "maximize",
                "negative_context_metric": "false_trigger_rate",
            },
            "dependencies": [],
            "rationale": "Synthetic positive fixture for R4 postmortem diagnostics.",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the handcrafted positive Skill promotion fixture.")
    parser.add_argument("--out", default="outputs/r4_handcrafted_positive_fixture")
    args = parser.parse_args()
    manifest = run_handcrafted_positive_skill_fixture(args.out)
    print(json.dumps({"run_manifest": str(Path(args.out) / "run_manifest.json"), "decision": manifest["decision"]}))


if __name__ == "__main__":
    main()
