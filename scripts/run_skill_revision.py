from __future__ import annotations

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import yaml

from evogrid.llm.deepseek_client import DeepSeekClient
from evogrid.llm.skill_prompts import (
    SKILL_REVISION_PROMPT,
    SKILL_REVISION_PROMPT_ID,
    SKILL_REVISION_PROMPT_VERSION,
    skill_revision_prompt_hash,
)
from evogrid.constants import Tile
from evogrid.skills.predicates import ALLOWED_FEATURES, ALLOWED_OPS
from evogrid.skills.proposer import _extract_json_objects
from evogrid.skills.registry import SkillRegistry
from evogrid.skills.schemas import SkillSpec
from scripts.calibrate_fractal_maps import _runtime_metadata, _stable_hash


def run_skill_revision(config_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    revision_config = config.get("revision", {})
    started_at = datetime.now(timezone.utc).isoformat()

    candidate_path = Path(str(revision_config["candidate_path"]))
    report_path = Path(str(revision_config["verification_report_path"]))
    candidate_record = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    previous_spec = SkillSpec.from_dict(candidate_record.get("spec", candidate_record))
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    feedback = _verification_feedback_summary(report)
    new_version = str(revision_config.get("new_version", _bump_patch_version(previous_spec.version)))

    registry = SkillRegistry(out_dir / "skills")
    backend_name = str(revision_config.get("backend", "fixture_revision_valid"))
    backend_metadata: dict[str, Any] = {"schema_version": 1, "backend": backend_name}
    backend = _build_backend(backend_name, revision_config, backend_metadata, previous_spec, feedback, new_version)

    accepted = []
    rejected = []
    try:
        proposals = _extract_json_objects(backend(_revision_payload(previous_spec, feedback, new_version)))
    except Exception as exc:  # noqa: BLE001
        proposals = []
        rejected.append({"event_type": "proposal_rejected", "reason": str(exc), "proposal": None})

    for proposal in proposals:
        try:
            proposal = dict(proposal)
            proposal.setdefault("schema_version", 1)
            proposal["status"] = "proposed"
            proposal["skill_id"] = previous_spec.skill_id
            proposal["version"] = new_version
            proposal["source"] = {
                **dict(proposal.get("source", {})),
                "proposer": "llm",
                "source_episode_ids": _train_source_episode_ids(previous_spec),
                "base_prompt_hash": skill_revision_prompt_hash(),
                "prompt_template_id": SKILL_REVISION_PROMPT_ID,
                "prompt_template_version": SKILL_REVISION_PROMPT_VERSION,
                "revision_of_spec_hash": previous_spec.spec_hash,
                "revision_feedback_report_hash": str(report.get("report_hash", "")),
            }
            proposed = SkillSpec.from_dict(proposal)
            if proposed.skill_id != previous_spec.skill_id:
                raise ValueError("revision must keep the same skill_id")
            if proposed.version != new_version:
                raise ValueError("revision must use requested new_version")
            if _executable_signature(proposed) == _executable_signature(previous_spec):
                raise ValueError("revision must change executable Skill contract, not only metadata")
            candidate_data = proposed.to_dict()
            candidate_data["status"] = "candidate"
            candidate_data.pop("spec_hash", None)
            record = registry.register_candidate(SkillSpec.from_dict(candidate_data))
            accepted.append(record)
        except Exception as exc:  # noqa: BLE001
            rejected.append({"event_type": "proposal_rejected", "reason": str(exc), "proposal": proposal})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    proposal_manifest = {
        "schema_version": 1,
        "accepted": [
            {
                "skill_id": record.spec.skill_id,
                "version": record.spec.version,
                "status": record.spec.status,
                "spec_hash": record.spec.spec_hash,
                "path": f"skills/candidates/{record.spec.skill_id}/{record.spec.version}.json",
            }
            for record in accepted
        ],
        "rejected": rejected,
        "backend": backend_name,
        "backend_metadata": backend_metadata,
        "source_partition": "train",
        "revision": {
            "schema_version": 1,
            "previous_skill_id": previous_spec.skill_id,
            "previous_version": previous_spec.version,
            "previous_spec_hash": previous_spec.spec_hash,
            "new_version": new_version,
            "feedback_report_hash": str(report.get("report_hash", "")),
            "feedback_is_aggregate_only": True,
            "reuse_previous_verify_seeds_allowed": False,
        },
        "verification_started": False,
        "verified_written": False,
    }
    (out_dir / "proposal_manifest.json").write_text(
        json.dumps(proposal_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "run_id": out_dir.name,
        "experiment_type": "skill_proposal",
        "mode": str(revision_config.get("mode", "revision_pilot")),
        "mock_smoke": True,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "resolved_config_hash": _stable_hash(config),
        "accepted_candidate_count": len(accepted),
        "rejected_proposal_count": len(rejected),
        "revision_of_spec_hash": previous_spec.spec_hash,
        "feedback_report_hash": str(report.get("report_hash", "")),
        "runtime": _runtime_metadata(),
        "outputs": {
            "config_resolved": "config_resolved.yaml",
            "proposal_manifest": "proposal_manifest.json",
            "registry_events": "skills/registry_events.jsonl",
        },
        "formal_acceptance": {
            "passed": False,
            "conclusion_level": "E0",
            "gate_report": "proposal_manifest.json",
        },
        "completion_status": "completed",
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Revise a failed Candidate Skill using aggregate verification feedback.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = run_skill_revision(args.config, args.out)
    print(
        json.dumps(
            {
                "run_manifest": str(Path(args.out) / "run_manifest.json"),
                "accepted_candidate_count": manifest["accepted_candidate_count"],
                "rejected_proposal_count": manifest["rejected_proposal_count"],
            }
        )
    )


def _build_backend(
    name: str,
    config: dict[str, Any],
    metadata: dict[str, Any],
    previous_spec: SkillSpec,
    feedback: dict[str, Any],
    new_version: str,
):
    if name == "deepseek":
        return _deepseek_revision_backend(config.get("deepseek", {}), metadata)
    if name == "fixture_revision_invalid":
        return lambda payload: {"skills": [{"skill_id": previous_spec.skill_id, "version": new_version}]}
    if name == "fixture_revision_noop":
        return lambda payload: {"skills": [_fixture_revision_noop(previous_spec, new_version)]}
    return lambda payload: {"skills": [_fixture_revision(previous_spec, feedback, new_version)]}


def _deepseek_revision_backend(config: dict[str, Any], metadata: dict[str, Any], client: Any | None = None):
    client = client or DeepSeekClient(
        base_url=config.get("base_url"),
        model=config.get("model"),
        timeout=config.get("timeout"),
        max_tokens=config.get("max_tokens"),
        json_mode=bool(config.get("json_mode", True)),
    )
    temperature = float(config.get("temperature", 0.2))
    metadata.update(
        {
            "provider": "deepseek",
            "model": client.model,
            "base_url": client.base_url,
            "temperature": temperature,
            "prompt_hash": skill_revision_prompt_hash(),
            "response_received": False,
            "api_key_env": "DEEPSEEK_API_KEY",
        }
    )

    def backend(payload: dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": SKILL_REVISION_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]
        completion = client.chat_completion(messages, temperature=temperature, json_mode=True)
        content = completion["content"]
        metadata.update(
            {
                "response_received": True,
                "finish_reason": completion.get("finish_reason"),
                "response_model": completion.get("model"),
                "usage": completion.get("usage", {}),
                "content_chars": len(content),
                "content_preview": content[:500],
            }
        )
        return content

    return backend


def _revision_payload(previous_spec: SkillSpec, feedback: dict[str, Any], new_version: str) -> dict[str, Any]:
    return {
        "task": "revise_candidate_skill",
        "constraints": {
            "allowed_status": "proposed",
            "source_partition": "train",
            "new_version": new_version,
            "same_skill_id": previous_spec.skill_id,
            "allowed_applicability_features": sorted(ALLOWED_FEATURES),
            "allowed_applicability_ops": sorted(ALLOWED_OPS),
            "forbidden_applicability_ops": ["==", "!=", ">=", "<=", "not in", "not-in"],
            "forbidden_applicability_features": [
                "current.road_exists",
                "road.already_built",
                "map.route_reused",
                "evaluator.*",
                "audit.*",
            ],
            "observable_tile_ids": {tile.name: int(tile) for tile in Tile},
            "build_road_tile_type_guard": {
                "feature": "current.tile_type",
                "op": "in",
                "value": [int(Tile.GROUND), int(Tile.ROUGH)],
                "reason": "BUILD_ROAD is legal only on buildable non-road current tiles.",
            },
            "bucket_feature_contract": {
                "route.remaining_length_bucket": {
                    "type": "string_enum",
                    "values": ["short", "medium", "long", "unknown"],
                    "allowed_ops": ["eq", "ne", "in", "not_in"],
                    "valid_examples": [
                        {"feature": "route.remaining_length_bucket", "op": "in", "value": ["medium", "long"]},
                        {"feature": "route.remaining_length_bucket", "op": "ne", "value": "short"},
                    ],
                    "invalid_examples": [
                        {"feature": "route.remaining_length_bucket", "op": "gte", "value": 2}
                    ],
                },
                "memory.visit_count_bucket": {
                    "type": "string_enum",
                    "values": ["low", "medium", "high"],
                    "allowed_ops": ["eq", "ne", "in", "not_in"],
                    "valid_examples": [
                        {"feature": "memory.visit_count_bucket", "op": "in", "value": ["medium", "high"]}
                    ],
                },
            },
            "recommended_executable_revision_examples": [
                {
                    "reason": "Reduce overbuilding after BUILD_ROAD is already tile-guarded.",
                    "applicability_leaf": {
                        "feature": "route.remaining_length_bucket",
                        "op": "in",
                        "value": ["medium", "long"],
                    },
                },
                {
                    "reason": "Require observed repeated visits without depending on cross-episode payoff memory.",
                    "applicability_leaf": {
                        "feature": "memory.visit_count_bucket",
                        "op": "in",
                        "value": ["medium", "high"],
                    },
                },
            ],
            "allowed_procedure_ops": [
                "ACT",
                "CALL_SKILL",
                "ESTIMATE",
                "FOLLOW_ROUTE",
                "IF",
                "PLAN_ROUTE",
                "RETURN",
                "SELECT_TARGET",
            ],
            "do_not_use_verify_seeds_for_next_verification": True,
        },
        "previous_skill": previous_spec.to_dict(),
        "aggregate_verification_feedback": feedback,
        "revision_guidance": [
            "Use the aggregate activation_rate to decide whether to narrow or relax applicability.",
            "If activation_rate is zero, remove or relax applicability gates that require unavailable prior memory.",
            "Narrow applicability so the Skill does not trigger on every rough tile when false triggers are high.",
            "If runtime failures occur with BUILD_ROAD, add the build_road_tile_type_guard exactly.",
            "If aggregate enabled_mean_num_build_road is high while effect is negative, reduce overbuilding with current.tile_type plus a conservative visit or route-length gate.",
            "When adding a route-length gate, use the exact enum shape route.remaining_length_bucket in ['medium', 'long']; do not use gte or numeric values.",
            "Prefer route evidence for repeated transport; use positive memory thresholds only when the verifier memory can contain that evidence before activation.",
            "Avoid repeated BUILD_ROAD after a road has already been built or when route reuse is not established.",
            "Keep the output as a complete SkillSpec JSON object inside skills[].",
        ],
    }


def _verification_feedback_summary(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    gates = []
    for gate in report.get("gates", []):
        if not isinstance(gate, dict):
            continue
        gates.append(
            {
                "gate": gate.get("gate", gate.get("gate_id")),
                "passed": gate.get("passed"),
                "mean_delta": gate.get("mean_delta"),
                "positive_rate": gate.get("positive_rate"),
                "runtime_failure_rate": gate.get("runtime_failure_rate"),
                "false_trigger_rate": gate.get("false_trigger_rate"),
                "activation_rate": gate.get("activation_rate"),
            }
        )
    return {
        "schema_version": 1,
        "protocol_id": report.get("protocol_id"),
        "decision": report.get("decision"),
        "failure_reasons": list(report.get("failure_reasons", [])),
        "gates": gates,
        "metrics": {
            "primary_metric": metrics.get("primary_metric"),
            "paired_delta_mean": metrics.get("paired_delta_mean"),
            "paired_delta_bootstrap_ci": metrics.get("paired_delta_bootstrap_ci"),
            "success_rate": metrics.get("success_rate"),
            "false_trigger_rate": metrics.get("false_trigger_rate"),
            "activation_rate": metrics.get("activation_rate"),
            "runtime_failure_rate": metrics.get("runtime_failure_rate"),
            "transfer": metrics.get("transfer"),
            "aggregate_episode_metrics": _aggregate_episode_metrics(metrics),
        },
        "omitted_fields": ["paired_seeds", "disabled", "enabled", "paired_deltas", "paired_deltas_by_stratum"],
    }


def _aggregate_episode_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    enabled = metrics.get("enabled", [])
    disabled = metrics.get("disabled", [])
    return {
        "enabled_mean_num_build_road": _mean_metric(enabled, "num_build_road"),
        "disabled_mean_num_build_road": _mean_metric(disabled, "num_build_road"),
        "enabled_mean_road_usage_rate": _mean_metric(enabled, "road_usage_rate"),
        "disabled_mean_road_usage_rate": _mean_metric(disabled, "road_usage_rate"),
        "enabled_mean_episode_reward": _mean_metric(enabled, "episode_reward"),
        "disabled_mean_episode_reward": _mean_metric(disabled, "episode_reward"),
        "enabled_mean_ore_delivered": _mean_metric(enabled, "ore_delivered"),
        "disabled_mean_ore_delivered": _mean_metric(disabled, "ore_delivered"),
        "activated_mean_num_build_road": _mean_metric(_activated_rows(enabled), "num_build_road"),
        "activated_mean_road_net_payoff": _mean_metric(_activated_rows(enabled), "road_net_payoff"),
        "activated_mean_road_usage_rate": _mean_metric(_activated_rows(enabled), "road_usage_rate"),
    }


def _activated_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict) and float(row.get("activation_rate", 0.0) or 0.0) > 0.0
    ]


def _mean_metric(rows: Any, key: str) -> float | None:
    if not isinstance(rows, list):
        return None
    values = []
    for row in rows:
        if isinstance(row, dict) and row.get(key) is not None:
            values.append(float(row[key]))
    if not values:
        return None
    return sum(values) / len(values)


def _executable_signature(spec: SkillSpec) -> dict[str, Any]:
    return {
        "applicability": spec.applicability,
        "procedure": spec.procedure,
        "budget": spec.budget,
        "objective": spec.objective,
        "dependencies": spec.dependencies,
    }


def _fixture_revision(previous_spec: SkillSpec, feedback: dict[str, Any], new_version: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "skill_id": previous_spec.skill_id,
        "version": new_version,
        "status": "proposed",
        "name": f"{previous_spec.name} revised",
        "description": "Revision narrows road building to known reused transport routes.",
        "problem_addressed": previous_spec.problem_addressed,
        "source": {"proposer": "llm"},
        "applicability": {
            "all": [
                {"feature": "current.terrain_band", "op": "in", "value": ["ROUGH", "VERY_ROUGH"]},
                {"feature": "current.tile_type", "op": "eq", "value": 0},
                {"feature": "route.is_known_transport_route", "op": "eq", "value": True},
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
        "budget": {"max_runtime_steps": 4, "max_environment_actions": 1, "max_nested_skill_depth": 0},
        "objective": {
            "primary_metric": "road_net_payoff",
            "direction": "maximize",
            "negative_context_metric": "false_trigger_rate",
        },
        "dependencies": [],
        "rationale": f"Aggregate feedback failed {feedback.get('failure_reasons', [])}; narrow activation.",
    }


def _fixture_revision_noop(previous_spec: SkillSpec, new_version: str) -> dict[str, Any]:
    data = previous_spec.to_dict(include_hash=False)
    data["version"] = new_version
    data["status"] = "proposed"
    data["description"] = f"{previous_spec.description} Metadata-only no-op revision."
    data["rationale"] = "No executable change."
    return data


def _train_source_episode_ids(spec: SkillSpec) -> list[str]:
    source_ids = [str(item) for item in spec.source.get("source_episode_ids", [])]
    train_ids = [item for item in source_ids if item.startswith("train/")]
    return train_ids or ["train/revision/source"]


def _bump_patch_version(version: str) -> str:
    parts = [int(part) for part in version.split(".")]
    if len(parts) != 3:
        raise ValueError("version must use semantic version")
    parts[2] += 1
    return ".".join(str(part) for part in parts)


if __name__ == "__main__":
    main()
