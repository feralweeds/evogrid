"""Skill proposal validation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable

from evogrid.llm.skill_prompts import (
    SKILL_PROPOSER_PROMPT_ID,
    SKILL_PROPOSER_PROMPT_VERSION,
    skill_proposer_prompt_hash,
)
from evogrid.skills.registry import SkillRegistry
from evogrid.skills.schemas import SkillRecord, SkillSpec
from evogrid.skills.schemas import PROCEDURE_OPS


HIDDEN_TRAJECTORY_KEYS = {
    "grid",
    "ore_positions",
    "audit",
    "static_diagnostics",
    "shortest_path_length",
    "minimum_cost_path_cost",
    "largest_component_fraction",
    "route_rough_tile_count",
    "off_route_rough_tile_count",
    "positive_road_opportunity_count",
    "evaluator",
}


ProposalBackend = Callable[[dict[str, Any]], str | dict[str, Any] | list[Any]]


@dataclass
class ProposalResult:
    accepted: list[SkillRecord] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)


class SkillProposer:
    def __init__(self, registry: SkillRegistry, backend: ProposalBackend | None = None):
        self.registry = registry
        self.backend = backend or _empty_backend

    def propose_from_trajectories(
        self,
        trajectories: list[dict[str, Any]],
        existing_records: list[SkillRecord] | None = None,
    ) -> ProposalResult:
        existing_hashes = {record.spec.spec_hash for record in existing_records or []}
        partition_errors = _train_partition_errors(trajectories)
        if partition_errors:
            return ProposalResult(
                rejected=[
                    {
                        "event_type": "proposal_rejected",
                        "reason": "non_train_source_trajectory",
                        "details": partition_errors,
                        "proposal": None,
                    }
                ]
            )
        payload = {
            "prompt_id": SKILL_PROPOSER_PROMPT_ID,
            "prompt_version": SKILL_PROPOSER_PROMPT_VERSION,
            "prompt_hash": skill_proposer_prompt_hash(),
            "allowed_status": "proposed",
            "source_partition": "train",
            "allowed_procedure_ops": sorted(PROCEDURE_OPS),
            "trajectories": [_observable_trajectory(item) for item in trajectories],
        }
        result = ProposalResult()
        try:
            proposals = _extract_json_objects(self.backend(payload))
        except Exception as exc:  # noqa: BLE001
            result.rejected.append({"event_type": "proposal_rejected", "reason": str(exc), "proposal": None})
            return result

        for proposal in proposals:
            try:
                proposal = dict(proposal)
                proposal.setdefault("schema_version", 1)
                proposal["status"] = "proposed"
                proposal.setdefault("source", {})
                proposal["source"] = {
                    **proposal["source"],
                    "proposer": "llm",
                    "source_episode_ids": _source_episode_ids(trajectories),
                    "base_prompt_hash": skill_proposer_prompt_hash(),
                    "prompt_template_id": SKILL_PROPOSER_PROMPT_ID,
                    "prompt_template_version": SKILL_PROPOSER_PROMPT_VERSION,
                }
                proposed = SkillSpec.from_dict(proposal)
                candidate_data = proposed.to_dict()
                candidate_data["status"] = "candidate"
                candidate_data.pop("spec_hash", None)
                candidate = SkillSpec.from_dict(candidate_data)
                if candidate.spec_hash in existing_hashes:
                    raise ValueError("duplicate proposal spec_hash")
                record = self.registry.register_candidate(candidate)
                existing_hashes.add(record.spec.spec_hash)
                result.accepted.append(record)
            except Exception as exc:  # noqa: BLE001
                result.rejected.append(
                    {
                        "event_type": "proposal_rejected",
                        "reason": str(exc),
                        "proposal": proposal,
                    }
                )
        return result


def _empty_backend(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return []


def _extract_json_objects(raw: str | dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        if isinstance(raw.get("skills"), list):
            return [dict(item) for item in raw["skills"]]
        return [raw]
    if isinstance(raw, list):
        return [dict(item) for item in raw]
    parsed = json.loads(raw)
    return _extract_json_objects(parsed)


def _observable_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    return _strip_hidden(trajectory)


def _strip_hidden(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_hidden(item)
            for key, item in value.items()
            if str(key) not in HIDDEN_TRAJECTORY_KEYS
        }
    if isinstance(value, list):
        return [_strip_hidden(item) for item in value]
    return value


def _source_episode_ids(trajectories: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("episode_id", f"train/episode/{idx}")) for idx, item in enumerate(trajectories)]


def _train_partition_errors(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors = []
    for index, trajectory in enumerate(trajectories):
        partition = str(trajectory.get("partition", "train"))
        episode_id = str(trajectory.get("episode_id", ""))
        if partition != "train" or episode_id.startswith(("verify/", "test/", "gate/")):
            errors.append({"index": index, "partition": partition, "episode_id": episode_id})
    return errors
