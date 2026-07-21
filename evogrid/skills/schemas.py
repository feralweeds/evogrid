"""Schema objects for Candidate/Verified Skills."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

from evogrid.constants import ACTION_NAMES
from evogrid.skills.predicates import ALLOWED_FEATURES, ALLOWED_OPS


ORDERED_NUMERIC_FEATURES = {
    "current.tile_type",
    "memory.similar_outcome_count",
    "memory.similar_mean_payoff",
    "local.adjacent_obstacle_count",
    "local.frontier_count",
    "episode_budget.steps_remaining",
}
ENUM_FEATURE_VALUES = {
    "current.terrain_band": {"SMOOTH", "NORMAL", "ROUGH", "VERY_ROUGH"},
    "route.remaining_length_bucket": {"short", "medium", "long", "unknown"},
    "memory.visit_count_bucket": {"low", "medium", "high"},
}
BOOLEAN_FEATURES = {"cargo.has_ore", "route.exists", "route.is_known_transport_route"}
ORDERED_OPS = {"lt", "lte", "gt", "gte"}
SKILL_STATUSES = {
    "proposed",
    "candidate",
    "verifying",
    "verified",
    "revision_required",
    "rejected",
    "deprecated",
}
REPORT_DECISIONS = {"verified", "revision_required", "rejected", "verification_invalid"}
SOURCE_PROPOSERS = {"llm", "rule", "human", "fixture", "handcrafted"}
PROCEDURE_OPS = {
    "ESTIMATE",
    "SELECT_TARGET",
    "PLAN_ROUTE",
    "IF",
    "ACT",
    "FOLLOW_ROUTE",
    "CALL_SKILL",
    "RETURN",
}
TEXT_KEYS_FORBIDDEN_IN_PROCEDURE = {"description", "rationale", "instruction", "prompt", "natural_language"}
LEGAL_TRANSITIONS = {
    "proposed": {"candidate"},
    "candidate": {"verifying"},
    "verifying": {"verified", "revision_required", "rejected"},
    "revision_required": {"candidate"},
    "verified": {"deprecated"},
    "deprecated": {"candidate"},
    "rejected": set(),
}


@dataclass(frozen=True)
class SkillSpec:
    schema_version: int
    skill_id: str
    version: str
    status: str
    name: str
    description: str
    problem_addressed: str
    source: dict[str, Any]
    applicability: dict[str, Any]
    procedure: list[dict[str, Any]]
    budget: dict[str, Any]
    objective: dict[str, Any]
    dependencies: list[str] = field(default_factory=list)
    rationale: str = ""
    spec_hash: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillSpec":
        _require_keys(
            data,
            {
                "schema_version",
                "skill_id",
                "version",
                "status",
                "name",
                "description",
                "problem_addressed",
                "source",
                "applicability",
                "procedure",
                "budget",
                "objective",
            },
            "SkillSpec",
        )
        spec = cls(
            schema_version=int(data["schema_version"]),
            skill_id=_identifier(data["skill_id"], "skill_id"),
            version=_version(data["version"]),
            status=_enum(data["status"], SKILL_STATUSES, "status"),
            name=str(data["name"]),
            description=str(data["description"]),
            problem_addressed=str(data["problem_addressed"]),
            source=_dict(data["source"], "source"),
            applicability=_dict(data["applicability"], "applicability"),
            procedure=_procedure(data["procedure"]),
            budget=_dict(data["budget"], "budget"),
            objective=_dict(data["objective"], "objective"),
            dependencies=_dependencies(data.get("dependencies", [])),
            rationale=str(data.get("rationale", "")),
            spec_hash=str(data.get("spec_hash", "")),
        )
        spec.validate()
        expected_hash = compute_spec_hash(spec)
        if spec.spec_hash and spec.spec_hash != expected_hash:
            raise ValueError("spec_hash: does not match canonical SkillSpec content")
        return spec if spec.spec_hash else spec.with_hash()

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version: expected 1")
        proposer = self.source.get("proposer")
        if proposer is not None and proposer not in SOURCE_PROPOSERS:
            raise ValueError("source.proposer: unsupported value")
        if self.source.get("proposer") == "llm" and self.status == "verified":
            raise ValueError("status: LLM/proposer output cannot directly create verified Skill")
        _validate_budget(self.budget)
        _validate_objective(self.objective)
        _validate_applicability(self.applicability)
        _procedure(self.procedure)

    def with_hash(self) -> "SkillSpec":
        return SkillSpec(
            schema_version=self.schema_version,
            skill_id=self.skill_id,
            version=self.version,
            status=self.status,
            name=self.name,
            description=self.description,
            problem_addressed=self.problem_addressed,
            source=dict(self.source),
            applicability=dict(self.applicability),
            procedure=[dict(item) for item in self.procedure],
            budget=dict(self.budget),
            objective=dict(self.objective),
            dependencies=list(self.dependencies),
            rationale=self.rationale,
            spec_hash=compute_spec_hash(self),
        )

    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "skill_id": self.skill_id,
            "version": self.version,
            "status": self.status,
            "name": self.name,
            "description": self.description,
            "problem_addressed": self.problem_addressed,
            "source": _json_ready(self.source),
            "applicability": _json_ready(self.applicability),
            "procedure": _json_ready(self.procedure),
            "budget": _json_ready(self.budget),
            "objective": _json_ready(self.objective),
            "dependencies": _json_ready(self.dependencies),
            "rationale": self.rationale,
        }
        if include_hash:
            result["spec_hash"] = self.spec_hash
        return result


@dataclass(frozen=True)
class SkillRecord:
    spec: SkillSpec
    storage_status: str
    created_at: str = ""
    updated_at: str = ""
    verification_reports: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillRecord":
        _require_keys(data, {"spec", "storage_status"}, "SkillRecord")
        spec = SkillSpec.from_dict(_dict(data["spec"], "spec"))
        storage_status = _enum(data["storage_status"], SKILL_STATUSES, "storage_status")
        if storage_status != spec.status:
            raise ValueError("storage_status: must match spec.status")
        return cls(
            spec=spec,
            storage_status=storage_status,
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            verification_reports=[str(item) for item in data.get("verification_reports", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "storage_status": self.storage_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "verification_reports": list(self.verification_reports),
        }


@dataclass(frozen=True)
class VerificationReport:
    schema_version: int
    verification_id: str
    skill_id: str
    skill_version: str
    spec_hash: str
    protocol_id: str
    candidate_source_partitions: list[str]
    verification_partition: str
    paired_seeds: list[int]
    environment_strata: list[str]
    baseline: str
    sample_size: int
    metrics: dict[str, Any]
    gates: list[dict[str, Any]]
    decision: str
    failure_reasons: list[str]
    created_at: str = ""
    report_hash: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VerificationReport":
        _require_keys(
            data,
            {
                "schema_version",
                "verification_id",
                "skill_id",
                "skill_version",
                "spec_hash",
                "protocol_id",
                "candidate_source_partitions",
                "verification_partition",
                "paired_seeds",
                "environment_strata",
                "baseline",
                "sample_size",
                "metrics",
                "gates",
                "decision",
                "failure_reasons",
            },
            "VerificationReport",
        )
        report = cls(
            schema_version=int(data["schema_version"]),
            verification_id=_identifier(data["verification_id"], "verification_id"),
            skill_id=_identifier(data["skill_id"], "skill_id"),
            skill_version=_version(data["skill_version"]),
            spec_hash=str(data["spec_hash"]),
            protocol_id=str(data["protocol_id"]),
            candidate_source_partitions=[str(item) for item in data["candidate_source_partitions"]],
            verification_partition=str(data["verification_partition"]),
            paired_seeds=[int(item) for item in data["paired_seeds"]],
            environment_strata=[str(item) for item in data["environment_strata"]],
            baseline=str(data["baseline"]),
            sample_size=int(data["sample_size"]),
            metrics=_dict(data["metrics"], "metrics"),
            gates=[_dict(item, "gates[]") for item in data["gates"]],
            decision=_enum(data["decision"], REPORT_DECISIONS, "decision"),
            failure_reasons=[str(item) for item in data["failure_reasons"]],
            created_at=str(data.get("created_at", "")),
            report_hash=str(data.get("report_hash", "")),
        )
        report.validate()
        expected_hash = compute_report_hash(report)
        if report.report_hash and report.report_hash != expected_hash:
            raise ValueError("report_hash: does not match canonical VerificationReport content")
        return report if report.report_hash else report.with_hash()

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("schema_version: expected 1")
        if self.sample_size < 0:
            raise ValueError("sample_size: expected non-negative")
        if self.sample_size != len(self.paired_seeds):
            raise ValueError("sample_size: must match paired_seeds length")

    def with_hash(self) -> "VerificationReport":
        return VerificationReport(
            schema_version=self.schema_version,
            verification_id=self.verification_id,
            skill_id=self.skill_id,
            skill_version=self.skill_version,
            spec_hash=self.spec_hash,
            protocol_id=self.protocol_id,
            candidate_source_partitions=list(self.candidate_source_partitions),
            verification_partition=self.verification_partition,
            paired_seeds=list(self.paired_seeds),
            environment_strata=list(self.environment_strata),
            baseline=self.baseline,
            sample_size=self.sample_size,
            metrics=dict(self.metrics),
            gates=[dict(item) for item in self.gates],
            decision=self.decision,
            failure_reasons=list(self.failure_reasons),
            created_at=self.created_at,
            report_hash=compute_report_hash(self),
        )

    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "verification_id": self.verification_id,
            "skill_id": self.skill_id,
            "skill_version": self.skill_version,
            "spec_hash": self.spec_hash,
            "protocol_id": self.protocol_id,
            "candidate_source_partitions": list(self.candidate_source_partitions),
            "verification_partition": self.verification_partition,
            "paired_seeds": list(self.paired_seeds),
            "environment_strata": list(self.environment_strata),
            "baseline": self.baseline,
            "sample_size": self.sample_size,
            "metrics": _json_ready(self.metrics),
            "gates": _json_ready(self.gates),
            "decision": self.decision,
            "failure_reasons": list(self.failure_reasons),
            "created_at": self.created_at,
        }
        if include_hash:
            result["report_hash"] = self.report_hash
        return result


def validate_status_transition(current: str, next_status: str) -> None:
    current = _enum(current, SKILL_STATUSES, "current")
    next_status = _enum(next_status, SKILL_STATUSES, "next_status")
    if next_status not in LEGAL_TRANSITIONS[current]:
        raise ValueError(f"status transition not allowed: {current} -> {next_status}")


def compute_spec_hash(spec: SkillSpec | dict[str, Any]) -> str:
    data = spec.to_dict(include_hash=False) if isinstance(spec, SkillSpec) else dict(spec)
    data.pop("spec_hash", None)
    data.pop("status", None)
    return _sha256(data)


def compute_report_hash(report: VerificationReport | dict[str, Any]) -> str:
    data = report.to_dict(include_hash=False) if isinstance(report, VerificationReport) else dict(report)
    data.pop("report_hash", None)
    return _sha256(data)


def canonical_json(value: Any) -> str:
    return json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_keys(data: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(data))
    if missing:
        raise ValueError(f"{label}: missing required keys {missing}")


def _identifier(value: Any, path: str) -> str:
    text = str(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", text):
        raise ValueError(f"{path}: invalid identifier")
    return text


def _version(value: Any) -> str:
    text = str(value)
    if not re.fullmatch(r"\d+\.\d+\.\d+", text):
        raise ValueError("version: expected semantic version like 1.0.0")
    return text


def _enum(value: Any, allowed: set[str], path: str) -> str:
    text = str(value)
    if text not in allowed:
        raise ValueError(f"{path}: expected one of {sorted(allowed)}")
    return text


def _dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return dict(value)


def _dependencies(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("dependencies: expected list")
    return [_identifier(item, "dependencies[]") for item in value]


def _procedure(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("procedure: expected non-empty list")
    nodes = [_dict(item, "procedure[]") for item in value]
    for node in nodes:
        _validate_procedure_node(node)
    return nodes


def _validate_procedure_node(node: dict[str, Any]) -> None:
    for key in TEXT_KEYS_FORBIDDEN_IN_PROCEDURE:
        if key in node:
            raise ValueError(f"procedure: natural-language key {key!r} is not executable")
    op = _enum(node.get("op"), PROCEDURE_OPS, "procedure.op")
    if op == "IF":
        if "condition" not in node:
            raise ValueError("procedure.IF.condition: required")
        for branch_name in ("then", "else"):
            if branch_name in node:
                _procedure(node[branch_name])
    elif op == "ACT":
        if "action" not in node:
            raise ValueError("procedure.ACT.action: required")
        if str(node["action"]) not in ACTION_NAMES:
            raise ValueError("procedure.ACT.action: unknown primitive action")
    elif op == "CALL_SKILL":
        if "skill_id" not in node:
            raise ValueError("procedure.CALL_SKILL.skill_id: required")


def _validate_budget(budget: dict[str, Any]) -> None:
    for key in ("max_runtime_steps", "max_environment_actions", "max_nested_skill_depth", "max_uses_per_episode"):
        if key in budget and int(budget[key]) < 0:
            raise ValueError(f"budget.{key}: expected non-negative")
    if "stop_after_success" in budget and not isinstance(budget["stop_after_success"], bool):
        raise ValueError("budget.stop_after_success: expected boolean")


def _validate_objective(objective: dict[str, Any]) -> None:
    if "primary_metric" not in objective:
        raise ValueError("objective.primary_metric: required")
    if objective.get("direction") not in {"maximize", "minimize"}:
        raise ValueError("objective.direction: expected maximize or minimize")


def _validate_applicability(applicability: dict[str, Any]) -> None:
    if not any(key in applicability for key in ("all", "any", "not")):
        raise ValueError("applicability: expected at least one logical node")
    _validate_applicability_node(applicability)


def _validate_applicability_node(node: dict[str, Any], depth: int = 0, max_depth: int = 8) -> None:
    if depth > max_depth:
        raise ValueError("applicability: maximum nesting depth exceeded")
    if not isinstance(node, dict):
        raise ValueError("applicability: expected object")
    if "all" in node:
        _validate_applicability_children(node["all"], depth, max_depth, "all")
        return
    if "any" in node:
        _validate_applicability_children(node["any"], depth, max_depth, "any")
        return
    if "not" in node:
        _validate_applicability_node(_dict(node["not"], "applicability.not"), depth + 1, max_depth)
        return
    feature = str(node.get("feature"))
    if feature not in ALLOWED_FEATURES:
        raise ValueError(f"applicability.feature: not allowed: {feature}")
    op = str(node.get("op"))
    if op not in ALLOWED_OPS:
        raise ValueError(f"applicability.op: not allowed: {op}")
    if "value" not in node:
        raise ValueError("applicability.value: required")
    _validate_applicability_value(feature, op, node["value"])


def _validate_applicability_value(feature: str, op: str, value: Any) -> None:
    if feature in BOOLEAN_FEATURES:
        if op not in {"eq", "ne"} or not isinstance(value, bool):
            raise ValueError(f"applicability.value: {feature} requires boolean eq/ne")
        return
    if feature in ENUM_FEATURE_VALUES:
        if op not in {"eq", "ne", "in", "not_in"}:
            raise ValueError(f"applicability.op: {feature} supports only enum equality ops")
        allowed = ENUM_FEATURE_VALUES[feature]
        values = value if op in {"in", "not_in"} else [value]
        if not isinstance(values, list) or any(str(item) not in allowed for item in values):
            raise ValueError(f"applicability.value: {feature} expected one of {sorted(allowed)}")
        return
    if feature in ORDERED_NUMERIC_FEATURES:
        values = value if op in {"in", "not_in"} else [value]
        if not isinstance(values, list) or any(
            isinstance(item, bool) or not isinstance(item, (int, float)) for item in values
        ):
            raise ValueError(f"applicability.value: {feature} requires numeric value")
        return
    if op in ORDERED_OPS:
        raise ValueError(f"applicability.op: ordered comparison not supported for {feature}")


def _validate_applicability_children(value: Any, depth: int, max_depth: int, path: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"applicability.{path}: expected list")
    for child in value:
        _validate_applicability_node(_dict(child, f"applicability.{path}[]"), depth + 1, max_depth)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return [_json_ready(item) for item in sorted(value)]
    return value
