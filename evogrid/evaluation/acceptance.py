"""Audit saved experiment outputs against the technical-spec guardrails."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from evogrid.skills.schemas import SkillRecord


REQUIRED_SEED_PARTITIONS = {"train", "gate", "verify", "test", "bootstrap"}
FORMAL_GATE_PROTOCOLS = {
    "map_calibration": {
        "protocol_id": "map_calibration_A0_A5_v1",
        "gate_key": "gate_id",
        "required_gates": {"A0", "A1", "A2", "A3", "A4", "A5"},
        "verified_decision": None,
    },
    "skill_verification": {
        "protocol_id": "skill_verification_v1",
        "gate_key": "gate",
        "required_gates": {
            "G0_data_integrity",
            "G1_effect",
            "G2_reliability",
            "G3_negative_safety",
            "G4_transfer",
            "G5_non_redundancy",
        },
        "verified_decision": "verified",
    },
    "continuous_terrain": {
        "protocol_id": "continuous_terrain_B0_B3_v1",
        "gate_key": "gate_id",
        "required_gates": {"B0", "B1", "B2", "B3"},
        "verified_decision": None,
    },
}


@dataclass(frozen=True)
class AcceptanceIssue:
    severity: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "path": self.path, "message": self.message}


@dataclass(frozen=True)
class AcceptanceReport:
    run_dir: str
    run_id: str
    experiment_type: str
    completion_status: str
    conclusion_level: str
    passed: bool
    issues: list[AcceptanceIssue] = field(default_factory=list)
    formal_acceptance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_dir": self.run_dir,
            "run_id": self.run_id,
            "experiment_type": self.experiment_type,
            "completion_status": self.completion_status,
            "conclusion_level": self.conclusion_level,
            "passed": self.passed,
            "formal_acceptance": self.formal_acceptance,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def audit_run_directory(run_dir: str | Path) -> AcceptanceReport:
    run_dir = Path(run_dir)
    issues: list[AcceptanceIssue] = []
    manifest_path = run_dir / "run_manifest.json"
    manifest: dict[str, Any] = {}
    if not manifest_path.exists():
        issues.append(_issue("error", manifest_path, "missing run_manifest.json", run_dir))
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            issues.append(_issue("error", manifest_path, f"invalid JSON: {exc}", run_dir))
            manifest = {}

    _audit_manifest(manifest, manifest_path, run_dir, issues)
    _audit_structured_outputs(run_dir, issues)
    _audit_experiment_guardrails(manifest, run_dir, issues)
    formal_acceptance = _audit_formal_acceptance(manifest, run_dir, issues)
    conclusion_level = infer_conclusion_level(manifest, formal_acceptance.get("validated"))
    passed = not any(issue.severity == "error" for issue in issues)
    return AcceptanceReport(
        run_dir=str(run_dir),
        run_id=str(manifest.get("run_id", run_dir.name)),
        experiment_type=str(manifest.get("experiment_type", "")),
        completion_status=str(manifest.get("completion_status", "")),
        conclusion_level=conclusion_level,
        passed=passed,
        issues=issues,
        formal_acceptance=formal_acceptance,
    )


def infer_conclusion_level(manifest: dict[str, Any], formal_validated: bool | None = None) -> str:
    mode = str(manifest.get("mode", ""))
    if bool(manifest.get("mock_smoke")) or "smoke" in mode or manifest.get("experiment_type", "").endswith("_smoke"):
        return "E0"
    formal = manifest.get("formal_acceptance", {})
    if not isinstance(formal, dict) or formal.get("passed") is not True:
        return "E0"
    if formal_validated is False:
        return "E0"
    claimed = str(formal.get("conclusion_level", "E0"))
    return claimed if claimed in {"E0", "E1", "E2", "E3", "E4", "E5"} else "E0"


def _audit_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    run_dir: Path,
    issues: list[AcceptanceIssue],
) -> None:
    if not manifest:
        return
    for key in ("schema_version", "run_id", "experiment_type", "completion_status"):
        if key not in manifest:
            issues.append(_issue("error", manifest_path, f"manifest missing {key}", run_dir))
    if manifest.get("completion_status") != "completed":
        issues.append(_issue("error", manifest_path, "run is not completed", run_dir))
    if "seed_partitions" in manifest:
        partitions = set(manifest.get("seed_partitions", {}).get("partitions", {}))
        missing = sorted(REQUIRED_SEED_PARTITIONS - partitions)
        if missing:
            issues.append(_issue("error", manifest_path, f"seed partitions missing {missing}", run_dir))
    conclusion_strength = str(manifest.get("conclusion_strength", ""))
    if bool(manifest.get("mock_smoke")) and conclusion_strength not in {"", "smoke_only_no_scientific_claim"}:
        issues.append(_issue("error", manifest_path, "mock smoke cannot make scientific claims", run_dir))
    if not bool(manifest.get("mock_smoke")) and manifest.get("experiment_type") in {
        "map_calibration",
        "continuous_terrain",
        "skill_verification",
        "skill_evolution",
        "curriculum_ablation",
    }:
        formal = manifest.get("formal_acceptance", {})
        if not isinstance(formal, dict) or formal.get("passed") is not True:
            issues.append(
                _issue(
                    "warning",
                    manifest_path,
                    "formal scientific gates not audited; conclusion level capped at E0",
                    run_dir,
                )
            )
    readiness = manifest.get("formal_readiness")
    formal = manifest.get("formal_acceptance", {})
    if isinstance(readiness, dict) and isinstance(formal, dict) and formal.get("passed") is True:
        if readiness.get("passed") is not True:
            issues.append(_issue("error", manifest_path, "formal acceptance failed readiness checks", run_dir))


def _audit_structured_outputs(run_dir: Path, issues: list[AcceptanceIssue]) -> None:
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            _audit_json(path, run_dir, issues)
        elif path.suffix == ".jsonl":
            _audit_jsonl(path, run_dir, issues)
        elif path.suffix == ".csv":
            _audit_csv(path, run_dir, issues)


def _audit_json(path: Path, run_dir: Path, issues: list[AcceptanceIssue]) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        issues.append(_issue("error", path, f"invalid JSON: {exc}", run_dir))
        return
    if not _has_schema(data):
        issues.append(_issue("error", path, "JSON output missing schema_version", run_dir))


def _audit_jsonl(path: Path, run_dir: Path, issues: list[AcceptanceIssue]) -> None:
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(_issue("error", path, f"line {line_number}: invalid JSON: {exc}", run_dir))
            continue
        if not _has_schema(data):
            issues.append(_issue("error", path, f"line {line_number}: JSONL row missing schema_version", run_dir))


def _audit_csv(path: Path, run_dir: Path, issues: list[AcceptanceIssue]) -> None:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            issues.append(_issue("warning", path, "empty CSV output", run_dir))
            return
    if "schema_version" not in header:
        issues.append(_issue("error", path, "CSV output missing schema_version column", run_dir))


def _audit_experiment_guardrails(manifest: dict[str, Any], run_dir: Path, issues: list[AcceptanceIssue]) -> None:
    experiment_type = manifest.get("experiment_type")
    if experiment_type == "curriculum_ablation" and manifest.get("independent_test_partition") != "test":
        issues.append(_issue("error", run_dir / "run_manifest.json", "curriculum ablation must reserve independent test", run_dir))
    if experiment_type == "skill_proposal":
        _audit_skill_proposal_guardrails(manifest, run_dir, issues)
    if experiment_type == "chunk_world_smoke":
        if not manifest.get("order_independent"):
            issues.append(_issue("error", run_dir / "run_manifest.json", "chunk generation is not order-independent", run_dir))
        boundary = manifest.get("boundary_metrics", {})
        if float(boundary.get("east_west_halo_max_abs_diff", 1.0)) != 0.0:
            issues.append(_issue("error", run_dir / "run_manifest.json", "east/west chunk boundary is not continuous", run_dir))
        if float(boundary.get("north_south_halo_max_abs_diff", 1.0)) != 0.0:
            issues.append(_issue("error", run_dir / "run_manifest.json", "north/south chunk boundary is not continuous", run_dir))
        if not manifest.get("event_survived_reload"):
            issues.append(_issue("error", run_dir / "run_manifest.json", "chunk events did not survive reload", run_dir))


def _audit_skill_proposal_guardrails(
    manifest: dict[str, Any],
    run_dir: Path,
    issues: list[AcceptanceIssue],
) -> None:
    manifest_path = run_dir / "run_manifest.json"
    outputs = manifest.get("outputs", {})
    proposal_ref = outputs.get("proposal_manifest") if isinstance(outputs, dict) else None
    formal = manifest.get("formal_acceptance", {})
    if proposal_ref is None and isinstance(formal, dict):
        proposal_ref = formal.get("gate_report")
    if not proposal_ref:
        issues.append(_issue("error", manifest_path, "skill proposal run is missing proposal_manifest output", run_dir))
        return

    proposal_path = _resolve_run_path(run_dir, proposal_ref)
    if not proposal_path.exists():
        issues.append(_issue("error", proposal_path, "skill proposal manifest is missing", run_dir))
        return
    try:
        proposal = json.loads(proposal_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return

    accepted = proposal.get("accepted", [])
    rejected = proposal.get("rejected", [])
    if not isinstance(accepted, list):
        issues.append(_issue("error", proposal_path, "proposal accepted field must be a list", run_dir))
        accepted = []
    if not isinstance(rejected, list):
        issues.append(_issue("error", proposal_path, "proposal rejected field must be a list", run_dir))
        rejected = []

    if proposal.get("source_partition") != "train":
        issues.append(_issue("error", proposal_path, "skill proposals must use train partition only", run_dir))
    if proposal.get("verification_started") is not False:
        issues.append(_issue("error", proposal_path, "proposal stage must not start verification", run_dir))
    if proposal.get("verified_written") is not False:
        issues.append(_issue("error", proposal_path, "proposal stage must not write verified skills", run_dir))
    if manifest.get("accepted_candidate_count") != len(accepted):
        issues.append(_issue("error", manifest_path, "accepted candidate count does not match proposal manifest", run_dir))
    if manifest.get("rejected_proposal_count") != len(rejected):
        issues.append(_issue("error", manifest_path, "rejected proposal count does not match proposal manifest", run_dir))

    backend = proposal.get("backend")
    if backend == "deepseek":
        metadata = proposal.get("backend_metadata", {})
        if not isinstance(metadata, dict):
            issues.append(_issue("error", proposal_path, "deepseek proposal manifest missing backend metadata", run_dir))
        else:
            for key in ("provider", "model", "base_url", "prompt_hash", "response_received", "api_key_env"):
                if key not in metadata:
                    issues.append(_issue("error", proposal_path, f"deepseek backend metadata missing {key}", run_dir))
            if metadata.get("provider") != "deepseek":
                issues.append(_issue("error", proposal_path, "deepseek backend metadata has wrong provider", run_dir))
            if "api_key" in metadata or "api_key_value" in metadata:
                issues.append(_issue("error", proposal_path, "deepseek backend metadata must not record API secrets", run_dir))

    for index, item in enumerate(accepted):
        if not isinstance(item, dict):
            issues.append(_issue("error", proposal_path, f"accepted proposal {index} must be an object", run_dir))
            continue
        if item.get("status") != "candidate":
            issues.append(_issue("error", proposal_path, f"accepted proposal {index} is not stored as candidate", run_dir))
        if str(item.get("status")) == "verified":
            issues.append(_issue("error", proposal_path, f"accepted proposal {index} attempts verified status", run_dir))
        item_path = item.get("path")
        if not item_path:
            issues.append(_issue("error", proposal_path, f"accepted proposal {index} missing candidate path", run_dir))
            continue
        candidate_path = Path(str(item_path))
        if candidate_path.is_absolute():
            issues.append(_issue("error", proposal_path, f"accepted proposal {index} candidate path must be relative", run_dir))
            continue
        candidate_path = run_dir / candidate_path
        if not candidate_path.exists():
            issues.append(_issue("error", candidate_path, f"accepted proposal {index} candidate file is missing", run_dir))
            continue
        _audit_candidate_record(candidate_path, item, run_dir, issues)

    for index, item in enumerate(rejected):
        if not isinstance(item, dict):
            issues.append(_issue("error", proposal_path, f"rejected proposal {index} must be an object", run_dir))
            continue
        if item.get("event_type") != "proposal_rejected":
            issues.append(_issue("error", proposal_path, f"rejected proposal {index} has unexpected event_type", run_dir))
        if not item.get("reason"):
            issues.append(_issue("error", proposal_path, f"rejected proposal {index} is missing reason", run_dir))


def _audit_candidate_record(
    candidate_path: Path,
    manifest_item: dict[str, Any],
    run_dir: Path,
    issues: list[AcceptanceIssue],
) -> None:
    try:
        record = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return
    try:
        SkillRecord.from_dict(record)
    except Exception as exc:  # noqa: BLE001
        issues.append(_issue("error", candidate_path, f"proposal candidate schema invalid: {exc}", run_dir))
        return
    spec = record.get("spec", {})
    if record.get("storage_status") != "candidate":
        issues.append(_issue("error", candidate_path, "proposal candidate record has non-candidate storage_status", run_dir))
    if not isinstance(spec, dict):
        issues.append(_issue("error", candidate_path, "proposal candidate record missing spec object", run_dir))
        return
    if spec.get("status") != "candidate":
        issues.append(_issue("error", candidate_path, "proposal candidate spec has non-candidate status", run_dir))
    for key in ("skill_id", "version", "spec_hash"):
        if spec.get(key) != manifest_item.get(key):
            issues.append(_issue("error", candidate_path, f"proposal candidate {key} does not match manifest", run_dir))
    source = spec.get("source", {})
    if not isinstance(source, dict):
        issues.append(_issue("error", candidate_path, "proposal candidate source must be an object", run_dir))
        return
    if source.get("proposer") != "llm":
        issues.append(_issue("error", candidate_path, "proposal candidate must record llm proposer", run_dir))
    for episode_id in source.get("source_episode_ids", []):
        if not str(episode_id).startswith("train/"):
            issues.append(_issue("error", candidate_path, "proposal candidate source episode is not train partition", run_dir))


def _audit_formal_acceptance(
    manifest: dict[str, Any],
    run_dir: Path,
    issues: list[AcceptanceIssue],
) -> dict[str, Any]:
    formal = manifest.get("formal_acceptance", {})
    summary: dict[str, Any] = {
        "manifest_passed": formal.get("passed") if isinstance(formal, dict) else None,
        "claimed_conclusion_level": formal.get("conclusion_level") if isinstance(formal, dict) else None,
        "gate_report": formal.get("gate_report") if isinstance(formal, dict) else None,
        "gate_report_passed": None,
        "validated": None,
        "failed_gates": [],
    }
    if not isinstance(formal, dict):
        return summary

    gate_report_ref = formal.get("gate_report")
    if not gate_report_ref:
        if formal.get("passed") is True and manifest.get("experiment_type") in FORMAL_GATE_PROTOCOLS:
            issues.append(_issue("error", run_dir / "run_manifest.json", "formal acceptance is missing gate_report", run_dir))
            summary["validated"] = False
        return summary

    gate_report_path = Path(str(gate_report_ref))
    if not gate_report_path.is_absolute():
        gate_report_path = run_dir / gate_report_path
    if not gate_report_path.exists():
        if formal.get("passed") is True:
            issues.append(_issue("error", gate_report_path, "formal gate_report is missing", run_dir))
        summary["validated"] = False
        return summary

    try:
        gate_report = json.loads(gate_report_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        if formal.get("passed") is True:
            issues.append(_issue("error", gate_report_path, f"formal gate_report has invalid JSON: {exc}", run_dir))
        summary["validated"] = False
        return summary

    protocol = FORMAL_GATE_PROTOCOLS.get(str(manifest.get("experiment_type", "")), {})
    gate_key = str(protocol.get("gate_key", "gate_id"))
    failed_gates = [str(_gate_label(gate, gate_key)) for gate in gate_report.get("gates", []) if gate.get("passed") is not True]
    required_gates = set(protocol.get("required_gates", set()))
    observed_gates = {str(_gate_label(gate, gate_key)) for gate in gate_report.get("gates", [])}
    missing_gates = sorted(required_gates - observed_gates)
    verified_decision = protocol.get("verified_decision")
    decision_passed = verified_decision is None or gate_report.get("decision") == verified_decision
    report_pass_flag = bool(gate_report.get("passed", True))
    gate_report_passed = report_pass_flag and not failed_gates and not missing_gates and decision_passed
    summary.update(
        {
            "protocol_id": gate_report.get("protocol_id"),
            "gate_report_passed": gate_report_passed,
            "validated": gate_report_passed if formal.get("passed") is True else None,
            "failed_gates": failed_gates,
            "missing_gates": missing_gates,
            "decision": gate_report.get("decision"),
        }
    )
    expected_protocol = protocol.get("protocol_id")
    if formal.get("passed") is True and expected_protocol and gate_report.get("protocol_id") != expected_protocol:
        issues.append(_issue("error", gate_report_path, "formal gate_report has unexpected protocol_id", run_dir))
        summary["validated"] = False
    if formal.get("passed") is True and not decision_passed:
        issues.append(_issue("error", gate_report_path, "formal gate_report decision is not verified", run_dir))
        summary["validated"] = False
    if formal.get("passed") is True and not gate_report_passed:
        issues.append(_issue("error", gate_report_path, "formal gates did not all pass", run_dir))
        summary["validated"] = False
    return summary


def _resolve_run_path(run_dir: Path, path: Any) -> Path:
    resolved = Path(str(path))
    if not resolved.is_absolute():
        resolved = run_dir / resolved
    return resolved


def _gate_label(gate: dict[str, Any], gate_key: str) -> str:
    return str(gate.get(gate_key, gate.get("gate_id", gate.get("gate", gate.get("name", "unknown")))))


def _has_schema(data: Any) -> bool:
    if isinstance(data, dict):
        return "schema_version" in data or (isinstance(data.get("spec"), dict) and "schema_version" in data["spec"])
    return False


def _issue(severity: str, path: Path, message: str, run_dir: Path) -> AcceptanceIssue:
    try:
        relative = str(path.relative_to(run_dir)).replace("\\", "/")
    except ValueError:
        relative = str(path)
    return AcceptanceIssue(severity=severity, path=relative, message=message)
