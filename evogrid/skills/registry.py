"""Filesystem-backed Skill registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from evogrid.skills.schemas import SkillRecord, SkillSpec, VerificationReport, validate_status_transition

REQUIRED_VERIFICATION_GATES = {
    "G0_data_integrity",
    "G1_effect",
    "G2_reliability",
    "G3_negative_safety",
    "G4_transfer",
    "G5_non_redundancy",
}
FORMAL_SKILL_PROTOCOL_ID = "skill_verification_v1"


@dataclass(frozen=True)
class VerificationLease:
    skill_id: str
    version: str
    lease_id: str


class SkillRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        for child in ("candidates", "verified", "deprecated", "reports"):
            (self.root / child).mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "registry_events.jsonl"
        self._leases: dict[tuple[str, str], str] = {}

    def register_candidate(self, spec: SkillSpec) -> SkillRecord:
        if spec.status not in {"candidate", "revision_required", "deprecated"}:
            raise ValueError("register_candidate requires candidate/revision/deprecated source status")
        if spec.status != "candidate":
            validate_status_transition(spec.status, "candidate")
            spec = _with_status(spec, "candidate")
        record = SkillRecord(spec=spec, storage_status="candidate", created_at=_now(), updated_at=_now())
        self._write_record("candidates", record)
        self._append_event("register_candidate", record)
        return record

    def begin_verification(self, skill_id: str, version: str) -> VerificationLease:
        record = self._load_record("candidates", skill_id, version)
        key = (skill_id, version)
        if key in self._leases:
            raise RuntimeError("verification lease already active")
        validate_status_transition(record.spec.status, "verifying")
        verifying = SkillRecord(
            spec=_with_status(record.spec, "verifying"),
            storage_status="verifying",
            created_at=record.created_at,
            updated_at=_now(),
            verification_reports=list(record.verification_reports),
        )
        self._write_record("candidates", verifying)
        lease = VerificationLease(skill_id=skill_id, version=version, lease_id=str(uuid.uuid4()))
        self._leases[key] = lease.lease_id
        self._append_event("begin_verification", verifying, {"lease_id": lease.lease_id})
        return lease

    def apply_verification(self, report: VerificationReport, lease_id: str | None = None) -> SkillRecord:
        record = self._load_record("candidates", report.skill_id, report.skill_version)
        if record.spec.status != "verifying":
            raise ValueError("apply_verification requires candidate in verifying status")
        expected_lease = self._leases.get((report.skill_id, report.skill_version))
        if expected_lease is None:
            raise ValueError("apply_verification requires an active verification lease")
        if lease_id != expected_lease:
            raise ValueError("verification lease_id does not match active lease")
        if report.spec_hash != record.spec.spec_hash:
            raise ValueError("verification report spec_hash does not match candidate")
        _validate_report_decision(report, record.spec)
        if report.decision == "verified":
            next_status = "verified"
            target_bucket = "verified"
        elif report.decision == "verification_invalid":
            next_status = "revision_required"
            target_bucket = "candidates"
        else:
            next_status = report.decision
            target_bucket = "candidates"
        validate_status_transition(record.spec.status, next_status)
        report_path = self._write_report(report)
        next_record = SkillRecord(
            spec=_with_status(record.spec, next_status),
            storage_status=next_status,
            created_at=record.created_at,
            updated_at=_now(),
            verification_reports=[*record.verification_reports, str(report_path.relative_to(self.root))],
        )
        self._write_record(target_bucket, next_record)
        if target_bucket != "candidates":
            self._delete_record("candidates", record.spec.skill_id, record.spec.version)
        self._leases.pop((record.spec.skill_id, record.spec.version), None)
        self._append_event("apply_verification", next_record, {"report": str(report_path.relative_to(self.root))})
        return next_record

    def deprecate(self, skill_id: str, version: str, reason: str) -> SkillRecord:
        record = self._load_record("verified", skill_id, version)
        validate_status_transition(record.spec.status, "deprecated")
        deprecated = SkillRecord(
            spec=_with_status(record.spec, "deprecated"),
            storage_status="deprecated",
            created_at=record.created_at,
            updated_at=_now(),
            verification_reports=list(record.verification_reports),
        )
        self._write_record("deprecated", deprecated)
        self._delete_record("verified", skill_id, version)
        self._append_event("deprecate", deprecated, {"reason": reason})
        return deprecated

    def list_available(self, context=None, status: str = "verified") -> list[SkillRecord]:
        bucket = _bucket_for_status(status)
        records = []
        for path in sorted((self.root / bucket).glob("*/*.json")):
            record = SkillRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if record.spec.status == status:
                records.append(record)
        return records

    def _write_record(self, bucket: str, record: SkillRecord) -> Path:
        path = self.root / bucket / record.spec.skill_id / f"{record.spec.version}.json"
        _atomic_write_json(path, record.to_dict())
        return path

    def _load_record(self, bucket: str, skill_id: str, version: str) -> SkillRecord:
        path = self.root / bucket / skill_id / f"{version}.json"
        if not path.exists():
            raise FileNotFoundError(str(path))
        return SkillRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _delete_record(self, bucket: str, skill_id: str, version: str) -> None:
        path = self.root / bucket / skill_id / f"{version}.json"
        if path.exists():
            path.unlink()

    def _write_report(self, report: VerificationReport) -> Path:
        path = self.root / "reports" / report.skill_id / report.skill_version / f"{report.verification_id}.json"
        _atomic_write_json(path, report.to_dict())
        return path

    def _append_event(self, event_type: str, record: SkillRecord, extra: dict | None = None) -> None:
        event = {
            "schema_version": 1,
            "event_type": event_type,
            "created_at": _now(),
            "skill_id": record.spec.skill_id,
            "version": record.spec.version,
            "status": record.spec.status,
            "spec_hash": record.spec.spec_hash,
        }
        if extra:
            event.update(extra)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _with_status(spec: SkillSpec, status: str) -> SkillSpec:
    data = spec.to_dict()
    data["status"] = status
    data.pop("spec_hash", None)
    return SkillSpec.from_dict(data)


def _bucket_for_status(status: str) -> str:
    if status in {"candidate", "verifying", "revision_required", "rejected"}:
        return "candidates"
    if status == "verified":
        return "verified"
    if status == "deprecated":
        return "deprecated"
    raise ValueError(f"unsupported status: {status}")


def _validate_report_decision(report: VerificationReport, spec: SkillSpec) -> None:
    if report.protocol_id != FORMAL_SKILL_PROTOCOL_ID:
        raise ValueError("verification report protocol_id is not supported")
    if report.skill_id != spec.skill_id or report.skill_version != spec.version:
        raise ValueError("verification report candidate identity does not match candidate")
    if report.verification_partition != "verify":
        raise ValueError("verification report must use verify partition")
    if "train" not in report.candidate_source_partitions:
        raise ValueError("verification report must record train source partition")
    if report.sample_size != len(report.paired_seeds):
        raise ValueError("verification report sample_size must match paired_seeds")
    if len(set(report.paired_seeds)) != len(report.paired_seeds):
        raise ValueError("verification report paired_seeds must be unique")
    if not report.gates:
        raise ValueError("verification report gates are required")
    observed_gates = {str(gate.get("gate", gate.get("gate_id", "<unknown>"))) for gate in report.gates}
    missing = sorted(REQUIRED_VERIFICATION_GATES - observed_gates)
    if missing:
        raise ValueError(f"verification report missing required gates: {missing}")
    failed = [str(gate.get("gate", gate.get("gate_id", "<unknown>"))) for gate in report.gates if gate.get("passed") is not True]
    expected_decision = _expected_decision(observed_gates, failed)
    if report.decision != expected_decision:
        raise ValueError(f"verification report decision inconsistent with gates: expected {expected_decision}")
    expected_failure_reasons = sorted(failed)
    if sorted(report.failure_reasons) != expected_failure_reasons:
        raise ValueError("verification report failure_reasons inconsistent with failed gates")
    if report.decision == "verified":
        if failed:
            raise ValueError(f"verified report has failed gates: {failed}")
        if report.failure_reasons:
            raise ValueError("verified report must not include failure_reasons")


def _expected_decision(observed_gates: set[str], failed: list[str]) -> str:
    if not failed:
        return "verified"
    failed_set = set(failed)
    if "G0_data_integrity" in failed_set:
        return "verification_invalid"
    if "G1_effect" in failed_set:
        return "rejected"
    return "revision_required"


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
