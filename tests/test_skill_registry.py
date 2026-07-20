from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from evogrid.skills.registry import SkillRegistry
from evogrid.skills.schemas import SkillSpec, VerificationReport
from tests.test_skill_schema import _report_dict, _skill_dict


class SkillRegistryTest(unittest.TestCase):
    def test_register_candidate_writes_physical_record_and_event(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            spec = SkillSpec.from_dict(_skill_dict())

            record = registry.register_candidate(spec)

            path = Path(temp) / "candidates" / spec.skill_id / f"{spec.version}.json"
            self.assertTrue(path.exists())
            self.assertEqual(record.spec.spec_hash, spec.spec_hash)
            events = (Path(temp) / "registry_events.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(events), 1)
            self.assertEqual(json.loads(events[0])["event_type"], "register_candidate")

    def test_begin_verification_uses_lease_and_updates_status(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            spec = SkillSpec.from_dict(_skill_dict())
            registry.register_candidate(spec)

            lease = registry.begin_verification(spec.skill_id, spec.version)

            self.assertTrue(lease.lease_id)
            with self.assertRaisesRegex(RuntimeError, "lease"):
                registry.begin_verification(spec.skill_id, spec.version)

    def test_apply_verified_report_moves_record_to_verified_bucket(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            spec = SkillSpec.from_dict(_skill_dict())
            registry.register_candidate(spec)
            lease = registry.begin_verification(spec.skill_id, spec.version)
            report = VerificationReport.from_dict(_report_dict(spec))

            record = registry.apply_verification(report, lease_id=lease.lease_id)

            self.assertEqual(record.spec.status, "verified")
            self.assertTrue((Path(temp) / "verified" / spec.skill_id / f"{spec.version}.json").exists())
            self.assertFalse((Path(temp) / "candidates" / spec.skill_id / f"{spec.version}.json").exists())
            self.assertTrue((Path(temp) / "reports" / spec.skill_id / spec.version / f"{report.verification_id}.json").exists())

    def test_report_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            spec = SkillSpec.from_dict(_skill_dict())
            other = SkillSpec.from_dict({**_skill_dict(), "skill_id": "other_skill"})
            registry.register_candidate(spec)
            lease = registry.begin_verification(spec.skill_id, spec.version)
            report_data = _report_dict(other)
            report_data["skill_id"] = spec.skill_id
            report = VerificationReport.from_dict(report_data)

            with self.assertRaisesRegex(ValueError, "spec_hash"):
                registry.apply_verification(report, lease_id=lease.lease_id)

    def test_revision_required_stays_in_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            spec = SkillSpec.from_dict(_skill_dict())
            registry.register_candidate(spec)
            lease = registry.begin_verification(spec.skill_id, spec.version)
            report_data = _report_dict(spec)
            report_data["decision"] = "revision_required"
            for gate in report_data["gates"]:
                if gate["gate"] == "G3_negative_safety":
                    gate["passed"] = False
            report_data["failure_reasons"] = ["G3_negative_safety"]
            report = VerificationReport.from_dict(report_data)

            record = registry.apply_verification(report, lease_id=lease.lease_id)

            self.assertEqual(record.spec.status, "revision_required")
            self.assertTrue((Path(temp) / "candidates" / spec.skill_id / f"{spec.version}.json").exists())

    def test_verified_report_with_failed_gate_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            spec = SkillSpec.from_dict(_skill_dict())
            registry.register_candidate(spec)
            lease = registry.begin_verification(spec.skill_id, spec.version)
            report_data = _report_dict(spec)
            for gate in report_data["gates"]:
                if gate["gate"] == "G1_effect":
                    gate["passed"] = False
            report_data["decision"] = "rejected"
            report_data["failure_reasons"] = ["G1_effect"]
            report = VerificationReport.from_dict(report_data)

            record = registry.apply_verification(report, lease_id=lease.lease_id)

            self.assertEqual(record.spec.status, "rejected")

    def test_verified_decision_with_failed_gate_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            spec = SkillSpec.from_dict(_skill_dict())
            registry.register_candidate(spec)
            lease = registry.begin_verification(spec.skill_id, spec.version)
            report_data = _report_dict(spec)
            for gate in report_data["gates"]:
                if gate["gate"] == "G1_effect":
                    gate["passed"] = False
            report_data["failure_reasons"] = ["G1_effect"]
            report = VerificationReport.from_dict(report_data)

            with self.assertRaisesRegex(ValueError, "decision inconsistent"):
                registry.apply_verification(report, lease_id=lease.lease_id)

    def test_verified_report_with_failure_reasons_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            spec = SkillSpec.from_dict(_skill_dict())
            registry.register_candidate(spec)
            lease = registry.begin_verification(spec.skill_id, spec.version)
            report_data = _report_dict(spec)
            report_data["failure_reasons"] = ["manual_override"]
            report = VerificationReport.from_dict(report_data)

            with self.assertRaisesRegex(ValueError, "failure_reasons"):
                registry.apply_verification(report, lease_id=lease.lease_id)

    def test_missing_required_gate_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            spec = SkillSpec.from_dict(_skill_dict())
            registry.register_candidate(spec)
            lease = registry.begin_verification(spec.skill_id, spec.version)
            report_data = _report_dict(spec)
            report_data["gates"] = [gate for gate in report_data["gates"] if gate["gate"] != "G5_non_redundancy"]
            report = VerificationReport.from_dict(report_data)

            with self.assertRaisesRegex(ValueError, "missing required gates"):
                registry.apply_verification(report, lease_id=lease.lease_id)

    def test_duplicate_verify_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            spec = SkillSpec.from_dict(_skill_dict())
            registry.register_candidate(spec)
            lease = registry.begin_verification(spec.skill_id, spec.version)
            report_data = _report_dict(spec)
            report_data["paired_seeds"] = [101, 101]
            report = VerificationReport.from_dict(report_data)

            with self.assertRaisesRegex(ValueError, "paired_seeds must be unique"):
                registry.apply_verification(report, lease_id=lease.lease_id)

    def test_apply_verification_requires_matching_lease_id(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            spec = SkillSpec.from_dict(_skill_dict())
            registry.register_candidate(spec)
            registry.begin_verification(spec.skill_id, spec.version)
            report = VerificationReport.from_dict(_report_dict(spec))

            with self.assertRaisesRegex(ValueError, "lease_id"):
                registry.apply_verification(report, lease_id="wrong-lease")

    def test_deprecate_moves_verified_record(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            spec = SkillSpec.from_dict(_skill_dict())
            registry.register_candidate(spec)
            lease = registry.begin_verification(spec.skill_id, spec.version)
            registry.apply_verification(VerificationReport.from_dict(_report_dict(spec)), lease_id=lease.lease_id)

            deprecated = registry.deprecate(spec.skill_id, spec.version, "superseded")

            self.assertEqual(deprecated.spec.status, "deprecated")
            self.assertTrue((Path(temp) / "deprecated" / spec.skill_id / f"{spec.version}.json").exists())
            self.assertFalse((Path(temp) / "verified" / spec.skill_id / f"{spec.version}.json").exists())

    def test_list_available_reload_from_disk(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            spec = SkillSpec.from_dict(_skill_dict())
            registry.register_candidate(spec)
            lease = registry.begin_verification(spec.skill_id, spec.version)
            registry.apply_verification(VerificationReport.from_dict(_report_dict(spec)), lease_id=lease.lease_id)

            reloaded = SkillRegistry(temp).list_available(status="verified")

            self.assertEqual(len(reloaded), 1)
            self.assertEqual(reloaded[0].spec.spec_hash, spec.spec_hash)


if __name__ == "__main__":
    unittest.main()
