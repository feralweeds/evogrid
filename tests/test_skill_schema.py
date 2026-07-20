from __future__ import annotations

import json
import unittest

from evogrid.skills.schemas import (
    SkillRecord,
    SkillSpec,
    VerificationReport,
    compute_spec_hash,
    validate_status_transition,
)


class SkillSchemaTest(unittest.TestCase):
    def test_skill_spec_computes_canonical_hash_and_serializes(self):
        spec = SkillSpec.from_dict(_skill_dict())

        self.assertTrue(spec.spec_hash)
        self.assertEqual(spec.spec_hash, compute_spec_hash(spec))
        json.dumps(spec.to_dict(), sort_keys=True)

    def test_status_transition_does_not_change_spec_hash(self):
        candidate = SkillSpec.from_dict(_skill_dict(status="candidate"))
        verifying_data = candidate.to_dict()
        verifying_data["status"] = "verifying"
        verifying_data.pop("spec_hash")
        verifying = SkillSpec.from_dict(verifying_data)

        self.assertEqual(candidate.spec_hash, verifying.spec_hash)

    def test_executable_change_changes_hash(self):
        first = SkillSpec.from_dict(_skill_dict())
        changed = _skill_dict()
        changed["procedure"] = [
            {"op": "ESTIMATE", "estimator": "future_route_uses", "store_as": "n_use"},
            {"op": "RETURN", "result": "not_applicable"},
        ]
        second = SkillSpec.from_dict(changed)

        self.assertNotEqual(first.spec_hash, second.spec_hash)

    def test_text_change_changes_hash_but_text_is_not_executable(self):
        first = SkillSpec.from_dict(_skill_dict(description="Build where reuse pays."))
        second = SkillSpec.from_dict(_skill_dict(description="Build on repeated costly transport."))

        self.assertNotEqual(first.spec_hash, second.spec_hash)

    def test_rejects_natural_language_inside_procedure(self):
        data = _skill_dict()
        data["procedure"] = [{"op": "ACT", "action": "BUILD_ROAD", "rationale": "because it seems useful"}]

        with self.assertRaisesRegex(ValueError, "natural-language"):
            SkillSpec.from_dict(data)

    def test_rejects_bad_hash(self):
        data = _skill_dict()
        data["spec_hash"] = "bad"

        with self.assertRaisesRegex(ValueError, "spec_hash"):
            SkillSpec.from_dict(data)

    def test_llm_cannot_directly_create_verified_skill(self):
        data = _skill_dict(status="verified")
        data["source"]["proposer"] = "llm"

        with self.assertRaisesRegex(ValueError, "LLM"):
            SkillSpec.from_dict(data)

    def test_status_machine_allows_only_registered_transitions(self):
        validate_status_transition("proposed", "candidate")
        validate_status_transition("candidate", "verifying")
        validate_status_transition("verifying", "verified")
        validate_status_transition("verified", "deprecated")

        with self.assertRaisesRegex(ValueError, "not allowed"):
            validate_status_transition("candidate", "verified")
        with self.assertRaisesRegex(ValueError, "not allowed"):
            validate_status_transition("rejected", "candidate")

    def test_skill_record_requires_storage_status_match(self):
        spec = SkillSpec.from_dict(_skill_dict())
        record = SkillRecord.from_dict({"spec": spec.to_dict(), "storage_status": "candidate"})

        self.assertEqual(record.spec.spec_hash, spec.spec_hash)

        with self.assertRaisesRegex(ValueError, "storage_status"):
            SkillRecord.from_dict({"spec": spec.to_dict(), "storage_status": "verified"})

    def test_verification_report_hash_and_decision_contract(self):
        spec = SkillSpec.from_dict(_skill_dict())
        report = VerificationReport.from_dict(_report_dict(spec))

        self.assertTrue(report.report_hash)
        self.assertEqual(report.spec_hash, spec.spec_hash)
        json.dumps(report.to_dict(), sort_keys=True)

    def test_verification_report_rejects_sample_size_mismatch(self):
        spec = SkillSpec.from_dict(_skill_dict())
        report = _report_dict(spec)
        report["sample_size"] = 3

        with self.assertRaisesRegex(ValueError, "sample_size"):
            VerificationReport.from_dict(report)

    def test_unknown_procedure_op_is_rejected(self):
        data = _skill_dict()
        data["procedure"] = [{"op": "EXEC", "code": "print(1)"}]

        with self.assertRaisesRegex(ValueError, "procedure.op"):
            SkillSpec.from_dict(data)

    def test_unknown_applicability_feature_is_rejected(self):
        data = _skill_dict()
        data["applicability"] = {"all": [{"feature": "current.road_exists", "op": "eq", "value": False}]}

        with self.assertRaisesRegex(ValueError, "applicability.feature"):
            SkillSpec.from_dict(data)

    def test_enum_applicability_rejects_ordered_numeric_comparison(self):
        data = _skill_dict()
        data["applicability"]["all"].append(
            {"feature": "route.remaining_length_bucket", "op": "gte", "value": 2}
        )

        with self.assertRaisesRegex(ValueError, "enum equality"):
            SkillSpec.from_dict(data)

    def test_enum_applicability_accepts_string_bucket_membership(self):
        data = _skill_dict()
        data["applicability"]["all"].append(
            {"feature": "route.remaining_length_bucket", "op": "in", "value": ["medium", "long"]}
        )

        spec = SkillSpec.from_dict(data)

        self.assertTrue(spec.spec_hash)


def _skill_dict(status: str = "candidate", description: str = "Build only where repeated transport can repay construction.") -> dict:
    return {
        "schema_version": 1,
        "skill_id": "reusable_road_building",
        "version": "1.0.0",
        "status": status,
        "name": "Reusable road building",
        "description": description,
        "problem_addressed": "Repeated high-cost transport",
        "source": {
            "proposer": "fixture",
            "source_episode_ids": ["run/episode/3"],
            "base_prompt_hash": "abc123",
        },
        "applicability": {
            "all": [
                {"feature": "current.terrain_band", "op": "in", "value": ["ROUGH", "VERY_ROUGH"]},
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
        "rationale": "Fixture for schema tests only.",
    }


def _report_dict(spec: SkillSpec) -> dict:
    return {
        "schema_version": 1,
        "verification_id": "verify_reusable_road_building_001",
        "skill_id": spec.skill_id,
        "skill_version": spec.version,
        "spec_hash": spec.spec_hash,
        "protocol_id": "skill_verification_v1",
        "candidate_source_partitions": ["train"],
        "verification_partition": "verify",
        "paired_seeds": [101, 102],
        "environment_strata": ["positive", "negative"],
        "baseline": "same_agent_skill_disabled",
        "sample_size": 2,
        "metrics": {"paired_delta_mean": 0.2},
        "gates": [
            {"gate": "G0_data_integrity", "passed": True},
            {"gate": "G1_effect", "passed": True},
            {"gate": "G2_reliability", "passed": True},
            {"gate": "G3_negative_safety", "passed": True},
            {"gate": "G4_transfer", "passed": True},
            {"gate": "G5_non_redundancy", "passed": True},
        ],
        "decision": "verified",
        "failure_reasons": [],
        "created_at": "2026-07-19T00:00:00Z",
    }


if __name__ == "__main__":
    unittest.main()
