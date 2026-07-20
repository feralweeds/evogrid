from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from evogrid.llm.skill_prompts import SKILL_REVISION_PROMPT, SKILL_REVISION_PROMPT_VERSION
from scripts.run_skill_revision import _deepseek_revision_backend, _revision_payload, _verification_feedback_summary, run_skill_revision
from tests.test_skill_schema import _report_dict, _skill_dict
from evogrid.skills.schemas import SkillSpec, VerificationReport


class SkillRevisionScriptTest(unittest.TestCase):
    def test_fixture_revision_writes_new_candidate_only(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            candidate_path = temp_path / "candidate.json"
            report_path = temp_path / "report.json"
            config_path = temp_path / "revision.yaml"
            out_dir = temp_path / "out"
            spec = SkillSpec.from_dict(_skill_dict(status="candidate"))
            candidate_path.write_text(json.dumps({"schema_version": 1, "spec": spec.to_dict()}), encoding="utf-8")
            report = VerificationReport.from_dict(_report_dict(spec))
            report_path.write_text(json.dumps(report.to_dict()), encoding="utf-8")
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "revision": {
                            "schema_version": 1,
                            "mode": "revision_pilot",
                            "backend": "fixture_revision_valid",
                            "candidate_path": str(candidate_path),
                            "verification_report_path": str(report_path),
                            "new_version": "1.0.1",
                        }
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_skill_revision(config_path, out_dir)

            proposal = json.loads((out_dir / "proposal_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["accepted_candidate_count"], 1)
            self.assertEqual(proposal["accepted"][0]["version"], "1.0.1")
            self.assertFalse(proposal["verification_started"])
            self.assertFalse(proposal["verified_written"])
            candidate = out_dir / proposal["accepted"][0]["path"]
            record = json.loads(candidate.read_text(encoding="utf-8"))
            self.assertEqual(record["spec"]["status"], "candidate")
            self.assertEqual(record["spec"]["source"]["revision_of_spec_hash"], spec.spec_hash)

    def test_noop_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            candidate_path = temp_path / "candidate.json"
            report_path = temp_path / "report.json"
            config_path = temp_path / "revision.yaml"
            out_dir = temp_path / "out"
            spec = SkillSpec.from_dict(_skill_dict(status="candidate"))
            candidate_path.write_text(json.dumps({"schema_version": 1, "spec": spec.to_dict()}), encoding="utf-8")
            report = VerificationReport.from_dict(_report_dict(spec))
            report_path.write_text(json.dumps(report.to_dict()), encoding="utf-8")
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "revision": {
                            "schema_version": 1,
                            "mode": "revision_pilot",
                            "backend": "fixture_revision_noop",
                            "candidate_path": str(candidate_path),
                            "verification_report_path": str(report_path),
                            "new_version": "1.0.1",
                        }
                    }
                ),
                encoding="utf-8",
            )

            manifest = run_skill_revision(config_path, out_dir)

            proposal = json.loads((out_dir / "proposal_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["accepted_candidate_count"], 0)
            self.assertEqual(manifest["rejected_proposal_count"], 1)
            self.assertIn("executable Skill contract", proposal["rejected"][0]["reason"])

    def test_revision_feedback_summary_omits_per_seed_rows(self):
        spec = SkillSpec.from_dict(_skill_dict(status="candidate"))
        report = VerificationReport.from_dict(_report_dict(spec)).to_dict()
        report["paired_seeds"] = [1, 2, 3]
        report["metrics"]["disabled"] = [{"road_net_payoff": 0.0, "num_build_road": 0}]
        report["metrics"]["enabled"] = [
            {"road_net_payoff": 1.0, "num_build_road": 2, "road_usage_rate": 0.5, "activation_rate": 1.0},
            {"road_net_payoff": 0.0, "num_build_road": 4, "road_usage_rate": 0.0, "activation_rate": 0.0},
        ]

        summary = _verification_feedback_summary(report)
        self.assertNotIn("paired_seeds", summary)
        self.assertNotIn("disabled", summary["metrics"])
        self.assertNotIn("enabled", summary["metrics"])
        self.assertNotIn("paired_deltas", summary["metrics"])
        self.assertIn("omitted_fields", summary)
        aggregates = summary["metrics"]["aggregate_episode_metrics"]
        self.assertEqual(aggregates["enabled_mean_num_build_road"], 3.0)
        self.assertEqual(aggregates["disabled_mean_num_build_road"], 0.0)
        self.assertEqual(aggregates["enabled_mean_road_usage_rate"], 0.25)
        self.assertEqual(aggregates["activated_mean_num_build_road"], 2.0)
        self.assertEqual(aggregates["activated_mean_road_net_payoff"], 1.0)

    def test_revision_payload_exposes_allowed_feature_contract(self):
        spec = SkillSpec.from_dict(_skill_dict(status="candidate"))

        payload = _revision_payload(spec, {"failure_reasons": ["G3_negative_safety"]}, "1.0.2")

        self.assertIn("current.tile_type", payload["constraints"]["allowed_applicability_features"])
        self.assertIn("route.is_known_transport_route", payload["constraints"]["allowed_applicability_features"])
        self.assertIn("not_in", payload["constraints"]["allowed_applicability_ops"])
        self.assertIn("current.road_exists", payload["constraints"]["forbidden_applicability_features"])
        self.assertIn("not in", payload["constraints"]["forbidden_applicability_ops"])
        self.assertEqual(payload["constraints"]["observable_tile_ids"]["GROUND"], 0)
        self.assertEqual(payload["constraints"]["build_road_tile_type_guard"]["value"], [0, 4])
        route_bucket = payload["constraints"]["bucket_feature_contract"]["route.remaining_length_bucket"]
        self.assertEqual(route_bucket["allowed_ops"], ["eq", "ne", "in", "not_in"])
        self.assertEqual(route_bucket["valid_examples"][0]["value"], ["medium", "long"])
        self.assertEqual(route_bucket["invalid_examples"][0]["op"], "gte")
        self.assertEqual(
            payload["constraints"]["recommended_executable_revision_examples"][0]["applicability_leaf"]["value"],
            ["medium", "long"],
        )

    def test_revision_prompt_names_feature_contract(self):
        self.assertEqual(SKILL_REVISION_PROMPT_VERSION, "1.7.0")
        self.assertIn("allowed_applicability_features", SKILL_REVISION_PROMPT)
        self.assertIn("allowed_applicability_ops", SKILL_REVISION_PROMPT)
        self.assertIn("current.road_exists", SKILL_REVISION_PROMPT)
        self.assertIn("not in", SKILL_REVISION_PROMPT)
        self.assertIn("not_in", SKILL_REVISION_PROMPT)
        self.assertIn("current.tile_type == 0", SKILL_REVISION_PROMPT)
        self.assertIn("zero activation", SKILL_REVISION_PROMPT)
        self.assertIn("ACT BUILD_ROAD", SKILL_REVISION_PROMPT)
        self.assertIn("metadata-only", SKILL_REVISION_PROMPT)
        self.assertIn("route.remaining_length_bucket is one of short", SKILL_REVISION_PROMPT)
        self.assertIn("never numeric comparisons", SKILL_REVISION_PROMPT)
        self.assertIn("\"route.remaining_length_bucket\",\"op\":\"in\"", SKILL_REVISION_PROMPT)

    def test_deepseek_revision_backend_sends_allowed_features(self):
        metadata = {}
        backend = _deepseek_revision_backend({"temperature": 0.1}, metadata, client=_FakeDeepSeekClient())
        spec = SkillSpec.from_dict(_skill_dict(status="candidate"))

        backend(_revision_payload(spec, {"failure_reasons": ["G1_effect"]}, "1.0.2"))

        user_payload = json.loads(_FakeDeepSeekClient.messages[1]["content"])
        self.assertIn("allowed_applicability_features", user_payload["constraints"])
        self.assertIn("allowed_applicability_ops", user_payload["constraints"])
        self.assertTrue(metadata["response_received"])


class _FakeDeepSeekClient:
    model = "fake-deepseek"
    base_url = "https://example.test"
    messages = []

    def chat_completion(self, messages, temperature=0.2, json_mode=True):
        type(self).messages = messages
        return {
            "content": '{"skills": []}',
            "finish_reason": "stop",
            "model": self.model,
            "usage": {"total_tokens": 7},
        }


if __name__ == "__main__":
    unittest.main()
