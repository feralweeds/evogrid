from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

import yaml

from scripts.run_skill_verification import run_skill_verification
from tests.test_skill_schema import _skill_dict


class SkillVerificationScriptTest(unittest.TestCase):
    def test_smoke_verification_writes_report_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            candidate_path = temp_path / "candidate.json"
            config_path = temp_path / "config.yaml"
            out_dir = temp_path / "out"
            candidate_path.write_text(json.dumps(_skill_dict(status="candidate")), encoding="utf-8")
            config_path.write_text(yaml.safe_dump(_config("fixture_positive")), encoding="utf-8")

            manifest = run_skill_verification(candidate_path, config_path, out_dir)

            self.assertEqual(manifest["completion_status"], "completed")
            self.assertEqual(manifest["decision"], "verified")
            self.assertTrue(manifest["smoke_not_promoted"])
            self.assertEqual(manifest["formal_acceptance"]["passed"], False)
            self.assertEqual(manifest["formal_acceptance"]["gate_report"], manifest["report"])
            report_path = out_dir / manifest["report"]
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["decision"], "verified")
            self.assertIn("paired_delta_bootstrap_ci", report["metrics"])
            self.assertIn("G4_transfer", [gate["gate"] for gate in report["gates"]])

    def test_failure_still_writes_complete_report(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            candidate_path = temp_path / "candidate.json"
            config_path = temp_path / "config.yaml"
            out_dir = temp_path / "out"
            candidate_path.write_text(json.dumps(_skill_dict(status="candidate")), encoding="utf-8")
            config_path.write_text(yaml.safe_dump(_config("fixture_negative")), encoding="utf-8")

            manifest = run_skill_verification(candidate_path, config_path, out_dir)

            report = json.loads((out_dir / manifest["report"]).read_text(encoding="utf-8"))
            self.assertEqual(report["decision"], "rejected")
            self.assertTrue(report["failure_reasons"])

    def test_formal_small_sample_cannot_claim_e2(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            candidate_path = temp_path / "candidate.json"
            config_path = temp_path / "config.yaml"
            out_dir = temp_path / "out"
            candidate_path.write_text(json.dumps(_skill_dict(status="candidate")), encoding="utf-8")
            config = _config("fixture_positive")
            config["verification"]["mode"] = "formal"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            manifest = run_skill_verification(candidate_path, config_path, out_dir)

            self.assertEqual(manifest["decision"], "verified")
            self.assertFalse(manifest["formal_acceptance"]["passed"])
            self.assertEqual(manifest["formal_acceptance"]["conclusion_level"], "E0")
            self.assertFalse(manifest["formal_readiness"]["passed"])
            self.assertIn("formal stratum sample size below 30 paired seeds", manifest["formal_readiness"]["failures"])

    def test_formal_minimum_sample_can_claim_e2_for_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            candidate_path = temp_path / "candidate.json"
            config_path = temp_path / "config.yaml"
            out_dir = temp_path / "out"
            candidate_path.write_text(json.dumps(_skill_dict(status="candidate")), encoding="utf-8")
            config = _config("fixture_positive")
            config["verification"]["mode"] = "formal"
            config["verification"]["paired_seeds"] = list(range(60))
            config["verification"]["environment_strata"] = ["fixture_positive", "fixture_negative"]
            config["verification"]["min_paired_samples"] = 60
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            manifest = run_skill_verification(candidate_path, config_path, out_dir)

            self.assertTrue(manifest["formal_readiness"]["passed"])
            self.assertEqual(manifest["formal_readiness"]["strata_counts"], {"fixture_negative": 30, "fixture_positive": 30})
            self.assertTrue(manifest["formal_acceptance"]["passed"])
            self.assertEqual(manifest["formal_acceptance"]["conclusion_level"], "E2")

    def test_candidate_json_with_utf8_bom_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            candidate_path = temp_path / "candidate.json"
            config_path = temp_path / "config.yaml"
            out_dir = temp_path / "out"
            candidate_path.write_text(json.dumps(_skill_dict(status="candidate")), encoding="utf-8-sig")
            config_path.write_text(yaml.safe_dump(_config("fixture_positive")), encoding="utf-8-sig")

            manifest = run_skill_verification(candidate_path, config_path, out_dir)

            self.assertEqual(manifest["completion_status"], "completed")
            self.assertEqual(manifest["decision"], "verified")

    def test_rollout_route_skill_evaluator_writes_real_episode_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            candidate_path = temp_path / "candidate.json"
            config_path = temp_path / "config.yaml"
            out_dir = temp_path / "out"
            candidate_path.write_text(json.dumps(_rollout_skill_dict()), encoding="utf-8")
            config = _config("rollout_route_skill")
            config["verification"].update(
                {
                    "mode": "pilot",
                    "paired_seeds": [31, 32],
                    "min_paired_samples": 2,
                    "episode_max_steps": 20,
                    "estimator_values": {"future_route_uses": 5, "road_break_even_uses": 3},
                    "env_config": {
                        "env": {
                            "grid_size": [8, 8],
                            "max_steps": 20,
                            "base_pos": [1, 1],
                            "ore_positions": [[6, 6]],
                            "rough_terrain": [[1, 2], [1, 3], [2, 3], [3, 3]],
                            "obstacles": [],
                            "observation": {"mode": "partial_obs", "local_view_radius": 3},
                            "shaping": {"allow_build_road": True, "allow_dig": True},
                        }
                    },
                }
            )
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            manifest = run_skill_verification(candidate_path, config_path, out_dir)

            report = json.loads((out_dir / manifest["report"]).read_text(encoding="utf-8"))
            self.assertEqual(report["sample_size"], 2)
            self.assertIn("disabled", report["metrics"])
            self.assertIn("enabled", report["metrics"])
            self.assertIn("num_build_road", report["metrics"]["enabled"][0])
            self.assertTrue((out_dir / "evaluator_registry" / "candidates" / "rollout_road_candidate" / "1.0.0.json").exists())


def _config(evaluator: str) -> dict:
    return {
        "verification": {
            "schema_version": 1,
            "protocol_id": "skill_verification_v1",
            "mode": "smoke",
            "primary_metric": "road_net_payoff",
            "direction": "maximize",
            "min_effect": 0.0,
            "min_success_rate": 0.6,
            "max_false_trigger_rate": 0.1,
            "min_paired_samples": 5,
            "bootstrap_iterations": 100,
            "bootstrap_seed": 0,
            "paired_seeds": [1, 2, 3, 4, 5],
            "environment_strata": ["fixture"],
            "source_train_seeds": [],
            "evaluator": evaluator,
        }
    }


def _rollout_skill_dict() -> dict:
    data = _skill_dict(status="candidate")
    data.update(
        {
            "skill_id": "rollout_road_candidate",
            "name": "Rollout road candidate",
            "applicability": {"all": [{"feature": "current.terrain_band", "op": "in", "value": ["ROUGH", "VERY_ROUGH"]}]},
        }
    )
    return data


if __name__ == "__main__":
    unittest.main()
