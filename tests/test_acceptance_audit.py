from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from evogrid.evaluation.acceptance import audit_run_directory, infer_conclusion_level
from scripts.run_chunk_world_smoke import run_chunk_world_smoke
from scripts.run_curriculum_ablation import run_curriculum_ablation
from scripts.run_skill_evolution_experiment import run_skill_evolution_experiment
from scripts.run_skill_proposal import run_skill_proposal


class AcceptanceAuditTest(unittest.TestCase):
    def test_audits_generated_smoke_runs_as_e0(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runs = [
                root / "m5_skill",
                root / "m6_curriculum",
                root / "m7_chunk",
            ]
            run_skill_evolution_experiment("configs/curriculum_self_evolution.yaml", runs[0])
            run_curriculum_ablation("configs/curriculum_ablation.yaml", runs[1])
            run_chunk_world_smoke("configs/chunk_world_smoke.yaml", runs[2])

            reports = [audit_run_directory(run_dir) for run_dir in runs]

        self.assertTrue(all(report.passed for report in reports))
        self.assertEqual([report.conclusion_level for report in reports], ["E0", "E0", "E0"])

    def test_missing_csv_schema_is_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "bad_run"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "bad_run",
                        "experiment_type": "skill_evolution",
                        "completion_status": "completed",
                        "mode": "mock_smoke",
                        "mock_smoke": True,
                    }
                ),
                encoding="utf-8",
            )
            with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["value"])
                writer.writeheader()
                writer.writerow({"value": 1})

            report = audit_run_directory(run_dir)

        self.assertFalse(report.passed)
        self.assertTrue(any("CSV output missing schema_version" in issue.message for issue in report.issues))

    def test_json_with_utf8_bom_is_audited(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "bom_run"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "bom_run",
                        "experiment_type": "skill_evolution_smoke",
                        "completion_status": "completed",
                        "mode": "smoke",
                        "mock_smoke": True,
                    }
                ),
                encoding="utf-8-sig",
            )
            (run_dir / "candidate.json").write_text(json.dumps({"schema_version": 1}), encoding="utf-8-sig")

            report = audit_run_directory(run_dir)

        self.assertTrue(report.passed)

    def test_chunk_guardrail_flags_failed_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "bad_chunk"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "bad_chunk",
                        "experiment_type": "chunk_world_smoke",
                        "completion_status": "completed",
                        "mode": "mock_smoke",
                        "mock_smoke": True,
                        "order_independent": True,
                        "boundary_metrics": {
                            "east_west_halo_max_abs_diff": 0.1,
                            "north_south_halo_max_abs_diff": 0.0,
                        },
                        "event_survived_reload": True,
                    }
                ),
                encoding="utf-8",
            )

            report = audit_run_directory(run_dir)

        self.assertFalse(report.passed)
        self.assertTrue(any("east/west" in issue.message for issue in report.issues))

    def test_skill_proposal_pilot_audits_candidate_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "skill_proposal"
            run_skill_proposal("configs/skill_proposal_pilot.yaml", run_dir)

            report = audit_run_directory(run_dir)

        self.assertTrue(report.passed)
        self.assertEqual(report.conclusion_level, "E0")

    def test_skill_proposal_cannot_write_verified_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "bad_skill_proposal"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "bad_skill_proposal",
                        "experiment_type": "skill_proposal",
                        "completion_status": "completed",
                        "mode": "pilot",
                        "mock_smoke": True,
                        "accepted_candidate_count": 0,
                        "rejected_proposal_count": 0,
                        "outputs": {"proposal_manifest": "proposal_manifest.json"},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "proposal_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "accepted": [],
                        "rejected": [],
                        "backend": "fixture_valid",
                        "source_partition": "train",
                        "verification_started": False,
                        "verified_written": True,
                    }
                ),
                encoding="utf-8",
            )

            report = audit_run_directory(run_dir)

        self.assertFalse(report.passed)
        self.assertTrue(any("must not write verified skills" in issue.message for issue in report.issues))

    def test_skill_proposal_candidate_schema_is_revalidated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "bad_skill_proposal_schema"
            candidate_dir = run_dir / "skills" / "candidates" / "bad_skill"
            candidate_dir.mkdir(parents=True)
            candidate_path = candidate_dir / "1.0.0.json"
            spec = {
                "schema_version": 1,
                "skill_id": "bad_skill",
                "version": "1.0.0",
                "status": "candidate",
                "name": "Bad skill",
                "description": "Invalid predicate feature.",
                "problem_addressed": "Schema hardening",
                "source": {"proposer": "llm", "source_episode_ids": ["train/ep/1"]},
                "applicability": {"all": [{"feature": "current.road_exists", "op": "eq", "value": False}]},
                "procedure": [{"op": "ACT", "action": "BUILD_ROAD"}],
                "budget": {"max_runtime_steps": 1, "max_environment_actions": 1, "max_nested_skill_depth": 0},
                "objective": {"primary_metric": "road_net_payoff", "direction": "maximize"},
                "dependencies": [],
                "spec_hash": "fake",
            }
            candidate_path.write_text(
                json.dumps({"schema_version": 1, "spec": spec, "storage_status": "candidate"}),
                encoding="utf-8",
            )
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "bad_skill_proposal_schema",
                        "experiment_type": "skill_proposal",
                        "completion_status": "completed",
                        "mode": "pilot",
                        "mock_smoke": True,
                        "accepted_candidate_count": 1,
                        "rejected_proposal_count": 0,
                        "outputs": {"proposal_manifest": "proposal_manifest.json"},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "proposal_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "accepted": [
                            {
                                "skill_id": "bad_skill",
                                "version": "1.0.0",
                                "status": "candidate",
                                "spec_hash": "fake",
                                "path": "skills/candidates/bad_skill/1.0.0.json",
                            }
                        ],
                        "rejected": [],
                        "backend": "deepseek",
                        "backend_metadata": {
                            "provider": "deepseek",
                            "model": "deepseek-chat",
                            "base_url": "https://api.deepseek.com",
                            "prompt_hash": "hash",
                            "response_received": True,
                            "api_key_env": "DEEPSEEK_API_KEY",
                        },
                        "source_partition": "train",
                        "verification_started": False,
                        "verified_written": False,
                    }
                ),
                encoding="utf-8",
            )

            report = audit_run_directory(run_dir)

        self.assertFalse(report.passed)
        self.assertTrue(any("proposal candidate schema invalid" in issue.message for issue in report.issues))

    def test_non_smoke_type_does_not_raise_conclusion_without_formal_acceptance(self):
        manifest = {
            "schema_version": 1,
            "run_id": "formalish",
            "experiment_type": "skill_evolution",
            "completion_status": "completed",
            "mode": "pilot",
            "mock_smoke": False,
        }

        self.assertEqual(infer_conclusion_level(manifest), "E0")

    def test_formal_acceptance_can_raise_conclusion_level(self):
        manifest = {
            "schema_version": 1,
            "run_id": "formal",
            "experiment_type": "skill_evolution",
            "completion_status": "completed",
            "mode": "formal",
            "mock_smoke": False,
            "formal_acceptance": {"passed": True, "conclusion_level": "E4"},
        }

        self.assertEqual(infer_conclusion_level(manifest), "E4")

    def test_map_calibration_formal_claim_requires_passing_gate_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "bad_formal_map"
            maps_dir = run_dir / "maps"
            maps_dir.mkdir(parents=True)
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "bad_formal_map",
                        "experiment_type": "map_calibration",
                        "completion_status": "completed",
                        "mode": "formal",
                        "mock_smoke": False,
                        "formal_acceptance": {
                            "passed": True,
                            "conclusion_level": "E1",
                            "gate_report": "maps/calibration_gates.json",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (maps_dir / "calibration_gates.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol_id": "map_calibration_A0_A5_v1",
                        "passed": False,
                        "gates": [
                            {"gate_id": "A0", "name": "completeness", "passed": False, "details": {}},
                            {"gate_id": "A1", "name": "reproducibility", "passed": True, "details": {}},
                            {"gate_id": "A2", "name": "p_open_control", "passed": True, "details": {}},
                            {"gate_id": "A3", "name": "hurst_separability", "passed": True, "details": {}},
                            {"gate_id": "A4", "name": "axis_anisotropy", "passed": True, "details": {}},
                            {"gate_id": "A5", "name": "critical_curve_estimation", "passed": True, "details": {}},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = audit_run_directory(run_dir)

        self.assertFalse(report.passed)
        self.assertEqual(report.conclusion_level, "E0")
        self.assertFalse(report.formal_acceptance["validated"])
        self.assertEqual(report.formal_acceptance["failed_gates"], ["A0"])
        self.assertTrue(any("formal gates did not all pass" in issue.message for issue in report.issues))

    def test_skill_verification_formal_claim_requires_g0_g5_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "bad_formal_skill"
            report_dir = run_dir / "skills" / "reports" / "s1" / "1.0.0"
            report_dir.mkdir(parents=True)
            report_ref = "skills/reports/s1/1.0.0/verify_s1.json"
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "bad_formal_skill",
                        "experiment_type": "skill_verification",
                        "completion_status": "completed",
                        "mode": "formal",
                        "mock_smoke": False,
                        "formal_acceptance": {
                            "passed": True,
                            "conclusion_level": "E2",
                            "gate_report": report_ref,
                        },
                    }
                ),
                encoding="utf-8",
            )
            gates = [
                {"gate": "G0_data_integrity", "passed": True},
                {"gate": "G1_effect", "passed": True},
                {"gate": "G2_reliability", "passed": True},
                {"gate": "G3_negative_safety", "passed": True},
                {"gate": "G4_transfer", "passed": True},
                {"gate": "G5_non_redundancy", "passed": False},
            ]
            (report_dir / "verify_s1.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol_id": "skill_verification_v1",
                        "decision": "revision_required",
                        "failure_reasons": ["G5_non_redundancy"],
                        "gates": gates,
                    }
                ),
                encoding="utf-8",
            )

            report = audit_run_directory(run_dir)

        self.assertFalse(report.passed)
        self.assertEqual(report.conclusion_level, "E0")
        self.assertFalse(report.formal_acceptance["validated"])
        self.assertEqual(report.formal_acceptance["failed_gates"], ["G5_non_redundancy"])
        self.assertTrue(any("formal gate_report decision is not verified" in issue.message for issue in report.issues))

    def test_formal_acceptance_requires_readiness_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "bad_readiness_skill"
            report_dir = run_dir / "skills" / "reports" / "s1" / "1.0.0"
            report_dir.mkdir(parents=True)
            report_ref = "skills/reports/s1/1.0.0/verify_s1.json"
            gates = [
                {"gate": "G0_data_integrity", "passed": True},
                {"gate": "G1_effect", "passed": True},
                {"gate": "G2_reliability", "passed": True},
                {"gate": "G3_negative_safety", "passed": True},
                {"gate": "G4_transfer", "passed": True},
                {"gate": "G5_non_redundancy", "passed": True},
            ]
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "bad_readiness_skill",
                        "experiment_type": "skill_verification",
                        "completion_status": "completed",
                        "mode": "formal",
                        "mock_smoke": False,
                        "formal_readiness": {
                            "schema_version": 1,
                            "passed": False,
                            "failures": ["formal stratum sample size below 30 paired seeds"],
                        },
                        "formal_acceptance": {
                            "passed": True,
                            "conclusion_level": "E2",
                            "gate_report": report_ref,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (report_dir / "verify_s1.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol_id": "skill_verification_v1",
                        "decision": "verified",
                        "failure_reasons": [],
                        "gates": gates,
                    }
                ),
                encoding="utf-8",
            )

            report = audit_run_directory(run_dir)

        self.assertFalse(report.passed)
        self.assertTrue(any("readiness" in issue.message for issue in report.issues))


if __name__ == "__main__":
    unittest.main()
