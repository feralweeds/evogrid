from __future__ import annotations

import unittest

from evogrid.evaluation.fdr import benjamini_hochberg
from evogrid.evaluation.skill_verifier import SkillVerifier, SkillVerificationProtocol
from evogrid.skills.schemas import SkillSpec
from tests.test_skill_schema import _skill_dict


class SkillVerifierTest(unittest.TestCase):
    def test_positive_paired_effect_verifies_candidate(self):
        candidate = SkillSpec.from_dict(_skill_dict(status="candidate"))
        verifier = SkillVerifier(
            SkillVerificationProtocol(min_effect=0.1, min_success_rate=0.6, max_false_trigger_rate=0.1)
        )

        report = verifier.verify(candidate, [1, 2, 3], _positive_evaluator)

        self.assertEqual(report.decision, "verified")
        self.assertEqual(report.sample_size, 3)
        self.assertGreater(report.metrics["paired_delta_mean"], 0.1)
        self.assertIn("paired_delta_bootstrap_ci", report.metrics)
        self.assertIn("G5_non_redundancy", [gate["gate"] for gate in report.gates])

    def test_negative_effect_rejects_candidate(self):
        candidate = SkillSpec.from_dict(_skill_dict(status="candidate"))

        report = SkillVerifier().verify(candidate, [1, 2, 3], _negative_evaluator)

        self.assertEqual(report.decision, "rejected")
        self.assertIn("G1_effect", report.failure_reasons)

    def test_false_trigger_failure_requires_revision(self):
        candidate = SkillSpec.from_dict(_skill_dict(status="candidate"))

        report = SkillVerifier().verify(candidate, [1, 2, 3], _unsafe_evaluator)

        self.assertEqual(report.decision, "revision_required")
        self.assertIn("G3_negative_safety", report.failure_reasons)

    def test_seed_overlap_invalidates_verification(self):
        candidate = SkillSpec.from_dict(_skill_dict(status="candidate"))

        report = SkillVerifier().verify(candidate, [1, 2, 3], _positive_evaluator, source_train_seeds={3, 4})

        self.assertEqual(report.decision, "verification_invalid")
        self.assertIn("verify seed leakage", report.failure_reasons[0])

    def test_min_sample_failure_invalidates_verification(self):
        candidate = SkillSpec.from_dict(_skill_dict(status="candidate"))
        verifier = SkillVerifier(SkillVerificationProtocol(min_paired_samples=5))

        report = verifier.verify(candidate, [1, 2, 3], _positive_evaluator)

        self.assertEqual(report.decision, "verification_invalid")
        self.assertIn("G0_data_integrity", report.failure_reasons)

    def test_bootstrap_ci_must_clear_zero(self):
        candidate = SkillSpec.from_dict(_skill_dict(status="candidate"))
        verifier = SkillVerifier(
            SkillVerificationProtocol(
                min_effect=0.0,
                min_success_rate=0.5,
                bootstrap_iterations=200,
                bootstrap_seed=7,
            )
        )

        report = verifier.verify(candidate, [1, 2, 3, 4], _mixed_evaluator)

        self.assertEqual(report.decision, "rejected")
        effect_gate = next(gate for gate in report.gates if gate["gate"] == "G1_effect")
        self.assertLessEqual(effect_gate["bootstrap_ci"]["low"], 0.0)

    def test_transfer_gate_requires_consistent_unseen_strata(self):
        candidate = SkillSpec.from_dict(_skill_dict(status="candidate"))
        verifier = SkillVerifier(
            SkillVerificationProtocol(
                requires_transfer=True,
                min_transfer_strata=2,
                min_success_rate=0.5,
                bootstrap_iterations=0,
            )
        )

        report = verifier.verify(candidate, [1, 2, 3, 4], _one_bad_stratum_evaluator)

        self.assertEqual(report.decision, "revision_required")
        self.assertIn("G4_transfer", report.failure_reasons)

    def test_redundancy_gate_requires_revision(self):
        candidate = SkillSpec.from_dict(_skill_dict(status="candidate"))
        verifier = SkillVerifier(SkillVerificationProtocol(max_redundancy_score=0.5))

        report = verifier.verify(candidate, [1, 2, 3], _redundant_evaluator)

        self.assertEqual(report.decision, "revision_required")
        self.assertIn("G5_non_redundancy", report.failure_reasons)

    def test_benjamini_hochberg_fdr_marks_prefix(self):
        decisions = benjamini_hochberg({"a": 0.001, "b": 0.02, "c": 0.2}, alpha=0.05)

        by_id = {decision.candidate_id: decision for decision in decisions}
        self.assertTrue(by_id["a"].rejected_null)
        self.assertTrue(by_id["b"].rejected_null)
        self.assertFalse(by_id["c"].rejected_null)


def _positive_evaluator(seed: int, enabled: bool) -> dict:
    return {
        "road_net_payoff": float(seed) + (1.0 if enabled else 0.0),
        "false_trigger_rate": 0.0,
        "runtime_failure_rate": 0.0,
        "invalid_action_rate": 0.0,
        "activation_rate": 1.0,
        "redundancy_score": 0.0,
    }


def _negative_evaluator(seed: int, enabled: bool) -> dict:
    row = _positive_evaluator(seed, False)
    row["road_net_payoff"] = float(seed) - (1.0 if enabled else 0.0)
    return row


def _unsafe_evaluator(seed: int, enabled: bool) -> dict:
    row = _positive_evaluator(seed, enabled)
    row["false_trigger_rate"] = 0.5 if enabled else 0.0
    return row


def _mixed_evaluator(seed: int, enabled: bool) -> dict:
    row = _positive_evaluator(seed, False)
    delta = 1.0 if seed % 2 else -0.9
    row["road_net_payoff"] = float(seed) + (delta if enabled else 0.0)
    return row


def _one_bad_stratum_evaluator(seed: int, enabled: bool) -> dict:
    row = _positive_evaluator(seed, False)
    row["road_net_payoff"] = float(seed) + (1.0 if enabled else 0.0)
    row["stratum"] = "only_seen_transfer_stratum"
    return row


def _redundant_evaluator(seed: int, enabled: bool) -> dict:
    row = _positive_evaluator(seed, enabled)
    row["redundancy_score"] = 0.9
    return row


if __name__ == "__main__":
    unittest.main()
