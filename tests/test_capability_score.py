from __future__ import annotations

import unittest

from evogrid.evaluation.capability import CapabilityTask, compute_capability


class CapabilityScoreTest(unittest.TestCase):
    def test_capability_score_keeps_count_and_score_separate(self):
        result = compute_capability(
            _tasks(),
            benchmark_results={
                "T1": {"success": 0.75},
                "T2": {"cost": 2.0},
            },
            verified_skill_ids=["road_skill"],
            skill_task_effects={"road_skill": {"T1": 0.1, "T2": 0.4}},
        )

        self.assertEqual(result.verified_skill_count, 1)
        self.assertAlmostEqual(result.capability_vector["T1"], 0.75)
        self.assertAlmostEqual(result.capability_vector["T2"], 0.6)
        self.assertAlmostEqual(result.capability_score, (0.75 + 0.6) / 2)
        self.assertEqual(result.skill_coverage_matrix[0]["skill_id"], "road_skill")

    def test_missing_task_result_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing benchmark"):
            compute_capability(_tasks(), {"T1": {"success": 0.5}}, [], {})

    def test_missing_skill_coverage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing skill coverage"):
            compute_capability(
                _tasks(),
                {"T1": {"success": 0.5}, "T2": {"cost": 3.0}},
                ["skill"],
                {"skill": {"T1": 0.1}},
            )

    def test_retention_score_requires_all_tasks(self):
        with self.assertRaisesRegex(ValueError, "missing retention"):
            compute_capability(
                _tasks(),
                {"T1": {"success": 0.5}, "T2": {"cost": 3.0}},
                [],
                {},
                retention_results={"T1": 0.8},
            )


def _tasks() -> list[CapabilityTask]:
    return [
        CapabilityTask("T1", "success", "maximize", floor=0.0, reference=1.0, weight=1.0),
        CapabilityTask("T2", "cost", "minimize", floor=5.0, reference=0.0, weight=1.0),
    ]


if __name__ == "__main__":
    unittest.main()
