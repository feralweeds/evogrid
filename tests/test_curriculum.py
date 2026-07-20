from __future__ import annotations

import unittest

import yaml

from evogrid.curriculum import CurriculumConfig, FixedScheduleCurriculum


class CurriculumTest(unittest.TestCase):
    def test_config_loads_explicit_environment_family(self):
        config = CurriculumConfig.from_dict(_config_data())

        self.assertEqual(config.allowed_partitions, ("train", "gate"))
        self.assertEqual(config.stages[0].stage_id, "open_small")
        self.assertIn("p_open", config.stages[0].env_family)
        self.assertEqual(config.stages[0].promotion_rule.consecutive_windows, 2)

    def test_test_partition_is_rejected(self):
        data = _config_data()
        data["allowed_partitions"] = ["train", "gate", "test"]

        with self.assertRaisesRegex(ValueError, "test partition"):
            CurriculumConfig.from_dict(data)

    def test_promotes_only_after_consecutive_gate_windows(self):
        curriculum = FixedScheduleCurriculum(CurriculumConfig.from_dict(_config_data()))

        first = curriculum.record_gate_result(0.7, window_id="w0")
        self.assertTrue(first["passed"])
        self.assertFalse(curriculum.should_promote())
        self.assertIsNone(curriculum.promote_if_ready())

        curriculum.record_gate_result(0.8, window_id="w1")
        event = curriculum.promote_if_ready()

        self.assertIsNotNone(event)
        self.assertEqual(event["from_stage"], "open_small")
        self.assertEqual(event["to_stage"], "mixed_medium")
        self.assertEqual(curriculum.current_stage().stage_id, "mixed_medium")
        self.assertEqual(curriculum.replayable_events()[0]["reason_code"], "gate_passed")

    def test_gate_uses_frozen_benchmark_id(self):
        curriculum = FixedScheduleCurriculum(CurriculumConfig.from_dict(_config_data()))

        with self.assertRaisesRegex(ValueError, "benchmark mismatch"):
            curriculum.record_gate_result(0.9, benchmark_id="test_benchmark")


def _config_data():
    return yaml.safe_load(
        """
curriculum_id: test_curriculum
controller_version: fixed_schedule_test
allowed_partitions: [train, gate]
stages:
  - stage_id: open_small
    env_family:
      sizes: [[16, 16]]
      p_open: [0.75]
      topology_hurst: [0.5]
      terrain_hurst: [0.2, 0.5]
    train_budget_episodes: 10
    gate_benchmark_id: open_small_gate_v1
    promotion_rule:
      capability_score_gte: 0.7
      consecutive_windows: 2
  - stage_id: mixed_medium
    env_family:
      sizes: [[32, 32]]
      p_open: [0.65]
      topology_hurst: [0.6]
      terrain_hurst: [0.5]
    train_budget_episodes: 20
    gate_benchmark_id: mixed_medium_gate_v1
    promotion_rule:
      capability_score_gte: 0.75
      consecutive_windows: 2
"""
    )


if __name__ == "__main__":
    unittest.main()
