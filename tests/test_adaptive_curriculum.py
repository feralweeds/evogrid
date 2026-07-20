from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from evogrid.curriculum import AdaptiveCurriculumController, AdaptiveEvidence, AdaptiveParameterRule


class AdaptiveCurriculumTest(unittest.TestCase):
    def test_rejects_test_partition_evidence(self):
        with self.assertRaisesRegex(ValueError, "cannot read partitions"):
            AdaptiveEvidence.from_dict(
                {
                    "train_score": 0.7,
                    "gate_score": 0.7,
                    "observed_partitions": ["train", "test"],
                }
            )

    def test_too_easy_increases_difficulty_with_delta_and_bounds(self):
        controller = AdaptiveCurriculumController(root_seed=11, target_min=0.4, target_max=0.7)
        event = controller.decide(
            "open_small",
            {"p_open": [0.46], "topology_hurst": [0.88], "terrain_hurst": [0.86]},
            {"train_score": 0.9, "gate_score": 0.82, "observed_partitions": ["train", "gate"]},
        )

        self.assertEqual(event["reason_code"], "too_easy")
        self.assertEqual(event["to_parameters"]["p_open"], [0.45])
        self.assertEqual(event["to_parameters"]["topology_hurst"], [0.9])
        self.assertEqual(event["to_parameters"]["terrain_hurst"], [0.9])
        self.assertEqual(event["observable_evidence"]["observed_partitions"], ["train", "gate"])

    def test_too_hard_decreases_difficulty_with_delta_and_bounds(self):
        controller = AdaptiveCurriculumController(root_seed=11, target_min=0.4, target_max=0.7)
        event = controller.decide(
            "rough_large",
            {"p_open": [0.83], "topology_hurst": [0.22], "terrain_hurst": [0.12]},
            AdaptiveEvidence(train_score=0.3, gate_score=0.2),
        )

        self.assertEqual(event["reason_code"], "too_hard")
        self.assertEqual(event["to_parameters"]["p_open"], [0.85])
        self.assertEqual(event["to_parameters"]["topology_hurst"], [0.2])
        self.assertEqual(event["to_parameters"]["terrain_hurst"], [0.1])

    def test_learning_frontier_keeps_parameters_clamped_but_not_pushed(self):
        controller = AdaptiveCurriculumController(root_seed=11, target_min=0.4, target_max=0.7)
        event = controller.decide(
            "mixed_medium",
            {"p_open": [0.6], "topology_hurst": [0.5], "terrain_hurst": [0.4]},
            {"train_score": 0.6, "gate_score": 0.55},
        )

        self.assertEqual(event["reason_code"], "learning_frontier")
        self.assertEqual(event["to_parameters"]["p_open"], [0.6])
        self.assertEqual(event["to_parameters"]["topology_hurst"], [0.5])
        self.assertEqual(event["to_parameters"]["terrain_hurst"], [0.4])

    def test_decision_seed_is_replayable_for_fixed_seed_and_index(self):
        first = AdaptiveCurriculumController(root_seed=99).decide(
            "stage",
            {"p_open": [0.7]},
            {"train_score": 0.9, "gate_score": 0.9},
            decision_index=3,
        )
        second = AdaptiveCurriculumController(root_seed=99).decide(
            "stage",
            {"p_open": [0.7]},
            {"train_score": 0.9, "gate_score": 0.9},
            decision_index=3,
        )

        self.assertEqual(first["decision_seed"], second["decision_seed"])
        self.assertEqual(first["to_parameters"], second["to_parameters"])

    def test_custom_rule_and_jsonl_event_output(self):
        controller = AdaptiveCurriculumController(
            root_seed=7,
            parameter_rules={
                "difficulty": AdaptiveParameterRule(
                    name="difficulty",
                    min_value=0.0,
                    max_value=1.0,
                    max_delta=0.2,
                    harder_delta_sign=1,
                )
            },
        )
        controller.decide("custom", {"difficulty": 0.5}, {"train_score": 0.9, "gate_score": 0.9})

        with tempfile.TemporaryDirectory() as temp_dir:
            path = controller.write_events_jsonl(Path(temp_dir) / "curriculum_events.jsonl")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(rows[0]["reason_code"], "too_easy")
        self.assertEqual(rows[0]["to_parameters"]["difficulty"], 0.7)


if __name__ == "__main__":
    unittest.main()
