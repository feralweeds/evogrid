from __future__ import annotations

import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path
from uuid import uuid4


class GeneralizationEvalTest(unittest.TestCase):
    def test_generalization_eval_smoke_outputs_train_test_comparison(self):
        out_dir = Path("outputs") / f"test_generalization_eval_{uuid4().hex}"
        command = [
            sys.executable,
            "scripts/run_generalization_eval.py",
            "--env-config",
            "configs/env_random_curriculum.yaml",
            "--train-seeds",
            "0:1",
            "--test-seeds",
            "1000:1001",
            "--episodes-per-seed",
            "1",
            "--max-steps",
            "30",
            "--groups",
            "route_only",
            "rough_rule_road",
            "exploration_road",
            "--out",
            str(out_dir),
        ]

        completed = subprocess.run(command, check=False, capture_output=True, text=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertIn("route_only", summary["groups"])
        self.assertIn("train", summary["groups"]["route_only"])
        self.assertIn("test", summary["groups"]["route_only"])
        self.assertTrue((out_dir / "metrics.csv").exists())
        self.assertTrue((out_dir / "group_comparison.csv").exists())

        with (out_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["phase"] for row in rows}, {"train", "test"})
        self.assertEqual({row["group"] for row in rows}, {"route_only", "rough_rule_road", "exploration_road"})

    def test_generalization_eval_mock_llm_outputs_llm_metrics(self):
        out_dir = Path("outputs") / f"test_generalization_eval_llm_{uuid4().hex}"
        command = [
            sys.executable,
            "scripts/run_generalization_eval.py",
            "--env-config",
            "configs/env_random_curriculum.yaml",
            "--train-seeds",
            "0:1",
            "--test-seeds",
            "1000:1001",
            "--episodes-per-seed",
            "1",
            "--max-steps",
            "30",
            "--groups",
            "llm_no_road_learning",
            "llm_with_road_learning",
            "--mock-deepseek",
            "--quiet-llm-calls",
            "--out",
            str(out_dir),
        ]

        completed = subprocess.run(command, check=False, capture_output=True, text=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["preflight"]["reason"], "mock LLM enabled")
        with (out_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({row["group"] for row in rows}, {"llm_no_road_learning", "llm_with_road_learning"})
        self.assertIn("llm_decision_count", rows[0])
        self.assertIn("p_llm_build_given_learned_positive", rows[0])

    def test_generalization_eval_supports_learned_only_test_budget(self):
        out_dir = Path("outputs") / f"test_generalization_eval_learned_only_{uuid4().hex}"
        command = [
            sys.executable,
            "scripts/run_generalization_eval.py",
            "--env-config",
            "configs/env_random_curriculum.yaml",
            "--train-seeds",
            "0:1",
            "--test-seeds",
            "1000:1001",
            "--episodes-per-seed",
            "1",
            "--max-steps",
            "30",
            "--groups",
            "llm_with_road_learning_medium_threshold",
            "--mock-deepseek",
            "--quiet-llm-calls",
            "--test-exploration-budget",
            "0",
            "--out",
            str(out_dir),
        ]

        completed = subprocess.run(command, check=False, capture_output=True, text=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["config"]["test_exploration_budget"], 0)
        with (out_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertIn("llm_strong_learned_evidence_count", rows[0])
        self.assertIn("llm_skip_given_strong_evidence_count", rows[0])

    def test_generalization_eval_supports_split_train_test_episode_counts(self):
        out_dir = Path("outputs") / f"test_generalization_eval_split_episodes_{uuid4().hex}"
        command = [
            sys.executable,
            "scripts/run_generalization_eval.py",
            "--env-config",
            "configs/env_random_curriculum.yaml",
            "--train-seeds",
            "0:1",
            "--test-seeds",
            "1000:1001",
            "--episodes-per-seed",
            "1",
            "--train-episodes-per-seed",
            "2",
            "--test-episodes-per-seed",
            "1",
            "--max-steps",
            "20",
            "--groups",
            "route_only",
            "--out",
            str(out_dir),
        ]

        completed = subprocess.run(command, check=False, capture_output=True, text=True)

        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["config"]["train_episodes_per_seed"], 2)
        self.assertEqual(summary["config"]["test_episodes_per_seed"], 1)
        with (out_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len([row for row in rows if row["phase"] == "train"]), 2)
        self.assertEqual(len([row for row in rows if row["phase"] == "test"]), 1)


if __name__ == "__main__":
    unittest.main()
