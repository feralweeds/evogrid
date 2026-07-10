from __future__ import annotations

import unittest

from evogrid.evaluation.compare_groups import summarize_group
from evogrid.evaluation.metrics_schema import EVAL_COLUMNS, standardize_eval_row


class EvaluationSchemaTest(unittest.TestCase):
    def test_standardize_eval_row_has_all_columns(self):
        row = standardize_eval_row(
            {
                "group": "full_shaping",
                "seed": 0,
                "episode": 1,
                "episode_reward": 1.5,
                "ore_delivered": 2,
            }
        )
        self.assertEqual(list(row.keys()), EVAL_COLUMNS)
        self.assertEqual(row["policy_type"], "")
        self.assertEqual(row["ore_delivered"], 2)
        self.assertEqual(row["num_dig"], 0)

    def test_summarize_group_nested_metrics(self):
        summary = summarize_group(
            [
                {"seed": 0, "episode_reward": 1.0, "ore_delivered": 1},
                {"seed": 1, "episode_reward": 3.0, "ore_delivered": 3},
            ]
        )
        self.assertEqual(summary["episode_count"], 2)
        self.assertEqual(summary["seeds"], [0, 1])
        self.assertEqual(summary["metrics"]["episode_reward"]["mean"], 2.0)
        self.assertEqual(summary["metrics"]["ore_delivered"]["max"], 3.0)


if __name__ == "__main__":
    unittest.main()

