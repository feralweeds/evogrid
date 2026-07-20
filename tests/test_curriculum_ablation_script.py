from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_curriculum_ablation import CURRICULUM_GROUPS, run_curriculum_ablation


class CurriculumAblationScriptTest(unittest.TestCase):
    def test_ablation_writes_test_partition_outputs_and_guardrail(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "m6_ablation"
            manifest = run_curriculum_ablation("configs/curriculum_ablation.yaml", out_dir)

            self.assertEqual(manifest["completion_status"], "completed")
            self.assertEqual(manifest["groups"], CURRICULUM_GROUPS)
            self.assertEqual(manifest["independent_test_partition"], "test")
            self.assertEqual(manifest["conclusion_strength"], "smoke_only_no_scientific_claim")
            self.assertTrue((out_dir / "capability" / "curriculum_ablation.csv").exists())
            self.assertTrue((out_dir / "capability" / "curriculum_ablation_summary.json").exists())
            self.assertTrue((out_dir / "curriculum_events.jsonl").exists())

            summary = json.loads(
                (out_dir / "capability" / "curriculum_ablation_summary.json").read_text(encoding="utf-8")
            )
            self.assertIn("adaptive_over_fixed_test_delta", summary)
            self.assertIn("claim_guardrail", summary)
            events = [
                json.loads(line)
                for line in (out_dir / "curriculum_events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [event["reason_code"] for event in events],
                ["too_easy", "learning_frontier", "too_hard"],
            )
            for event in events:
                self.assertEqual(event["observable_evidence"]["observed_partitions"], ["train", "gate"])


if __name__ == "__main__":
    unittest.main()
