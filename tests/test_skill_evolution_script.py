from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_skill_evolution_experiment import GROUPS, run_skill_evolution_experiment


class SkillEvolutionScriptTest(unittest.TestCase):
    def test_smoke_run_writes_required_manifest_and_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "m5_smoke"
            manifest = run_skill_evolution_experiment("configs/curriculum_self_evolution.yaml", out_dir)

            self.assertEqual(manifest["completion_status"], "completed")
            self.assertTrue(manifest["mock_smoke"])
            self.assertEqual(manifest["groups"], GROUPS)
            self.assertIn("train", manifest["seed_partitions"]["partitions"])
            self.assertIn("test", manifest["seed_partitions"]["partitions"])
            self.assertIn("self_proposed_candidate", manifest["candidate_verified_split"])
            self.assertEqual(
                manifest["candidate_verified_split"]["self_proposed_candidate"]["verified_skill_ids"],
                [],
            )
            self.assertTrue((out_dir / "config_resolved.yaml").exists())
            self.assertTrue((out_dir / "maps" / "map_manifest.jsonl").exists())
            self.assertTrue((out_dir / "episodes" / "metrics.csv").exists())
            self.assertTrue((out_dir / "prompts" / "prompt_manifest.jsonl").exists())
            self.assertTrue((out_dir / "skills" / "skill_trace.jsonl").exists())
            self.assertTrue((out_dir / "capability" / "capability_summary.json").exists())
            self.assertTrue((out_dir / "capability" / "capability_matrix.csv").exists())
            self.assertTrue((out_dir / "capability" / "checkpoints.csv").exists())

            saved = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["config_hash"], manifest["config_hash"])
            self.assertIn("capability/checkpoints.csv", saved["output_file_checksums"])

    def test_resume_rejects_config_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "m5_smoke"
            run_skill_evolution_experiment("configs/curriculum_self_evolution.yaml", out_dir)
            other_config = Path(temp_dir) / "other.yaml"
            other_config.write_text("experiment:\n  root_seed: 9\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "config hash"):
                run_skill_evolution_experiment(other_config, out_dir)


if __name__ == "__main__":
    unittest.main()
