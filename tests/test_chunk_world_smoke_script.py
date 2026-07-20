from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.run_chunk_world_smoke import run_chunk_world_smoke


class ChunkWorldSmokeScriptTest(unittest.TestCase):
    def test_smoke_run_checks_m7_contracts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "m7_chunk_smoke"
            manifest = run_chunk_world_smoke("configs/chunk_world_smoke.yaml", out_dir)

            self.assertEqual(manifest["completion_status"], "completed")
            self.assertTrue(manifest["order_independent"])
            self.assertEqual(manifest["boundary_metrics"]["east_west_halo_max_abs_diff"], 0.0)
            self.assertEqual(manifest["boundary_metrics"]["north_south_halo_max_abs_diff"], 0.0)
            self.assertTrue(manifest["event_survived_reload"])
            self.assertLessEqual(manifest["cache_count_after_smoke"], manifest["cache_bound"])
            self.assertTrue(manifest["skill_transfer"]["completed"])
            self.assertEqual(manifest["skill_transfer"]["chosen_action"], "MOVE_RIGHT")
            self.assertTrue((out_dir / "maps" / "chunk_manifest.jsonl").exists())
            self.assertTrue((out_dir / "chunk_events.jsonl").exists())
            self.assertTrue((out_dir / "skills" / "skill_transfer_trace.jsonl").exists())
            self.assertTrue((out_dir / "capability" / "chunk_world_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
