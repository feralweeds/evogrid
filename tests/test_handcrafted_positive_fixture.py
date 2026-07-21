from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from scripts.run_handcrafted_positive_skill_fixture import run_handcrafted_positive_skill_fixture


class HandcraftedPositiveFixtureTest(unittest.TestCase):
    def test_fixture_promotes_candidate_to_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            out_dir = Path(temp) / "fixture"

            manifest = run_handcrafted_positive_skill_fixture(out_dir)

            self.assertEqual(manifest["decision"], "verified")
            self.assertEqual(manifest["promoted_status"], "verified")
            self.assertEqual(manifest["metrics"]["success_rate"], 0.5)
            self.assertEqual(manifest["metrics"]["false_trigger_rate"], 0.0)
            verified_path = Path(manifest["verified"])
            self.assertTrue(verified_path.exists())
            self.assertFalse(
                (out_dir / "registry" / "candidates" / manifest["candidate_id"] / f"{manifest['candidate_version']}.json").exists()
            )
            verified = json.loads(verified_path.read_text(encoding="utf-8"))
            self.assertEqual(verified["storage_status"], "verified")
            self.assertEqual(verified["spec"]["budget"]["max_uses_per_episode"], 1)
            self.assertTrue(verified["spec"]["budget"]["stop_after_success"])


if __name__ == "__main__":
    unittest.main()
