from __future__ import annotations

import unittest

from scripts.run_handcrafted_return_gate_rollout import handcrafted_return_gate_candidate


class HandcraftedReturnGateRolloutTest(unittest.TestCase):
    def test_candidate_adds_return_phase_gate(self):
        spec = handcrafted_return_gate_candidate()

        self.assertEqual(spec.version, "1.0.1")
        self.assertIn(
            {"feature": "cargo.has_ore", "op": "eq", "value": True},
            spec.applicability["all"],
        )
        self.assertEqual(spec.procedure[0]["source"], "route.observed_tiles")
        self.assertEqual(spec.budget["episode_use_actions"], ["BUILD_ROAD"])


if __name__ == "__main__":
    unittest.main()
