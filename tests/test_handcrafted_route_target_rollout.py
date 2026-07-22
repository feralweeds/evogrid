from __future__ import annotations

import unittest

from scripts.run_handcrafted_route_target_rollout import handcrafted_route_target_candidate


class HandcraftedRouteTargetRolloutTest(unittest.TestCase):
    def test_candidate_uses_generic_route_target_dsl(self):
        spec = handcrafted_route_target_candidate()

        self.assertEqual(spec.status, "candidate")
        self.assertEqual(spec.procedure[0]["op"], "SELECT_TARGET")
        self.assertEqual(spec.procedure[0]["source"], "route.observed_tiles")
        self.assertEqual(spec.procedure[0]["episode_store_as"], "road_target")
        self.assertEqual(spec.budget["episode_use_actions"], ["BUILD_ROAD"])
        self.assertEqual(spec.budget["max_uses_per_episode"], 1)


if __name__ == "__main__":
    unittest.main()
