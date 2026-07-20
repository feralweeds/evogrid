from __future__ import annotations

import unittest

from evogrid.envs.chunk_world import ChunkConfig, ChunkMemory, ChunkWorld


class ChunkAgentMemoryTest(unittest.TestCase):
    def test_local_observation_is_radius_bounded(self):
        world = ChunkWorld(ChunkConfig(root_seed=8, chunk_size=8, halo=1))
        observation = world.local_observation((9, 2), radius=2)

        self.assertEqual(observation["walkable"].shape, (5, 5))
        self.assertEqual(observation["roughness"].shape, (5, 5))
        self.assertEqual(observation["ore"].shape, (5, 5))
        self.assertNotIn("full_map", observation)
        self.assertLessEqual(world.cached_chunk_count(), 0)

    def test_memory_keeps_cross_chunk_summary_for_hierarchical_planning(self):
        world = ChunkWorld(ChunkConfig(root_seed=8, chunk_size=8, halo=1))
        memory = ChunkMemory()
        memory.observe_chunk((0, 0), world.chunk_summary((0, 0)))
        memory.observe_chunk((1, 0), world.chunk_summary((1, 0)))

        hint = memory.hierarchical_plan_hint(start=(1, 1), goal=(17, 2), chunk_size=8)

        self.assertEqual(hint["start_chunk"], (0, 0))
        self.assertEqual(hint["goal_chunk"], (2, 0))
        self.assertEqual(hint["chunk_delta"], (2, 0))
        self.assertEqual(hint["known_chunk_count"], 2)


if __name__ == "__main__":
    unittest.main()
