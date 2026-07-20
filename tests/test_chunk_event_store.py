from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from evogrid.envs.chunk_world import ChunkConfig, ChunkEventStore, ChunkWorld


class ChunkEventStoreTest(unittest.TestCase):
    def test_dynamic_changes_survive_unload_and_reload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            event_path = Path(temp_dir) / "events.jsonl"
            config = ChunkConfig(root_seed=5, chunk_size=8, halo=1, max_cached_chunks=2)
            world = ChunkWorld(config, ChunkEventStore(event_path))
            before = world.get_chunk((0, 0))
            row = config.halo + 4
            col = config.halo + 3
            original_walkable = bool(before.walkable[row, col])
            world.apply_event({"event_type": "build_road", "x": 3, "y": 4, "actor_id": "agent"})
            world.apply_event({"event_type": "deplete_ore", "x": 3, "y": 4})
            world.unload_chunk((0, 0))

            reloaded = ChunkWorld(config, ChunkEventStore(event_path))
            chunk = reloaded.get_chunk((0, 0))

            self.assertTrue(chunk.roads[row, col])
            self.assertEqual(bool(chunk.walkable[row, col]), original_walkable)
            self.assertTrue(chunk.depleted[row, col])
            self.assertFalse(chunk.ore[row, col])

    def test_dig_not_road_changes_walkability(self):
        config = ChunkConfig(root_seed=5, chunk_size=8, halo=1, p_open=0.2)
        world = ChunkWorld(config)
        before = world.get_chunk((0, 0))
        target = None
        for row in range(before.walkable.shape[0]):
            for col in range(before.walkable.shape[1]):
                if not before.walkable[row, col]:
                    target = (col - config.halo, row - config.halo)
                    break
            if target is not None:
                break
        self.assertIsNotNone(target)
        x, y = target

        world.apply_event({"event_type": "build_road", "x": x, "y": y})
        after_road = world.local_observation((x, y), 0)
        self.assertFalse(after_road["walkable"][0, 0])
        self.assertTrue(after_road["roads"][0, 0])

        world.apply_event({"event_type": "dig", "x": x, "y": y})
        after_dig = world.local_observation((x, y), 0)
        self.assertTrue(after_dig["walkable"][0, 0])
        self.assertTrue(after_dig["roads"][0, 0])

    def test_cache_size_stays_bounded(self):
        config = ChunkConfig(root_seed=5, chunk_size=8, halo=1, max_cached_chunks=2)
        world = ChunkWorld(config)

        for cx in range(10):
            world.get_chunk((cx, 0))

        self.assertLessEqual(world.cached_chunk_count(), 2)


if __name__ == "__main__":
    unittest.main()
