from __future__ import annotations

import unittest

from evogrid.constants import Action
from evogrid.envs import EvoGridMineEnv
from evogrid.training.rollout import run_episode
from evogrid.agents.random_agent import RandomAgent


HIDDEN_INFO_KEYS = {
    "route_rough_tile_count",
    "off_route_rough_tile_count",
    "positive_road_opportunity_count",
    "transport_corridor_length",
    "shortest_path_length",
    "largest_component_fraction",
}


class PartialObservationLeakageTest(unittest.TestCase):
    def test_partial_info_excludes_hidden_map_diagnostics(self):
        env = EvoGridMineEnv(_partial_fractal_config())

        obs, info = env.reset(seed=0)

        self.assertEqual(obs["observation_mode"], "partial_obs")
        for key in HIDDEN_INFO_KEYS:
            self.assertNotIn(key, info)
        self.assertNotIn("ore_positions", info["map_summary"])

    def test_audit_snapshot_contains_hidden_diagnostics_outside_agent_path(self):
        env = EvoGridMineEnv(_partial_fractal_config())
        env.reset(seed=0)

        audit = env.get_audit_snapshot()

        self.assertIn("route_rough_tile_count", audit)
        self.assertIn("shortest_path_length", audit)
        self.assertIn("ore_positions", audit["map_summary"])

    def test_partial_steps_do_not_recompute_full_map_diagnostics_for_agent_info(self):
        env = EvoGridMineEnv(_partial_fractal_config())
        env.reset(seed=1)
        calls = 0
        original = env._map_diagnostics

        def counted():
            nonlocal calls
            calls += 1
            return original()

        env._map_diagnostics = counted
        for _ in range(5):
            env.step(Action.NOOP)

        self.assertEqual(calls, 0)
        env.get_audit_snapshot()
        self.assertEqual(calls, 1)

    def test_128_partial_steps_do_not_recompute_static_diagnostics(self):
        env = EvoGridMineEnv(_partial_fractal_config(size=128, max_steps=20))
        env.reset(seed=3)
        calls = 0
        original = env._map_diagnostics

        def counted():
            nonlocal calls
            calls += 1
            return original()

        env._map_diagnostics = counted
        for _ in range(10):
            env.step(Action.NOOP)

        self.assertEqual(calls, 0)

    def test_rollout_returns_audit_metrics_without_exposing_them_to_agent(self):
        env = EvoGridMineEnv(_partial_fractal_config(max_steps=3))
        agent = RandomAgent()

        result = run_episode(env, agent, seed=2)

        self.assertIn("route_rough_tile_count", result.metrics)
        self.assertIn("ore_positions", result.metrics["map_summary"])


def _partial_fractal_config(max_steps: int = 20, size: int = 16) -> dict:
    return {
        "env": {
            "map_mode": "fractal_percolation",
            "grid_size": [size, size],
            "max_steps": max_steps,
            "reward_mode": "continuous_terrain",
            "world": {
                "schema_version": 1,
                "generator_version": "spectral_fbm_v1",
                "topology": {
                    "model": "correlated_site",
                    "p_open": 0.8,
                    "hurst": 0.5,
                    "solvability_mode": "conditioned_same_component",
                    "task_component": "largest",
                    "min_task_component_fraction": 0.0,
                },
                "terrain": {
                    "hurst": 0.7,
                    "base_move_cost": 0.01,
                    "roughness_strength": 0.04,
                    "cost_exponent": 1.0,
                    "road_move_cost": 0.0,
                    "observation_bins": [0.25, 0.5, 0.75],
                },
                "resources": {
                    "distribution": "uniform",
                    "count": 1,
                    "hurst": 0.7,
                    "min_base_distance": max(4, size // 4),
                    "min_pair_distance": 1,
                },
                "placement": {"base_margin": 2, "max_attempts": 100},
            },
            "observation": {"mode": "partial_obs", "local_view_radius": 2},
            "shaping": {"allow_dig": True, "allow_build_road": True},
        }
    }


if __name__ == "__main__":
    unittest.main()
