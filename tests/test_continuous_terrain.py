from __future__ import annotations

import json
import unittest

from evogrid.constants import Action, Tile
from evogrid.envs import EvoGridMineEnv
from evogrid.envs.map_state import MapState


class ContinuousTerrainStateTest(unittest.TestCase):
    def test_legacy_maps_have_no_roughness_sidecar(self):
        env = EvoGridMineEnv()

        env.reset(seed=0)

        assert env.state is not None
        self.assertIsNone(env.state.roughness)
        self.assertEqual(env.state.static_diagnostics["placement_status"], "legacy_carved")

    def test_fractal_reset_stores_roughness_map_id_and_diagnostics(self):
        env = EvoGridMineEnv(_fractal_config())

        env.reset(seed=3)

        assert env.state is not None
        self.assertIsNotNone(env.state.roughness)
        self.assertTrue(env.state.map_id)
        self.assertEqual(env.state.static_diagnostics["placement_status"], "ok")
        self.assertEqual(len(env.state.roughness), env.state.height)
        self.assertEqual(len(env.state.roughness[0]), env.state.width)
        json.dumps(env.state.to_dict(), sort_keys=True)

    def test_build_road_changes_grid_but_keeps_roughness_sidecar(self):
        env = EvoGridMineEnv(_fractal_config(p_open=1.0))
        obs, _ = env.reset(seed=4)
        assert env.state is not None

        target = (env.state.agent_pos[0], env.state.agent_pos[1] + 1)
        before = env.state.roughness[target[0]][target[1]]
        env.step(Action.MOVE_RIGHT)
        obs, _, _, _, info = env.step(Action.BUILD_ROAD)

        self.assertEqual(Tile(env.state.grid[target[0]][target[1]]), Tile.ROAD)
        self.assertEqual(env.state.roughness[target[0]][target[1]], before)
        self.assertEqual(info["num_build_road"], 1)

    def test_dig_projection_keeps_potential_roughness_value(self):
        state = MapState(
            grid=[
                [int(Tile.GROUND), int(Tile.OBSTACLE)],
                [int(Tile.GROUND), int(Tile.GROUND)],
            ],
            roughness=[[0.1, 0.9], [0.2, 0.3]],
            map_id="manual",
            static_diagnostics={"placement_status": "ok"},
            base_pos=(0, 0),
            ore_positions={(1, 1)},
            agent_pos=(0, 0),
        )

        state.set_tile((0, 1), Tile.GROUND)

        self.assertEqual(Tile(state.grid[0][1]), Tile.GROUND)
        self.assertEqual(state.roughness[0][1], 0.9)

    def test_clear_shaping_restores_grid_not_roughness(self):
        config = _fractal_config(p_open=1.0)
        config["env"]["shaping"] = {"allow_dig": True, "allow_build_road": True, "reset_after_dropoff": True}
        env = EvoGridMineEnv(config)
        env.reset(seed=5)
        assert env.state is not None

        target = (env.state.agent_pos[0], env.state.agent_pos[1] + 1)
        before = env.state.roughness[target[0]][target[1]]
        env.step(Action.MOVE_RIGHT)
        env.step(Action.BUILD_ROAD)
        env._clear_shaping()

        self.assertNotEqual(Tile(env.state.grid[target[0]][target[1]]), Tile.ROAD)
        self.assertEqual(env.state.roughness[target[0]][target[1]], before)

    def test_continuous_move_cost_uses_roughness_table(self):
        for roughness, expected_cost in ((0.0, 0.01), (0.5, 0.03), (1.0, 0.05)):
            with self.subTest(roughness=roughness):
                env = EvoGridMineEnv(_fractal_config(p_open=1.0, reward_mode="continuous_terrain"))
                env.reset(seed=6)
                assert env.state is not None
                target = (env.state.agent_pos[0], env.state.agent_pos[1] + 1)
                env.state.grid[target[0]][target[1]] = int(Tile.GROUND)
                env.state.roughness[target[0]][target[1]] = roughness

                _, reward, _, _, _ = env.step(Action.MOVE_RIGHT)

                self.assertAlmostEqual(reward, -expected_cost)

    def test_continuous_road_credit_uses_actual_saved_cost(self):
        env = EvoGridMineEnv(_fractal_config(p_open=1.0, reward_mode="continuous_terrain"))
        env.reset(seed=7)
        assert env.state is not None
        target = (env.state.agent_pos[0], env.state.agent_pos[1] + 1)
        env.state.grid[target[0]][target[1]] = int(Tile.GROUND)
        env.state.roughness[target[0]][target[1]] = 0.5

        env.step(Action.MOVE_RIGHT)
        env.step(Action.BUILD_ROAD)
        record = env.state.road_credit_tracker.records[target]

        self.assertAlmostEqual(record.original_move_cost, 0.03)
        self.assertAlmostEqual(record.road_move_cost, 0.0)
        self.assertAlmostEqual(record.build_cost, 0.1)
        for _ in range(4):
            env.step(Action.MOVE_LEFT)
            _, reward, _, _, _ = env.step(Action.MOVE_RIGHT)
            self.assertAlmostEqual(reward, 0.0)

        self.assertEqual(record.usage_count, 4)
        self.assertAlmostEqual(record.saved_cost, 0.12)
        self.assertAlmostEqual(record.net_payoff, 0.02)

    def test_legacy_discrete_rewards_are_unchanged(self):
        env = EvoGridMineEnv()
        env.reset(seed=0)

        _, reward, _, _, _ = env.step(Action.MOVE_RIGHT)

        self.assertAlmostEqual(reward, -0.01)

    def test_partial_observation_exposes_bands_not_continuous_roughness_by_default(self):
        config = _fractal_config(p_open=1.0)
        config["env"]["observation"] = {"mode": "partial_obs", "local_view_radius": 1}
        env = EvoGridMineEnv(config)

        obs, _ = env.reset(seed=8)

        self.assertIn("local_terrain_bands", obs)
        self.assertIn("terrain_band", obs["visible_tiles"][0])
        self.assertNotIn("roughness", obs["visible_tiles"][0])

    def test_audit_counts_continuous_route_rough_bands(self):
        env = EvoGridMineEnv(_fractal_config(p_open=1.0))
        env.reset(seed=9)
        assert env.state is not None
        env.initial_grid = [
            [int(Tile.BASE), int(Tile.GROUND), int(Tile.ORE)],
            [int(Tile.GROUND), int(Tile.GROUND), int(Tile.GROUND)],
            [int(Tile.GROUND), int(Tile.GROUND), int(Tile.GROUND)],
        ]
        env.state.grid = [row[:] for row in env.initial_grid]
        env.state.base_pos = (0, 0)
        env.state.agent_pos = (0, 0)
        env.state.ore_positions = {(0, 2)}
        env.state.roughness = [
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1],
        ]

        audit = env.get_audit_snapshot()

        self.assertEqual(audit["route_rough_tile_count"], 1)
        self.assertEqual(audit["positive_road_opportunity_count"], 1)

    def test_partial_observation_can_expose_continuous_roughness_for_ablation(self):
        config = _fractal_config(p_open=1.0)
        config["env"]["observation"] = {
            "mode": "partial_obs",
            "local_view_radius": 1,
            "expose_continuous_roughness": True,
        }
        env = EvoGridMineEnv(config)

        obs, _ = env.reset(seed=8)

        self.assertIn("terrain_band", obs["visible_tiles"][0])
        self.assertIn("roughness", obs["visible_tiles"][0])


def _fractal_config(p_open: float = 0.8, reward_mode: str = "legacy_discrete") -> dict:
    return {
        "env": {
            "map_mode": "fractal_percolation",
            "grid_size": [16, 16],
            "max_steps": 100,
            "reward_mode": reward_mode,
            "world": {
                "schema_version": 1,
                "generator_version": "spectral_fbm_v1",
                "topology": {
                    "model": "correlated_site",
                    "p_open": p_open,
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
                    "min_base_distance": 4,
                    "min_pair_distance": 1,
                },
                "placement": {"base_margin": 2, "max_attempts": 100},
            },
            "observation": {"mode": "full_obs", "local_view_radius": 4},
            "shaping": {"allow_dig": True, "allow_build_road": True, "reset_after_dropoff": False},
        }
    }


if __name__ == "__main__":
    unittest.main()
