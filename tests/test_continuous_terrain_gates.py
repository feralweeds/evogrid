from __future__ import annotations

import unittest

from evogrid.evaluation.continuous_terrain_gates import evaluate_continuous_terrain_gates


class ContinuousTerrainGatesTest(unittest.TestCase):
    def test_valid_payload_passes_b0_b3(self):
        report = evaluate_continuous_terrain_gates(_payload())

        self.assertTrue(report.passed)
        self.assertEqual([gate.gate_id for gate in report.gates], ["B0", "B1", "B2", "B3"])

    def test_rejects_hidden_partial_observation_leakage(self):
        payload = _payload()
        payload["leakage_check"]["partial_observation_has_continuous_roughness"] = True

        report = evaluate_continuous_terrain_gates(payload)

        b3 = {gate.gate_id: gate for gate in report.gates}["B3"]
        self.assertFalse(b3.passed)
        self.assertIn("partial observation exposes continuous roughness", b3.details["failures"])


def _payload() -> dict:
    return {
        "schema_version": 1,
        "numeric_cases": [
            {
                "roughness": 0.0,
                "expected_move_cost": 0.01,
                "observed_move_cost": 0.01,
                "expected_saving_per_use": 0.01,
                "observed_saving_per_use": 0.01,
                "expected_break_even_uses": 10,
                "observed_break_even_uses": 10,
            },
            {
                "roughness": 0.5,
                "expected_move_cost": 0.03,
                "observed_move_cost": 0.03,
                "expected_saving_per_use": 0.03,
                "observed_saving_per_use": 0.03,
                "expected_break_even_uses": 4,
                "observed_break_even_uses": 4,
            },
            {
                "roughness": 1.0,
                "expected_move_cost": 0.05,
                "observed_move_cost": 0.05,
                "expected_saving_per_use": 0.05,
                "observed_saving_per_use": 0.05,
                "expected_break_even_uses": 2,
                "observed_break_even_uses": 2,
            },
        ],
        "causal_cases": {
            "no_road": {"move_onto_target_reward": -0.03},
            "reused_road": {
                "move_onto_target_reward_after_road": 0.0,
                "road_net_payoff": 0.02,
                "dropoff_reward_changed": False,
            },
            "unused_road": {"road_net_payoff": -0.1},
            "dig_vs_build_road": {
                "build_road_opened_obstacle": False,
                "dig_opened_obstacle": True,
            },
        },
        "leakage_check": {
            "agent_info_hidden_keys": [],
            "partial_observation_has_full_grid": False,
            "partial_observation_has_continuous_roughness": False,
        },
        "performance_check": {
            "step_count": 200,
            "static_diagnostic_calls_during_steps": 0,
        },
    }


if __name__ == "__main__":
    unittest.main()
