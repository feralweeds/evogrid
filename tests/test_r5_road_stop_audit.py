from __future__ import annotations

import unittest

from scripts.run_r5_road_stop_audit import (
    NO_SKILL,
    PRIMARY_METRIC,
    RETURN_GATED_SKILL,
    UNGATED_SKILL,
    compare_groups,
    stop_audit_decision,
    summarize_groups,
    task_no_harm_check,
)


class R5RoadStopAuditTest(unittest.TestCase):
    def test_summary_counts_positive_and_nonpositive_builds(self):
        rows = [
            _row(1, NO_SKILL, road_net=0.0, builds=0),
            _row(1, UNGATED_SKILL, road_net=-0.1, builds=1),
            _row(1, RETURN_GATED_SKILL, road_net=0.5, builds=1),
            _row(2, NO_SKILL, road_net=0.0, builds=0),
            _row(2, UNGATED_SKILL, road_net=0.2, builds=1),
            _row(2, RETURN_GATED_SKILL, road_net=0.0, builds=0),
        ]

        summaries = summarize_groups(rows)

        self.assertEqual(summaries[UNGATED_SKILL]["nonpositive_build_episode_count"], 1)
        self.assertEqual(summaries[UNGATED_SKILL]["positive_build_episode_count"], 1)
        self.assertEqual(summaries[RETURN_GATED_SKILL]["nonpositive_build_episode_count"], 0)
        self.assertEqual(summaries[RETURN_GATED_SKILL]["positive_build_episode_count"], 1)

    def test_decision_freezes_when_gate_improves_and_does_not_harm_task(self):
        rows = [
            _row(1, NO_SKILL, road_net=0.0, builds=0, reward=10.0, ore=1, activation=0.0),
            _row(1, UNGATED_SKILL, road_net=-0.1, builds=1, reward=9.8, ore=1),
            _row(1, RETURN_GATED_SKILL, road_net=0.5, builds=1, reward=10.2, ore=1),
            _row(2, NO_SKILL, road_net=0.0, builds=0, reward=10.0, ore=1, activation=0.0),
            _row(2, UNGATED_SKILL, road_net=0.2, builds=1, reward=10.1, ore=1),
            _row(2, RETURN_GATED_SKILL, road_net=0.0, builds=0, reward=10.1, ore=1, activation=0.0),
        ]
        summaries = summarize_groups(rows)
        comparisons = compare_groups(rows)
        no_harm = task_no_harm_check(comparisons)

        decision = stop_audit_decision(summaries, comparisons, no_harm)

        self.assertTrue(no_harm["passed"])
        self.assertTrue(decision["freeze_handcrafted_return_gated_road_baseline"])
        self.assertEqual(decision["label_if_frozen"], "development_verified")

    def test_no_harm_fails_when_road_net_does_not_improve(self):
        rows = [
            _row(1, NO_SKILL, road_net=0.0, builds=0),
            _row(1, UNGATED_SKILL, road_net=0.2, builds=1),
            _row(1, RETURN_GATED_SKILL, road_net=0.0, builds=0),
        ]

        comparisons = compare_groups(rows)
        no_harm = task_no_harm_check(comparisons)

        self.assertLess(no_harm["road_net_mean_delta"], 0.0)
        self.assertFalse(no_harm["passed"])

    def test_no_harm_tracks_invalid_action_increase(self):
        rows = [
            _row(1, NO_SKILL, road_net=0.0, builds=0),
            _row(1, UNGATED_SKILL, road_net=0.0, builds=0, invalid=0),
            _row(1, RETURN_GATED_SKILL, road_net=0.1, builds=1, invalid=10),
            _row(2, NO_SKILL, road_net=0.0, builds=0),
            _row(2, UNGATED_SKILL, road_net=0.0, builds=0, invalid=0),
            _row(2, RETURN_GATED_SKILL, road_net=0.1, builds=1, invalid=10),
        ]

        comparisons = compare_groups(rows)
        no_harm = task_no_harm_check(comparisons)

        self.assertGreater(no_harm["invalid_actions_mean_delta"], 0.0)
        self.assertFalse(no_harm["invalid_actions_not_up"])
        self.assertFalse(no_harm["passed"])


def _row(
    seed: int,
    group: str,
    *,
    road_net: float,
    builds: int,
    reward: float = 0.0,
    ore: int = 0,
    activation: float = 1.0,
    invalid: int = 0,
) -> dict:
    return {
        "seed": seed,
        "group": group,
        PRIMARY_METRIC: road_net,
        "episode_reward": reward,
        "ore_delivered": ore,
        "steps": 10,
        "num_build_road": builds,
        "road_total_usage_count": 1 if road_net > 0 else 0,
        "road_usage_rate": 1.0 if road_net > 0 else 0.0,
        "transport_steps_per_ore": 10.0,
        "num_dig": 0,
        "num_mine": ore,
        "invalid_actions": invalid,
        "activation_rate": activation,
        "false_trigger_rate": 1.0 if builds > 0 and road_net <= 0.0 else 0.0,
        "runtime_failure_rate": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
