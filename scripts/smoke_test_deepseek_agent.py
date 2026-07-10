from __future__ import annotations

import _bootstrap  # noqa: F401

import json

from evogrid.agents.hybrid_agent import HybridAgent
from evogrid.envs import EvoGridMineEnv
from evogrid.training.rollout import run_episode


def main() -> None:
    mock = [
        json.dumps(
            {
                "mode": "plan",
                "subgoal": "start_transport",
                "preferred_actions": ["MOVE_RIGHT"],
                "reason": "Move away from base and start toward ore.",
                "confidence": 0.5,
            }
        )
    ]
    agent = HybridAgent(mock_responses=mock, replan_interval=999)
    env = EvoGridMineEnv()
    result = run_episode(env, agent, seed=0)
    print(
        f"deepseek-mock: reward={result.metrics['episode_reward']:.2f}, "
        f"delivered={result.metrics['ore_delivered']}, trace={len(result.trace)}"
    )


if __name__ == "__main__":
    main()
