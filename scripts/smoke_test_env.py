from __future__ import annotations

import _bootstrap  # noqa: F401

from evogrid.agents.greedy_agent import GreedyAgent
from evogrid.agents.random_agent import RandomAgent
from evogrid.envs import EvoGridMineEnv
from evogrid.training.rollout import run_episode


def main() -> None:
    for name, agent in [("random", RandomAgent()), ("greedy", GreedyAgent())]:
        env = EvoGridMineEnv()
        result = run_episode(env, agent, seed=0)
        metrics = result.metrics
        print(
            f"{name}: reward={metrics['episode_reward']:.2f}, "
            f"delivered={metrics['ore_delivered']}, "
            f"invalid={metrics['invalid_actions']}, steps={metrics['steps']}"
        )


if __name__ == "__main__":
    main()
