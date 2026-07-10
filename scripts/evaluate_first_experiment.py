from __future__ import annotations

import _bootstrap  # noqa: F401

from evogrid.agents.greedy_agent import GreedyAgent
from evogrid.agents.random_agent import RandomAgent
from evogrid.envs import EvoGridMineEnv
from evogrid.evaluation.compare_groups import summarize_group
from evogrid.evaluation.evaluate_policy import evaluate_policy


def main() -> None:
    groups = {
        "random": lambda: RandomAgent(),
        "greedy": lambda: GreedyAgent(),
    }
    for name, factory in groups.items():
        rows = evaluate_policy(lambda: EvoGridMineEnv(), factory, episodes=5)
        print(name, summarize_group(rows))


if __name__ == "__main__":
    main()
