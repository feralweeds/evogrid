"""Evaluate an agent in an environment."""

from __future__ import annotations

from evogrid.training.rollout import run_episode


def evaluate_policy(env_factory, agent_factory, episodes: int = 20, seed: int = 0) -> list[dict]:
    rows = []
    for idx in range(episodes):
        result = run_episode(env_factory(), agent_factory(), seed=seed + idx)
        row = dict(result.metrics)
        row["episode_index"] = idx
        rows.append(row)
    return rows

