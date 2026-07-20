"""Shared rollout loop for all agent types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpisodeResult:
    metrics: dict
    total_reward: float
    steps: int
    frames: list[str] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)


def run_episode(env, agent, seed: int | None = None, collect_frames: bool = False) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    agent.reset(seed)
    frames: list[str] = []
    total_reward = 0.0

    if collect_frames:
        frames.append(env.render())

    while True:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if collect_frames:
            frames.append(env.render())
        if terminated or truncated:
            break

    final_metrics = env.get_audit_snapshot() if hasattr(env, "get_audit_snapshot") else info
    trace = list(getattr(agent, "trace", []))
    return EpisodeResult(
        metrics=final_metrics,
        total_reward=total_reward,
        steps=final_metrics.get("steps", 0),
        frames=frames,
        trace=trace,
    )


def run_episodes(env_factory, agent_factory, episodes: int, seed: int = 0) -> list[EpisodeResult]:
    results = []
    for idx in range(episodes):
        env = env_factory()
        agent = agent_factory()
        results.append(run_episode(env, agent, seed=seed + idx))
    return results
