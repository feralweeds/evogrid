"""Qualitative rollout capture for maps and heatmaps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evogrid.agents.deepseek_agent import DeepSeekAgent
from evogrid.agents.greedy_agent import GreedyAgent
from evogrid.agents.hybrid_agent import HybridAgent
from evogrid.agents.random_agent import RandomAgent
from evogrid.envs import EvoGridMineEnv
from evogrid.envs.gym_wrapper import GymEvoGridMineEnv
from evogrid.llm.deepseek_client import DeepSeekClient


@dataclass
class QualitativeRollout:
    group: str
    before_grid: list[list[int]]
    after_grid: list[list[int]]
    agent_pos: tuple[int, int]
    visited_counts: dict[tuple[int, int], int]
    changed_cells: set[tuple[int, int]]
    built_roads: set[tuple[int, int]]
    dug_cells: set[tuple[int, int]]
    frames: list[str]
    metrics: dict
    trace: list[dict]


def capture_baseline_rollout(group: str, config: dict, seed: int = 0) -> QualitativeRollout:
    env = EvoGridMineEnv(config)
    agent = RandomAgent() if group == "random" else GreedyAgent()
    obs, info = env.reset(seed=seed)
    agent.reset(seed)
    before_grid = [row[:] for row in env.state.grid]
    frames = [env.render()]
    while True:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break
    return _rollout_from_env(group, env, before_grid, frames, info)


def capture_ppo_rollout(group: str, config: dict, model_path: str | Path, seed: int = 0) -> QualitativeRollout:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError("stable-baselines3 is required to capture PPO rollouts.") from exc

    model = PPO.load(str(model_path))
    env = GymEvoGridMineEnv(config)
    obs, info = env.reset(seed=seed)
    before_grid = [row[:] for row in env.env.state.grid]
    frames = [env.render()]
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        frames.append(env.render())
        if terminated or truncated:
            break
    return _rollout_from_env(group, env.env, before_grid, frames, info)


def capture_deepseek_rollout(
    group: str,
    config: dict,
    deepseek_runtime: dict,
    mock_responses=None,
    seed: int = 0,
    log_llm_calls: bool = True,
    trace_prompts: bool = False,
) -> QualitativeRollout:
    env = EvoGridMineEnv(config)
    client = None if mock_responses else DeepSeekClient(
        base_url=deepseek_runtime.get("base_url"),
        model=deepseek_runtime.get("model"),
        timeout=deepseek_runtime.get("timeout", 30),
        max_tokens=deepseek_runtime.get("max_tokens", 512),
        json_mode=deepseek_runtime.get("json_mode", True),
    )
    if group == "hybrid_deepseek_greedy":
        agent = HybridAgent(
            client=client,
            replan_interval=deepseek_runtime.get("replan_interval", 20),
            mock_responses=mock_responses,
            temperature=deepseek_runtime.get("temperature", 0.2),
            max_retries=deepseek_runtime.get("max_retries", 0),
            trace_prompts=trace_prompts,
            log_llm_calls=log_llm_calls,
            log_prefix=f"[render {group} seed={seed}]",
        )
    else:
        agent = DeepSeekAgent(
            client=client,
            mode=deepseek_runtime.get("mode", "planner"),
            replan_interval=deepseek_runtime.get("replan_interval", 20),
            mock_responses=mock_responses,
            temperature=deepseek_runtime.get("temperature", 0.2),
            max_retries=deepseek_runtime.get("max_retries", 0),
            trace_prompts=trace_prompts,
            log_llm_calls=log_llm_calls,
            log_prefix=f"[render {group} seed={seed}]",
        )
    obs, info = env.reset(seed=seed)
    agent.reset(seed)
    before_grid = [row[:] for row in env.state.grid]
    frames = [env.render()]
    while True:
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break
    return _rollout_from_env(group, env, before_grid, frames, info, trace=list(agent.trace))


def _rollout_from_env(
    group: str,
    env: EvoGridMineEnv,
    before_grid: list[list[int]],
    frames: list[str],
    metrics: dict,
    trace: list[dict] | None = None,
) -> QualitativeRollout:
    state = env.state
    return QualitativeRollout(
        group=group,
        before_grid=before_grid,
        after_grid=[row[:] for row in state.grid],
        agent_pos=state.agent_pos,
        visited_counts=dict(state.visited_counts),
        changed_cells=set(state.changed_cells),
        built_roads=set(state.built_roads),
        dug_cells=set(state.dug_cells),
        frames=frames,
        metrics={key: value for key, value in metrics.items() if key != "map_summary"},
        trace=trace or [],
    )
