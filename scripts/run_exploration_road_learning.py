from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import copy
import csv
import json
from pathlib import Path

from evogrid.agents import AgentMemory, ExplorationRoadAgent, LearnedRoadAgent, RouteOnlyAgent
from evogrid.agents.road_context import contextualize_road_credit_records
from evogrid.constants import ACTION_IDS, Action, Tile
from evogrid.envs import EvoGridMineEnv
from evogrid.utils.config import load_yaml


GROUPS = ["no_shaping", "route_only", "learned_no_explore", "exploration_road"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exploration-based road learning.")
    parser.add_argument("--env-config", default="configs/env_road_learning.yaml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--out", default="outputs/runs/exploration_road_learning")
    parser.add_argument("--groups", nargs="+", default=GROUPS)
    parser.add_argument("--epsilon-schedule", default="0.30,0.20,0.10,0.05")
    parser.add_argument("--epsilon-phase-length", type=int, default=5)
    parser.add_argument("--uncertainty-epsilon", type=float, default=0.60)
    parser.add_argument("--uncertainty-confidence-threshold", type=float, default=0.20)
    parser.add_argument("--max-exploratory-builds-per-episode", type=int, default=3)
    parser.add_argument("--learned-value-threshold", type=float, default=0.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    args = parser.parse_args()

    base_config = load_yaml(args.env_config)
    if args.max_steps is not None:
        base_config.setdefault("env", {})["max_steps"] = int(args.max_steps)

    out_dir = Path(args.out)
    episodes_dir = out_dir / "episodes"
    traces_dir = out_dir / "traces"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    epsilon_values = _parse_schedule(args.epsilon_schedule)
    all_rows: list[dict] = []
    summary = {
        "schema_version": 1,
        "config": {
            "env_config": args.env_config,
            "episodes": args.episodes,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "groups": args.groups,
            "epsilon_schedule": epsilon_values,
            "epsilon_phase_length": args.epsilon_phase_length,
            "uncertainty_epsilon": args.uncertainty_epsilon,
            "uncertainty_confidence_threshold": args.uncertainty_confidence_threshold,
            "max_exploratory_builds_per_episode": args.max_exploratory_builds_per_episode,
            "learned_value_threshold": args.learned_value_threshold,
            "confidence_threshold": args.confidence_threshold,
            "oracle_warm_start": False,
        },
        "groups": {},
        "artifacts": {
            "metrics_csv": str(out_dir / "metrics.csv"),
            "summary_json": str(out_dir / "summary.json"),
            "episodes_dir": str(episodes_dir),
            "traces_dir": str(traces_dir),
        },
    }

    for group in args.groups:
        rows = _run_group(
            group=group,
            base_config=base_config,
            episodes=args.episodes,
            seed=args.seed,
            epsilon_values=epsilon_values,
            epsilon_phase_length=args.epsilon_phase_length,
            uncertainty_epsilon=args.uncertainty_epsilon,
            uncertainty_confidence_threshold=args.uncertainty_confidence_threshold,
            max_exploratory_builds_per_episode=args.max_exploratory_builds_per_episode,
            learned_value_threshold=args.learned_value_threshold,
            confidence_threshold=args.confidence_threshold,
            episodes_dir=episodes_dir,
            traces_dir=traces_dir,
        )
        all_rows.extend(rows)
        summary["groups"][group] = _summarize_rows(rows)

    _write_csv(out_dir / "metrics.csv", all_rows)
    _write_json(out_dir / "summary.json", summary)
    print(f"Wrote {out_dir / 'metrics.csv'}")
    print(f"Wrote {out_dir / 'summary.json'}")


def _run_group(
    group: str,
    base_config: dict,
    episodes: int,
    seed: int,
    epsilon_values: list[float],
    epsilon_phase_length: int,
    uncertainty_epsilon: float,
    uncertainty_confidence_threshold: float,
    max_exploratory_builds_per_episode: int | None,
    learned_value_threshold: float,
    confidence_threshold: float,
    episodes_dir: Path,
    traces_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    config = _config_for_group(base_config, group)
    memory = AgentMemory()
    for episode in range(episodes):
        episode_seed = seed + episode
        epsilon = _scheduled_epsilon(episode, epsilon_values, epsilon_phase_length)
        env = EvoGridMineEnv(copy.deepcopy(config))
        agent = _agent_for_group(
            group=group,
            memory=memory,
            epsilon=epsilon,
            uncertainty_epsilon=uncertainty_epsilon,
            uncertainty_confidence_threshold=uncertainty_confidence_threshold,
            max_exploratory_builds_per_episode=max_exploratory_builds_per_episode,
            learned_value_threshold=learned_value_threshold,
            confidence_threshold=confidence_threshold,
        )
        result = _run_episode(env, agent, episode_seed)
        row = _row_from_result(result, group, seed, episode, epsilon)
        rows.append(row)
        _write_json(episodes_dir / group / f"episode_{episode:03d}.json", result)
        _write_json(traces_dir / group / f"episode_{episode:03d}_trace.json", {"trace": result["trace"]})
        print(
            "group={group} episode={episode} eps={epsilon:.2f} reward={reward:.2f} "
            "ore={ore} roads={roads} exploratory={exploratory} learned={learned} "
            "road_net={road_net:.3f}".format(
                group=group,
                episode=episode,
                epsilon=epsilon,
                reward=float(row["episode_reward"]),
                ore=int(row["ore_delivered"]),
                roads=int(row["num_build_road"]),
                exploratory=int(row["exploratory_build_count"]),
                learned=int(row["learned_build_count"]),
                road_net=float(row["road_net_payoff"]),
            ),
            flush=True,
        )

    _add_group_transition_metrics(rows)
    return rows


def _config_for_group(base_config: dict, group: str) -> dict:
    config = copy.deepcopy(base_config)
    shaping = config.setdefault("env", {}).setdefault("shaping", {})
    shaping["allow_build_road"] = group != "no_shaping"
    return config


def _agent_for_group(
    group: str,
    memory: AgentMemory,
    epsilon: float,
    uncertainty_epsilon: float,
    uncertainty_confidence_threshold: float,
    max_exploratory_builds_per_episode: int | None,
    learned_value_threshold: float,
    confidence_threshold: float,
):
    if group == "exploration_road":
        return ExplorationRoadAgent(
            memory=memory,
            epsilon=epsilon,
            uncertainty_epsilon=uncertainty_epsilon,
            uncertainty_confidence_threshold=uncertainty_confidence_threshold,
            max_exploratory_builds_per_episode=max_exploratory_builds_per_episode,
            learned_value_threshold=learned_value_threshold,
            confidence_threshold=confidence_threshold,
        )
    if group == "learned_no_explore":
        return LearnedRoadAgent(
            memory=memory,
            learned_value_threshold=learned_value_threshold,
            confidence_threshold=confidence_threshold,
        )
    return RouteOnlyAgent(memory=memory)


def _run_episode(env: EvoGridMineEnv, agent, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    agent.reset(seed)
    total_reward = 0.0
    step_trace: list[dict] = []
    while True:
        previous_info = dict(info)
        previous_obs = dict(obs)
        action = agent.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if hasattr(agent, "observe_result"):
            agent.observe_result(action, reward, obs, info, previous_info)
        step_trace.append(_step_record(previous_obs, obs, action, reward))
        if terminated or truncated:
            break
    trace = list(getattr(agent, "trace", []))
    metrics = dict(info)
    metrics["road_credit_records"] = contextualize_road_credit_records(
        list(metrics.get("road_credit_records", [])),
        trace,
    )
    agent_memory = getattr(agent, "memory", None)
    if metrics["road_credit_records"] and hasattr(agent_memory, "add_road_credit_records"):
        agent_memory.add_road_credit_records(metrics["road_credit_records"])
    return {
        "metrics": metrics,
        "total_reward": total_reward,
        "step_trace": step_trace,
        "trace": trace,
    }


def _step_record(previous_obs: dict, obs: dict, action: int, reward: float) -> dict:
    return {
        "step_before": previous_obs.get("step"),
        "step_after": obs.get("step"),
        "agent_pos_before": previous_obs.get("agent_pos"),
        "agent_pos_after": obs.get("agent_pos"),
        "action_id": int(action),
        "action": ACTION_IDS.get(int(action), str(action)),
        "reward": float(reward),
    }


def _row_from_result(result: dict, group: str, seed: int, episode: int, epsilon: float) -> dict:
    metrics = result["metrics"]
    trace = result["trace"]
    row = {
        "group": group,
        "seed": seed,
        "episode": episode,
        "epsilon": epsilon,
        "oracle_warm_start": False,
    }
    for key, value in metrics.items():
        if key in {"map_summary", "road_credit_records"}:
            continue
        row[key] = value
    row.update(_road_quality_metrics(metrics, trace))
    row.update(_learned_influence_metrics(trace))
    row.update(_build_source_metrics(metrics, trace))
    return row


def _road_quality_metrics(metrics: dict, trace: list[dict]) -> dict:
    records = metrics.get("road_credit_records", [])
    road_count = int(metrics.get("road_cells_built", 0) or 0)
    positive_count = int(metrics.get("positive_road_payoff_count", 0) or 0)
    rough_count = sum(1 for record in records if int(record.get("original_tile", -1)) == int(Tile.ROUGH))
    route_build_count = sum(
        1
        for item in trace
        if item.get("action") == "BUILD_ROAD"
        and item.get("shaping_opportunity", {}).get("route_context", {}).get("on_current_route")
    )
    road_net = float(metrics.get("road_net_payoff", 0.0) or 0.0)
    return {
        "positive_road_ratio": positive_count / road_count if road_count else 0.0,
        "avg_payoff_per_road": road_net / road_count if road_count else 0.0,
        "rough_road_ratio": rough_count / road_count if road_count else 0.0,
        "route_road_ratio": route_build_count / road_count if road_count else 0.0,
    }


def _learned_influence_metrics(trace: list[dict]) -> dict:
    seen = []
    positive = []
    nonpositive = []
    build_positive = 0
    build_nonpositive = 0
    for item in trace:
        opportunity = item.get("shaping_opportunity", {})
        estimate = opportunity.get("learned_estimate", {})
        if not opportunity.get("available") or int(estimate.get("evidence_count", 0) or 0) <= 0:
            continue
        seen.append(item)
        if float(estimate.get("learned_value", 0.0) or 0.0) > 0.0:
            positive.append(item)
            if item.get("action") == "BUILD_ROAD":
                build_positive += 1
        else:
            nonpositive.append(item)
            if item.get("action") == "BUILD_ROAD":
                build_nonpositive += 1
    return {
        "learned_estimate_seen_count": len(seen),
        "learned_estimate_positive_count": len(positive),
        "learned_estimate_nonpositive_count": len(nonpositive),
        "build_when_learned_positive_count": build_positive,
        "build_when_learned_nonpositive_count": build_nonpositive,
        "p_build_given_learned_positive": build_positive / len(positive) if positive else 0.0,
        "p_build_given_learned_nonpositive": build_nonpositive / len(nonpositive) if nonpositive else 0.0,
    }


def _build_source_metrics(metrics: dict, trace: list[dict]) -> dict:
    source_by_key = {}
    for item in trace:
        if item.get("action") != "BUILD_ROAD":
            continue
        pos = _position(item.get("agent_pos"))
        step = int(item.get("step", 0) or 0)
        source = item.get("build_decision_source")
        if source is None and item.get("learned_positive"):
            source = "learned"
        source_by_key[(pos, step)] = source or "unknown"

    counts = {
        "exploratory_build_count": 0,
        "exploratory_positive_count": 0,
        "learned_build_count": 0,
        "learned_positive_count": 0,
        "unknown_build_count": 0,
        "unknown_positive_count": 0,
    }
    for record in metrics.get("road_credit_records", []):
        pos = _position(record.get("position"))
        step = int(record.get("build_step", 0) or 0)
        source = source_by_key.get((pos, step), "unknown")
        source_key = source if source in {"exploratory", "learned"} else "unknown"
        counts[f"{source_key}_build_count"] += 1
        if float(record.get("net_payoff", 0.0) or 0.0) > 0.0:
            counts[f"{source_key}_positive_count"] += 1

    exploratory_count = counts["exploratory_build_count"]
    learned_count = counts["learned_build_count"]
    return {
        **counts,
        "exploratory_positive_rate": counts["exploratory_positive_count"] / exploratory_count
        if exploratory_count
        else 0.0,
        "learned_positive_rate": counts["learned_positive_count"] / learned_count if learned_count else 0.0,
        "episode_has_positive_road": int(metrics.get("positive_road_payoff_count", 0) or 0) > 0,
    }


def _add_group_transition_metrics(rows: list[dict]) -> None:
    first_positive = _first_episode(rows, lambda row: int(row.get("positive_road_payoff_count", 0) or 0) > 0)
    first_learned = _first_episode(rows, lambda row: int(row.get("learned_build_count", 0) or 0) > 0)
    cumulative_positive = 0
    for row in rows:
        cumulative_positive += int(row.get("positive_road_payoff_count", 0) or 0)
        row["cumulative_positive_road_count"] = cumulative_positive
        row["time_to_first_positive_road"] = first_positive
        row["cold_to_warm_transition_episode"] = first_learned


def _first_episode(rows: list[dict], predicate) -> int:
    for row in rows:
        if predicate(row):
            return int(row["episode"])
    return -1


def _summarize_rows(rows: list[dict]) -> dict:
    summary = {"episode_count": len(rows), "metrics": {}}
    if not rows:
        return summary
    numeric_keys = sorted(
        key
        for key, value in rows[0].items()
        if key != "group" and isinstance(value, (int, float, bool))
    )
    for key in numeric_keys:
        values = [float(row.get(key, 0.0) or 0.0) for row in rows]
        summary["metrics"][key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }
    return summary


def _parse_schedule(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("epsilon schedule cannot be empty")
    return values


def _scheduled_epsilon(episode: int, values: list[float], phase_length: int) -> float:
    phase_length = max(1, int(phase_length))
    index = min(len(values) - 1, int(episode) // phase_length)
    return float(values[index])


def _position(value) -> tuple[int, int]:
    return int(value[0]), int(value[1])


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
