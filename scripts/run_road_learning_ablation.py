from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import copy
import csv
import json
from pathlib import Path

from evogrid.agents import AgentMemory, LearnedRoadAgent, RouteOnlyAgent, RuleRoadOracleAgent
from evogrid.constants import ACTION_IDS, Action, Tile
from evogrid.envs import EvoGridMineEnv
from evogrid.utils.config import load_yaml


GROUPS = ["no_shaping", "route_only", "rule_road_oracle", "learned_road"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run road-learning ablation groups.")
    parser.add_argument("--env-config", default="configs/env_road_learning.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--out", default="outputs/runs/road_learning_ablation")
    parser.add_argument("--settings", nargs="+", default=["cold_start", "warm_start"])
    parser.add_argument("--warm-start-episodes", type=int, default=1)
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

    all_rows: list[dict] = []
    summary = {
        "schema_version": 1,
        "config": {
            "env_config": args.env_config,
            "episodes": args.episodes,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "settings": args.settings,
            "warm_start_episodes": args.warm_start_episodes,
            "learned_value_threshold": args.learned_value_threshold,
            "confidence_threshold": args.confidence_threshold,
        },
        "groups": {},
        "artifacts": {
            "metrics_csv": str(out_dir / "metrics.csv"),
            "summary_json": str(out_dir / "summary.json"),
            "episodes_dir": str(episodes_dir),
            "traces_dir": str(traces_dir),
        },
    }

    for setting in args.settings:
        warm_memory = None
        warm_record_count = 0
        if setting == "warm_start":
            warm_memory, warm_record_count = _collect_warm_start_memory(
                config=base_config,
                episodes=args.warm_start_episodes,
                seed=args.seed + 50_000,
                out_dir=episodes_dir / setting / "warm_start",
            )

        for group in GROUPS:
            rows = _run_group(
                setting=setting,
                group=group,
                base_config=base_config,
                episodes=args.episodes,
                seed=args.seed,
                warm_memory=warm_memory,
                warm_record_count=warm_record_count,
                learned_value_threshold=args.learned_value_threshold,
                confidence_threshold=args.confidence_threshold,
                episodes_dir=episodes_dir,
                traces_dir=traces_dir,
            )
            all_rows.extend(rows)
            summary["groups"][f"{setting}:{group}"] = _summarize_rows(rows)

    _write_csv(out_dir / "metrics.csv", all_rows)
    _write_json(out_dir / "summary.json", summary)
    print(f"Wrote {out_dir / 'metrics.csv'}")
    print(f"Wrote {out_dir / 'summary.json'}")


def _collect_warm_start_memory(config: dict, episodes: int, seed: int, out_dir: Path) -> tuple[AgentMemory, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    memory = AgentMemory()
    record_count = 0
    for episode in range(episodes):
        env = EvoGridMineEnv(_oracle_config(config))
        agent = RuleRoadOracleAgent()
        result = _run_episode(env, agent, seed + episode)
        records = result["metrics"].get("road_credit_records", [])
        memory.add_road_credit_records(records)
        record_count += len(records)
        _write_json(out_dir / f"episode_{episode:03d}.json", result)
    return memory, record_count


def _run_group(
    setting: str,
    group: str,
    base_config: dict,
    episodes: int,
    seed: int,
    warm_memory: AgentMemory | None,
    warm_record_count: int,
    learned_value_threshold: float,
    confidence_threshold: float,
    episodes_dir: Path,
    traces_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    config = _config_for_group(base_config, group)
    memory = _memory_for_group(group, warm_memory)
    for episode in range(episodes):
        episode_seed = seed + episode
        env = EvoGridMineEnv(copy.deepcopy(config))
        agent = _agent_for_group(
            group,
            memory,
            learned_value_threshold=learned_value_threshold,
            confidence_threshold=confidence_threshold,
        )
        result = _run_episode(env, agent, episode_seed)
        row = _row_from_result(
            result=result,
            setting=setting,
            group=group,
            seed=seed,
            episode=episode,
            warm_record_count=warm_record_count,
        )
        rows.append(row)
        _write_json(episodes_dir / setting / group / f"episode_{episode:03d}.json", result)
        _write_json(traces_dir / setting / group / f"episode_{episode:03d}_trace.json", {"trace": result["trace"]})
        print(
            "setting={setting} group={group} episode={episode} reward={reward:.2f} "
            "ore={ore} roads={roads} road_net={road_net:.3f} p_pos={p_pos:.3f}".format(
                setting=setting,
                group=group,
                episode=episode,
                reward=float(row["episode_reward"]),
                ore=int(row["ore_delivered"]),
                roads=int(row["num_build_road"]),
                road_net=float(row["road_net_payoff"]),
                p_pos=float(row["p_build_given_learned_positive"]),
            ),
            flush=True,
        )
    return rows


def _config_for_group(base_config: dict, group: str) -> dict:
    if group == "rule_road_oracle":
        return _oracle_config(base_config)
    config = copy.deepcopy(base_config)
    shaping = config.setdefault("env", {}).setdefault("shaping", {})
    shaping["allow_build_road"] = group != "no_shaping"
    return config


def _oracle_config(base_config: dict) -> dict:
    config = copy.deepcopy(base_config)
    env_config = config.setdefault("env", {})
    env_config["observation_mode"] = "full_obs"
    env_config.setdefault("observation", {})["mode"] = "full_obs"
    env_config.setdefault("shaping", {})["allow_build_road"] = True
    return config


def _memory_for_group(group: str, warm_memory: AgentMemory | None) -> AgentMemory | None:
    if group not in {"no_shaping", "route_only", "learned_road"}:
        return None
    if group == "learned_road" and warm_memory is not None:
        return copy.deepcopy(warm_memory)
    return AgentMemory()


def _agent_for_group(
    group: str,
    memory: AgentMemory | None,
    learned_value_threshold: float,
    confidence_threshold: float,
):
    if group == "rule_road_oracle":
        return RuleRoadOracleAgent()
    if group == "learned_road":
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
    return {
        "metrics": info,
        "total_reward": total_reward,
        "step_trace": step_trace,
        "trace": list(getattr(agent, "trace", [])),
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


def _row_from_result(
    result: dict,
    setting: str,
    group: str,
    seed: int,
    episode: int,
    warm_record_count: int,
) -> dict:
    metrics = result["metrics"]
    trace = result["trace"]
    row = {
        "setting": setting,
        "group": group,
        "seed": seed,
        "episode": episode,
        "warm_start_record_count": warm_record_count if setting == "warm_start" else 0,
    }
    for key, value in metrics.items():
        if key in {"map_summary", "road_credit_records"}:
            continue
        row[key] = value
    row.update(_road_quality_metrics(metrics, trace, group))
    row.update(_learned_influence_metrics(trace))
    return row


def _road_quality_metrics(metrics: dict, trace: list[dict], group: str) -> dict:
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
    if group == "rule_road_oracle" and road_count:
        route_build_count = road_count
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


def _summarize_rows(rows: list[dict]) -> dict:
    summary = {"episode_count": len(rows), "metrics": {}}
    if not rows:
        return summary
    numeric_keys = sorted(
        key
        for key, value in rows[0].items()
        if key not in {"setting", "group"} and isinstance(value, (int, float, bool))
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
