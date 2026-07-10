from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
import itertools
import json
from pathlib import Path

from evogrid.agents import AgentMemory, PartialGreedyAgent, SelfEvolutionAgent
from evogrid.constants import ACTION_IDS
from evogrid.envs import EvoGridMineEnv
from evogrid.llm.deepseek_client import DeepSeekClient
from evogrid.llm.parser import extract_json_object
from evogrid.llm.planner import decision_to_action_id
from evogrid.llm.reflection import generate_rule_reflection
from evogrid.llm.schemas import LLMDecision
from evogrid.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run partial-observation self-evolution experiments.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--agent", choices=["self_evolution", "partial_greedy"], default="self_evolution")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/runs/self_evolution")
    parser.add_argument("--env-config", default="configs/env_full_shaping.yaml")
    parser.add_argument("--deepseek-config", default="configs/deepseek.yaml")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--local-view-radius", type=int, default=4)
    parser.add_argument("--replan-interval", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--mock-deepseek", action="store_true")
    parser.add_argument("--mock-response", action="append", default=[])
    parser.add_argument("--skip-api-check", action="store_true")
    parser.add_argument("--require-api", action="store_true")
    parser.add_argument("--trace-prompts", action="store_true")
    parser.add_argument("--quiet-llm-calls", action="store_true")
    args = parser.parse_args()

    env_config = load_yaml(args.env_config)
    env_settings = env_config.setdefault("env", {})
    if args.max_steps is not None:
        env_settings["max_steps"] = int(args.max_steps)
    env_settings.setdefault("observation", {})
    env_settings["observation"].update(
        {
            "mode": "partial_obs",
            "local_view_radius": int(args.local_view_radius),
        }
    )

    use_deepseek = args.agent == "self_evolution"
    deepseek_config = load_yaml(args.deepseek_config).get("deepseek", {}) if use_deepseek else {}
    run_config = _build_run_config(args, deepseek_config)
    mock_responses = _mock_responses(args) if use_deepseek else None
    client = _build_client(run_config) if use_deepseek and not mock_responses else None

    preflight = {"ok": True, "skipped": True, "reason": "agent does not use DeepSeek"}
    if use_deepseek:
        preflight = {"ok": True, "skipped": True, "reason": "mock responses enabled"}
    if use_deepseek and not mock_responses and not args.skip_api_check:
        preflight = _preflight_api(client, run_config["temperature"])
        if args.require_api and not preflight["ok"]:
            raise SystemExit(f"DeepSeek API preflight failed: {preflight['error']}")

    out_dir = Path(args.out)
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "llm_trace.jsonl"
    route_trace_path = out_dir / "route_trace.jsonl"
    step_trace_path = out_dir / "step_trace.jsonl"
    metrics_csv_path = out_dir / "metrics.csv"
    summary_path = out_dir / "summary.json"
    memory_path = out_dir / "memory.json"

    memory = AgentMemory()
    reflection: dict = {}
    episode_rows: list[dict] = []

    with trace_path.open("w", encoding="utf-8") as trace_file, route_trace_path.open(
        "w", encoding="utf-8"
    ) as route_trace_file, step_trace_path.open("w", encoding="utf-8") as step_trace_file:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            if args.agent == "self_evolution":
                agent = SelfEvolutionAgent(
                    client=client,
                    memory=memory,
                    reflection=reflection,
                    replan_interval=run_config["replan_interval"],
                    mock_responses=mock_responses,
                    temperature=run_config["temperature"],
                    max_retries=run_config["max_retries"],
                    trace_prompts=args.trace_prompts,
                    log_llm_calls=not args.quiet_llm_calls,
                    log_prefix=f"[self_evolution episode={episode} seed={episode_seed}]",
                )
            else:
                agent = PartialGreedyAgent(memory=memory)
            env = EvoGridMineEnv(env_config)
            metrics, total_reward, step_trace = _run_episode(env, agent, episode_seed)
            trace = list(getattr(agent, "trace", []))
            route_trace = list(getattr(agent, "route_trace", []))
            row = _episode_row(
                metrics,
                trace,
                memory,
                env_config,
                episode,
                episode_seed,
                total_reward,
                args.agent,
            )
            if args.agent == "self_evolution":
                reflection = generate_rule_reflection(row, memory)
                memory.add_reflection(reflection)
            else:
                reflection = {
                    "source": "none",
                    "reason": "partial_greedy baseline does not consume episode reflection",
                }
            row["reflection_count"] = len(memory.reflections)
            episode_rows.append(row)

            for trace_id, trace_item in enumerate(trace):
                trace_record = dict(trace_item)
                trace_record["episode"] = episode
                trace_record["seed"] = episode_seed
                trace_record["trace_id"] = trace_id
                trace_file.write(json.dumps(trace_record, ensure_ascii=False) + "\n")

            for route_trace_id, route_item in enumerate(route_trace):
                route_record = dict(route_item)
                route_record["episode"] = episode
                route_record["seed"] = episode_seed
                route_record["route_trace_id"] = route_trace_id
                route_trace_file.write(json.dumps(route_record, ensure_ascii=False) + "\n")

            for step_id, step_item in enumerate(step_trace):
                step_record = dict(step_item)
                step_record["episode"] = episode
                step_record["seed"] = episode_seed
                step_record["step_trace_id"] = step_id
                step_trace_file.write(json.dumps(step_record, ensure_ascii=False) + "\n")

            _write_json(episodes_dir / f"episode_{episode:03d}_metrics.json", row)
            _write_json(episodes_dir / f"episode_{episode:03d}_memory.json", memory.to_dict())
            _write_json(episodes_dir / f"episode_{episode:03d}_reflection.json", reflection)
            _write_json(episodes_dir / f"episode_{episode:03d}_route_trace.json", {"route_plans": route_trace})
            _write_json(episodes_dir / f"episode_{episode:03d}_step_trace.json", {"steps": step_trace})
            print(
                "episode={episode} reward={reward:.2f} ore={ore} "
                "mine={mine} carrying={carrying} known_tiles={known_tiles} known_ore={known_ore} "
                "llm_calls={calls} fallbacks={fallbacks}".format(
                    episode=episode,
                    reward=float(row["episode_reward"]),
                    ore=int(row["ore_delivered"]),
                    mine=int(row.get("num_mine", 0) or 0),
                    carrying=int(bool(row.get("final_has_ore", False))),
                    known_tiles=int(row["known_tile_count"]),
                    known_ore=int(row["known_ore_count"]),
                    calls=int(row["llm_calls"]),
                    fallbacks=int(row["llm_fallbacks"]),
                ),
                flush=True,
            )

    _write_csv(metrics_csv_path, episode_rows)
    _write_json(memory_path, memory.to_dict())
    summary = {
        "schema_version": 1,
        "config": {
            "episodes": args.episodes,
            "agent": args.agent,
            "seed": args.seed,
            "env_config": args.env_config,
            "deepseek_config": args.deepseek_config,
            "observation_mode": "partial_obs",
            "local_view_radius": args.local_view_radius,
            **run_config,
            "mock": bool(mock_responses),
        },
        "preflight": preflight,
        "episodes": episode_rows,
        "summary": _summary(episode_rows),
        "artifacts": {
            "metrics_csv": str(metrics_csv_path),
            "llm_trace": str(trace_path),
            "route_trace": str(route_trace_path),
            "step_trace": str(step_trace_path),
            "memory": str(memory_path),
            "episodes_dir": str(episodes_dir),
        },
    }
    _write_json(summary_path, summary)
    print(f"Wrote {summary_path}")
    print(f"Wrote {metrics_csv_path}")
    print(f"Wrote {trace_path}")
    print(f"Wrote {route_trace_path}")
    print(f"Wrote {step_trace_path}")
    print(f"Wrote {memory_path}")


def _run_episode(env: EvoGridMineEnv, agent, seed: int) -> tuple[dict, float, list[dict]]:
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
        step_trace.append(_step_record(previous_obs, obs, previous_info, info, action, reward, terminated, truncated))
        if terminated or truncated:
            break
    return info, total_reward, step_trace


def _step_record(
    previous_obs: dict,
    obs: dict,
    previous_info: dict,
    info: dict,
    action: int,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> dict:
    action_id = int(action)
    return {
        "step_before": previous_obs.get("step"),
        "step_after": obs.get("step"),
        "action_id": action_id,
        "action": ACTION_IDS.get(action_id, str(action_id)),
        "reward": float(reward),
        "agent_pos_before": previous_obs.get("agent_pos"),
        "agent_pos_after": obs.get("agent_pos"),
        "has_ore_before": previous_obs.get("has_ore"),
        "has_ore_after": obs.get("has_ore"),
        "ore_delivered_before": previous_obs.get("ore_delivered"),
        "ore_delivered_after": obs.get("ore_delivered"),
        "invalid_action_delta": _metric_delta(info, previous_info, "invalid_actions"),
        "num_mine_delta": _metric_delta(info, previous_info, "num_mine"),
        "num_dig_delta": _metric_delta(info, previous_info, "num_dig"),
        "num_build_road_delta": _metric_delta(info, previous_info, "num_build_road"),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def _metric_delta(info: dict, previous_info: dict, key: str) -> int:
    return int(info.get(key, 0) or 0) - int(previous_info.get(key, 0) or 0)


def _build_run_config(args: argparse.Namespace, config: dict) -> dict:
    return {
        "replan_interval": int(args.replan_interval or _config_value(config.get("replan_interval"), 20)),
        "temperature": float(
            args.temperature if args.temperature is not None else _config_value(config.get("temperature"), 0.2)
        ),
        "timeout": int(args.timeout or _config_value(config.get("timeout"), 30)),
        "max_tokens": int(args.max_tokens or _config_value(config.get("max_tokens"), 512)),
        "json_mode": bool(_config_value(config.get("json_mode"), True)),
        "max_retries": int(
            args.max_retries if args.max_retries is not None else _config_value(config.get("max_retries"), 0)
        ),
        "model": args.model or _config_value(config.get("model"), None),
        "base_url": args.base_url or _config_value(config.get("base_url"), None),
    }


def _build_client(config: dict) -> DeepSeekClient:
    return DeepSeekClient(
        base_url=config["base_url"],
        model=config["model"],
        timeout=config["timeout"],
        max_tokens=config["max_tokens"],
        json_mode=config["json_mode"],
    )


def _mock_responses(args: argparse.Namespace):
    default_response = json.dumps(
        {
            "mode": "action",
            "action": "MOVE_RIGHT",
            "action_id": 3,
            "reason": "offline self-evolution smoke test",
            "confidence": 1.0,
        }
    )
    if args.mock_response and args.mock_deepseek:
        return itertools.chain(args.mock_response, itertools.repeat(default_response))
    if args.mock_response:
        return iter(args.mock_response)
    if args.mock_deepseek:
        return itertools.repeat(default_response)
    return None


def _preflight_api(client: DeepSeekClient, temperature: float) -> dict:
    messages = [
        {"role": "system", "content": "Return JSON only. Do not include markdown."},
        {
            "role": "user",
            "content": 'Return {"ok": true, "mode": "action", "action": "NOOP", "action_id": 8} as JSON.',
        },
    ]
    try:
        raw = client.chat(messages, temperature=temperature, json_mode=True)
        parsed = extract_json_object(raw)
        decision = LLMDecision.from_dict(parsed)
        action_id = decision_to_action_id(decision)
        if action_id is None:
            raise ValueError("preflight response did not contain an executable action")
        return {
            "ok": True,
            "skipped": False,
            "parsed": parsed,
            "action_id": action_id,
        }
    except Exception as exc:
        return {
            "ok": False,
            "skipped": False,
            "error": str(exc),
        }


def _episode_row(
    metrics: dict,
    trace: list[dict],
    memory: AgentMemory,
    env_config: dict,
    episode: int,
    seed: int,
    total_reward: float,
    agent_name: str,
) -> dict:
    row = {key: value for key, value in metrics.items() if key != "map_summary"}
    height, width = env_config.get("env", {}).get("grid_size", [0, 0])
    total_cells = int(height) * int(width)
    llm_calls = len(trace)
    llm_fallbacks = sum(1 for item in trace if item.get("fallback_used"))
    row.update(
        {
            "group": f"{agent_name}_partial_obs",
            "policy_type": "llm_self_evolution" if agent_name == "self_evolution" else "partial_baseline",
            "episode": episode,
            "seed": seed,
            "episode_reward": total_reward,
            "observation_mode": "partial_obs",
            "known_tile_count": len(memory.seen_tiles),
            "known_ore_count": len(memory.seen_ore_locations),
            "known_ore_locations": [list(pos) for pos in sorted(memory.seen_ore_locations)],
            "known_obstacle_count": len(memory.seen_obstacles),
            "known_rough_count": len(memory.seen_rough_tiles),
            "known_road_count": len(memory.seen_roads),
            "visited_cell_count": len(memory.visited_counts),
            "map_coverage_rate": len(memory.seen_tiles) / total_cells if total_cells else 0.0,
            "memory_updates": memory.update_count,
            "first_ore_seen_step": memory.first_ore_seen_step,
            "memory_first_mine_step": memory.first_mine_step,
            "failed_action_memory_count": len(memory.failed_actions),
            "llm_calls": llm_calls,
            "llm_successes": llm_calls - llm_fallbacks,
            "llm_fallbacks": llm_fallbacks,
            "llm_success_rate": (llm_calls - llm_fallbacks) / llm_calls if llm_calls else 0.0,
        }
    )
    return row


def _summary(rows: list[dict]) -> dict:
    numeric_keys = [
        "episode_reward",
        "ore_delivered",
        "known_tile_count",
        "known_ore_count",
        "map_coverage_rate",
        "memory_updates",
        "first_ore_seen_step",
        "first_mine_step",
        "memory_first_mine_step",
        "invalid_actions",
        "num_mine",
        "num_dig",
        "num_build_road",
        "road_total_usage_count",
        "road_saved_cost",
        "road_build_cost",
        "road_net_payoff",
        "positive_road_payoff_count",
        "negative_road_payoff_count",
        "final_has_ore",
        "carrying_steps",
        "llm_calls",
        "llm_fallbacks",
        "llm_success_rate",
    ]
    summary = {"episode_count": len(rows), "metrics": {}}
    for key in numeric_keys:
        raw_values = [row.get(key) for row in rows if row.get(key) is not None and row.get(key) != ""]
        values = [float(value) for value in raw_values]
        if not values:
            continue
        mean = sum(values) / len(values)
        summary["metrics"][key] = {
            "mean": mean,
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }
    return summary


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _config_value(value, default):
    if value is None:
        return default
    text = str(value).strip()
    if not text or (text.startswith("${") and text.endswith("}")):
        return default
    return value


if __name__ == "__main__":
    main()
