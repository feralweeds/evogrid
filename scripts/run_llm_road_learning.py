from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import copy
import csv
import json
from pathlib import Path

from evogrid.agents import AgentMemory, ExplorationRoadAgent, LLMRoadLearningAgent, RouteOnlyAgent
from evogrid.agents.road_context import contextualize_road_credit_records
from evogrid.constants import ACTION_IDS, Action, Tile
from evogrid.envs import EvoGridMineEnv
from evogrid.llm.deepseek_client import DeepSeekClient
from evogrid.llm.parser import extract_json_object
from evogrid.llm.planner import decision_to_action_id
from evogrid.llm.schemas import LLMDecision
from evogrid.utils.config import load_yaml


GROUPS = ["route_only", "exploration_threshold", "llm_no_road_learning", "llm_with_road_learning"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-mediated road-learning experiments.")
    parser.add_argument("--env-config", default="configs/env_road_learning.yaml")
    parser.add_argument("--deepseek-config", default="configs/deepseek.yaml")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--out", default="outputs/runs/llm_road_learning")
    parser.add_argument("--groups", nargs="+", default=GROUPS)
    parser.add_argument("--mock-deepseek", action="store_true")
    parser.add_argument("--require-api", action="store_true")
    parser.add_argument("--skip-api-check", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--trace-prompts", action="store_true")
    parser.add_argument("--quiet-llm-calls", action="store_true")
    parser.add_argument("--epsilon-schedule", default="0.30,0.20,0.10,0.05")
    parser.add_argument("--epsilon-phase-length", type=int, default=5)
    parser.add_argument("--uncertainty-epsilon", type=float, default=0.60)
    parser.add_argument("--uncertainty-confidence-threshold", type=float, default=0.20)
    parser.add_argument("--exploration-budget", type=int, default=3)
    parser.add_argument("--max-learned-builds-per-episode", type=int, default=9)
    parser.add_argument("--learned-value-threshold", type=float, default=0.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--min-contextual-evidence-count", type=int, default=1)
    parser.add_argument("--positive-rate-threshold", type=float, default=0.0)
    parser.add_argument("--require-contextual-evidence", action="store_true")
    parser.add_argument("--require-on-route-learned-build", action="store_true")
    args = parser.parse_args()

    base_config = load_yaml(args.env_config)
    if args.max_steps is not None:
        base_config.setdefault("env", {})["max_steps"] = int(args.max_steps)

    deepseek_config = load_yaml(args.deepseek_config).get("deepseek", {})
    run_config = _build_run_config(args, deepseek_config)
    use_llm = any(group.startswith("llm_") for group in args.groups)
    client = _build_client(run_config) if use_llm and not args.mock_deepseek else None
    mock_client = MockRoadLearningClient() if use_llm and args.mock_deepseek else None
    preflight = {"ok": True, "skipped": True, "reason": "no LLM group requested"}
    if use_llm and args.mock_deepseek:
        preflight = {"ok": True, "skipped": True, "reason": "mock LLM enabled"}
    elif use_llm and not args.skip_api_check:
        preflight = _preflight_api(client, run_config["temperature"])
        if args.require_api and not preflight["ok"]:
            raise SystemExit(f"DeepSeek API preflight failed: {preflight['error']}")

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
            "mock_deepseek": bool(args.mock_deepseek),
            "epsilon_schedule": epsilon_values,
            "epsilon_phase_length": args.epsilon_phase_length,
            "exploration_budget": args.exploration_budget,
            "max_learned_builds_per_episode": args.max_learned_builds_per_episode,
            "learned_value_threshold": args.learned_value_threshold,
            "confidence_threshold": args.confidence_threshold,
            "min_contextual_evidence_count": args.min_contextual_evidence_count,
            "positive_rate_threshold": args.positive_rate_threshold,
            "require_contextual_evidence": bool(args.require_contextual_evidence),
            "require_on_route_learned_build": bool(args.require_on_route_learned_build),
            **run_config,
        },
        "preflight": preflight,
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
            exploration_budget=args.exploration_budget,
            max_learned_builds_per_episode=args.max_learned_builds_per_episode,
            learned_value_threshold=args.learned_value_threshold,
            confidence_threshold=args.confidence_threshold,
            min_contextual_evidence_count=args.min_contextual_evidence_count,
            positive_rate_threshold=args.positive_rate_threshold,
            require_contextual_evidence=args.require_contextual_evidence,
            require_on_route_learned_build=args.require_on_route_learned_build,
            run_config=run_config,
            client=mock_client or client,
            trace_prompts=args.trace_prompts,
            log_llm_calls=not args.quiet_llm_calls,
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
    exploration_budget: int,
    max_learned_builds_per_episode: int | None,
    learned_value_threshold: float,
    confidence_threshold: float,
    min_contextual_evidence_count: int,
    positive_rate_threshold: float,
    require_contextual_evidence: bool,
    require_on_route_learned_build: bool,
    run_config: dict,
    client,
    trace_prompts: bool,
    log_llm_calls: bool,
    episodes_dir: Path,
    traces_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    config = _config_for_group(base_config)
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
            exploration_budget=exploration_budget,
            max_learned_builds_per_episode=max_learned_builds_per_episode,
            learned_value_threshold=learned_value_threshold,
            confidence_threshold=confidence_threshold,
            min_contextual_evidence_count=min_contextual_evidence_count,
            positive_rate_threshold=positive_rate_threshold,
            require_contextual_evidence=require_contextual_evidence,
            require_on_route_learned_build=require_on_route_learned_build,
            run_config=run_config,
            client=client,
            trace_prompts=trace_prompts,
            log_llm_calls=log_llm_calls,
            episode=episode,
            seed=episode_seed,
        )
        result = _run_episode(env, agent, episode_seed)
        row = _row_from_result(result, group, seed, episode, epsilon)
        rows.append(row)
        _write_json(episodes_dir / group / f"episode_{episode:03d}.json", result)
        _write_json(traces_dir / group / f"episode_{episode:03d}_trace.json", {"trace": result["trace"]})
        print(
            "group={group} episode={episode} eps={epsilon:.2f} reward={reward:.2f} "
            "ore={ore} roads={roads} llm_calls={calls} llm_builds={llm_builds} "
            "llm_explore={llm_explore} llm_learned={llm_learned} road_net={road_net:.3f}".format(
                group=group,
                episode=episode,
                epsilon=epsilon,
                reward=float(row["episode_reward"]),
                ore=int(row["ore_delivered"]),
                roads=int(row["num_build_road"]),
                calls=int(row["llm_decision_count"]),
                llm_builds=int(row["llm_build_road_count"]),
                llm_explore=int(row["llm_exploration_build_count"]),
                llm_learned=int(row["llm_learned_build_count"]),
                road_net=float(row["road_net_payoff"]),
            ),
            flush=True,
        )

    _add_group_transition_metrics(rows)
    return rows


def _config_for_group(base_config: dict) -> dict:
    config = copy.deepcopy(base_config)
    config.setdefault("env", {}).setdefault("shaping", {})["allow_build_road"] = True
    return config


def _agent_for_group(
    group: str,
    memory: AgentMemory,
    epsilon: float,
    uncertainty_epsilon: float,
    uncertainty_confidence_threshold: float,
    exploration_budget: int,
    max_learned_builds_per_episode: int | None,
    learned_value_threshold: float,
    confidence_threshold: float,
    min_contextual_evidence_count: int,
    positive_rate_threshold: float,
    require_contextual_evidence: bool,
    require_on_route_learned_build: bool,
    run_config: dict,
    client,
    trace_prompts: bool,
    log_llm_calls: bool,
    episode: int,
    seed: int,
):
    if group == "route_only":
        return RouteOnlyAgent(memory=memory)
    if group == "exploration_threshold":
        return ExplorationRoadAgent(
            memory=memory,
            epsilon=epsilon,
            uncertainty_epsilon=uncertainty_epsilon,
            uncertainty_confidence_threshold=uncertainty_confidence_threshold,
            max_exploratory_builds_per_episode=exploration_budget,
            learned_value_threshold=learned_value_threshold,
            confidence_threshold=confidence_threshold,
            min_contextual_evidence_count=min_contextual_evidence_count,
            positive_rate_threshold=positive_rate_threshold,
            require_contextual_evidence=require_contextual_evidence,
            require_on_route_learned_build=require_on_route_learned_build,
        )
    if group == "llm_no_road_learning":
        return LLMRoadLearningAgent(
            client=client,
            memory=memory,
            use_road_learning=False,
            exploration_budget_per_episode=exploration_budget,
            max_learned_builds_per_episode=max_learned_builds_per_episode,
            learned_value_threshold=learned_value_threshold,
            confidence_threshold=confidence_threshold,
            min_contextual_evidence_count=min_contextual_evidence_count,
            positive_rate_threshold=positive_rate_threshold,
            require_contextual_evidence=require_contextual_evidence,
            require_on_route_learned_build=require_on_route_learned_build,
            temperature=run_config["temperature"],
            max_retries=run_config["max_retries"],
            trace_prompts=trace_prompts,
            log_llm_calls=log_llm_calls,
            log_prefix=f"[{group} episode={episode} seed={seed}]",
        )
    if group == "llm_with_road_learning":
        return LLMRoadLearningAgent(
            client=client,
            memory=memory,
            use_road_learning=True,
            exploration_budget_per_episode=exploration_budget,
            max_learned_builds_per_episode=max_learned_builds_per_episode,
            learned_value_threshold=learned_value_threshold,
            confidence_threshold=confidence_threshold,
            min_contextual_evidence_count=min_contextual_evidence_count,
            positive_rate_threshold=positive_rate_threshold,
            require_contextual_evidence=require_contextual_evidence,
            require_on_route_learned_build=require_on_route_learned_build,
            temperature=run_config["temperature"],
            max_retries=run_config["max_retries"],
            trace_prompts=trace_prompts,
            log_llm_calls=log_llm_calls,
            log_prefix=f"[{group} episode={episode} seed={seed}]",
        )
    raise ValueError(f"unknown group: {group}")


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
    }
    for key, value in metrics.items():
        if key in {"map_summary", "road_credit_records"}:
            continue
        row[key] = value
    row.update(_road_quality_metrics(metrics, trace))
    row.update(_llm_metrics(trace))
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
        if _trace_action(item) == "BUILD_ROAD"
        and item.get("shaping_opportunity", {}).get("route_context", {}).get("on_current_route")
    )
    road_net = float(metrics.get("road_net_payoff", 0.0) or 0.0)
    return {
        "positive_road_ratio": positive_count / road_count if road_count else 0.0,
        "avg_payoff_per_road": road_net / road_count if road_count else 0.0,
        "rough_road_ratio": rough_count / road_count if road_count else 0.0,
        "route_road_ratio": route_build_count / road_count if road_count else 0.0,
    }


def _llm_metrics(trace: list[dict]) -> dict:
    llm_decisions = [item for item in trace if item.get("attempt_count", 0)]
    llm_builds = [item for item in llm_decisions if _trace_action(item) == "BUILD_ROAD"]
    positive = []
    nonpositive = []
    build_positive = 0
    build_nonpositive = 0
    for item in llm_decisions:
        opportunity = item.get("shaping_opportunity", {})
        estimate = opportunity.get("learned_estimate", {})
        if not opportunity.get("available") or int(estimate.get("evidence_count", 0) or 0) <= 0:
            continue
        if float(estimate.get("learned_value", 0.0) or 0.0) > 0.0:
            positive.append(item)
            if _trace_action(item) == "BUILD_ROAD":
                build_positive += 1
        else:
            nonpositive.append(item)
            if _trace_action(item) == "BUILD_ROAD":
                build_nonpositive += 1
    return {
        "llm_decision_count": len(llm_decisions),
        "llm_build_road_count": len(llm_builds),
        "llm_exploration_decision_count": sum(
            1 for item in llm_builds if item.get("build_decision_source") == "llm_exploration"
        ),
        "llm_learned_decision_count": sum(
            1 for item in llm_builds if item.get("build_decision_source") == "llm_learned"
        ),
        "llm_rejected_candidate_count": sum(1 for item in llm_decisions if item.get("llm_rejected_candidate")),
        "llm_fallback_count": sum(1 for item in llm_decisions if item.get("fallback_used")),
        "p_llm_build_given_learned_positive": build_positive / len(positive) if positive else 0.0,
        "p_llm_build_given_learned_nonpositive": build_nonpositive / len(nonpositive) if nonpositive else 0.0,
    }


def _build_source_metrics(metrics: dict, trace: list[dict]) -> dict:
    source_by_key = {}
    for item in trace:
        if _trace_action(item) != "BUILD_ROAD":
            continue
        pos = _position(item.get("agent_pos") or item.get("shaping_opportunity", {}).get("position"))
        step = int(item.get("step", 0) or 0)
        source_by_key[(pos, step)] = item.get("build_decision_source") or "unknown"

    counts = {
        "exploratory_build_count": 0,
        "exploratory_positive_count": 0,
        "learned_build_count": 0,
        "learned_positive_count": 0,
        "llm_exploration_build_count": 0,
        "llm_exploration_positive_count": 0,
        "llm_learned_build_count": 0,
        "llm_learned_positive_count": 0,
        "unknown_build_count": 0,
        "unknown_positive_count": 0,
    }
    for record in metrics.get("road_credit_records", []):
        pos = _position(record.get("position"))
        step = int(record.get("build_step", 0) or 0)
        source = source_by_key.get((pos, step), "unknown")
        source_key = _source_key(source)
        counts[f"{source_key}_build_count"] += 1
        if float(record.get("net_payoff", 0.0) or 0.0) > 0.0:
            counts[f"{source_key}_positive_count"] += 1

    return {
        **counts,
        "exploratory_positive_rate": _ratio(
            counts["exploratory_positive_count"], counts["exploratory_build_count"]
        ),
        "learned_positive_rate": _ratio(counts["learned_positive_count"], counts["learned_build_count"]),
        "llm_exploration_positive_rate": _ratio(
            counts["llm_exploration_positive_count"], counts["llm_exploration_build_count"]
        ),
        "llm_learned_positive_rate": _ratio(
            counts["llm_learned_positive_count"], counts["llm_learned_build_count"]
        ),
    }


def _source_key(source: str) -> str:
    if source == "exploratory":
        return "exploratory"
    if source == "learned":
        return "learned"
    if source == "llm_exploration":
        return "llm_exploration"
    if source == "llm_learned":
        return "llm_learned"
    return "unknown"


def _add_group_transition_metrics(rows: list[dict]) -> None:
    first_positive = _first_episode(rows, lambda row: int(row.get("positive_road_payoff_count", 0) or 0) > 0)
    first_llm_learned = _first_episode(rows, lambda row: int(row.get("llm_learned_build_count", 0) or 0) > 0)
    cumulative_positive = 0
    for row in rows:
        cumulative_positive += int(row.get("positive_road_payoff_count", 0) or 0)
        row["cumulative_positive_road_count"] = cumulative_positive
        row["time_to_first_positive_road"] = first_positive
        row["cold_to_warm_transition_episode"] = first_llm_learned


def _first_episode(rows: list[dict], predicate) -> int:
    for row in rows:
        if predicate(row):
            return int(row["episode"])
    return -1


class MockRoadLearningClient:
    """Deterministic stand-in that follows the LLM road-learning prompt."""

    def chat(self, messages: list[dict], temperature: float = 0.0, json_mode: bool | None = None) -> str:
        payload = json.loads(messages[-1]["content"])
        opportunity = payload.get("shaping_opportunity", {})
        estimate = opportunity.get("learned_estimate", {})
        exploration = payload.get("exploration_state", {})
        route_action = payload.get("route_action", {})
        if (
            opportunity.get("available")
            and exploration.get("learned_evidence_strong")
        ):
            if _learned_budget_remaining(exploration):
                return _decision(Action.BUILD_ROAD, "learned_road", "strong learned payoff evidence")
            action_id = int(route_action.get("action_id", int(Action.NOOP)))
            return _decision(Action(action_id), "route", "learned road build budget exhausted")
        if (
            opportunity.get("available")
            and int(exploration.get("exploration_budget_remaining", 0) or 0) > 0
            and float(opportunity.get("cost", {}).get("saving_per_use", 0.0) or 0.0) > 0.0
            and opportunity.get("route_context", {}).get("on_current_route")
        ):
            return _decision(Action.BUILD_ROAD, "explore_road", "use exploration budget to sample road payoff")
        action_id = int(route_action.get("action_id", int(Action.NOOP)))
        return _decision(Action(action_id), "route", "follow route action")


def _decision(action: Action, source: str, reason: str) -> str:
    return json.dumps(
        {
            "mode": "action",
            "action": action.name,
            "action_id": int(action),
            "decision_source": source,
            "reason": reason,
            "confidence": 1.0,
        }
    )


def _learned_budget_remaining(exploration: dict) -> bool:
    remaining = exploration.get("learned_builds_remaining")
    return remaining is None or int(remaining) > 0


def _build_run_config(args: argparse.Namespace, config: dict) -> dict:
    return {
        "temperature": float(
            args.temperature if args.temperature is not None else _config_value(config.get("temperature"), 0.2)
        ),
        "timeout": int(args.timeout or _config_value(config.get("timeout"), 30)),
        "max_tokens": int(args.max_tokens or _config_value(config.get("max_tokens"), 768)),
        "json_mode": bool(_config_value(config.get("json_mode"), True)),
        "max_retries": int(args.max_retries if args.max_retries is not None else _config_value(config.get("max_retries"), 0)),
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
        return {"ok": True, "skipped": False, "parsed": parsed, "action_id": action_id}
    except Exception as exc:
        return {"ok": False, "skipped": False, "error": str(exc)}


def _summarize_rows(rows: list[dict]) -> dict:
    summary = {"episode_count": len(rows), "metrics": {}}
    if not rows:
        return summary
    numeric_keys = sorted(
        key for key, value in rows[0].items() if key != "group" and isinstance(value, (int, float, bool))
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


def _trace_action(item: dict) -> str:
    if "action" in item:
        return str(item["action"])
    return str(item.get("chosen_action_name", ""))


def _position(value) -> tuple[int, int]:
    return int(value[0]), int(value[1])


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


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


def _config_value(value, default):
    if value is None:
        return default
    text = str(value).strip()
    if not text or (text.startswith("${") and text.endswith("}")):
        return default
    return value


if __name__ == "__main__":
    main()
