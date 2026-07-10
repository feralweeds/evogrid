from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import itertools
import json
from pathlib import Path

from evogrid.agents.hybrid_agent import HybridAgent
from evogrid.envs import EvoGridMineEnv
from evogrid.evaluation.compare_groups import summarize_group
from evogrid.llm.deepseek_client import DeepSeekClient
from evogrid.llm.parser import extract_json_object
from evogrid.llm.planner import decision_to_action_id
from evogrid.llm.schemas import LLMDecision
from evogrid.training.rollout import run_episode
from evogrid.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real DeepSeek-backed HybridAgent.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/runs/deepseek_agent")
    parser.add_argument("--env-config", default="configs/env_full_shaping.yaml")
    parser.add_argument("--deepseek-config", default="configs/deepseek.yaml")
    parser.add_argument("--mode", choices=["planner", "action"])
    parser.add_argument("--replan-interval", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--mock-response", action="append", default=[])
    parser.add_argument("--mock-ok", action="store_true")
    parser.add_argument("--skip-api-check", action="store_true")
    parser.add_argument("--require-api", action="store_true")
    parser.add_argument("--trace-prompts", action="store_true")
    parser.add_argument("--quiet-llm-calls", action="store_true")
    args = parser.parse_args()

    env_config = load_yaml(args.env_config)
    deepseek_config = load_yaml(args.deepseek_config).get("deepseek", {})
    if args.max_steps is not None:
        env_config.setdefault("env", {})["max_steps"] = int(args.max_steps)

    run_config = _build_run_config(args, deepseek_config)
    mock_responses = _mock_responses(args)
    client = _build_client(run_config) if not mock_responses else None
    preflight = {"ok": True, "skipped": True, "reason": "mock responses enabled"}
    if not mock_responses and not args.skip_api_check:
        preflight = _preflight_api(client, run_config["temperature"])
        if args.require_api and not preflight["ok"]:
            raise SystemExit(f"DeepSeek API preflight failed: {preflight['error']}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / "llm_trace.jsonl"
    metrics_path = out_dir / "metrics.json"
    episodes: list[dict] = []

    with trace_path.open("w", encoding="utf-8") as trace_file:
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            agent = HybridAgent(
                client=client,
                replan_interval=run_config["replan_interval"],
                mock_responses=mock_responses,
                temperature=run_config["temperature"],
                max_retries=run_config["max_retries"],
                trace_prompts=args.trace_prompts,
                log_llm_calls=not args.quiet_llm_calls,
                log_prefix=f"[deepseek_agent episode={episode} seed={episode_seed}]",
            )
            agent.mode = run_config["mode"]
            result = run_episode(EvoGridMineEnv(env_config), agent, seed=episode_seed)
            row = _episode_row(result.metrics, result.trace, episode, episode_seed)
            episodes.append(row)
            for trace_id, trace in enumerate(result.trace):
                trace_record = dict(trace)
                trace_record["episode"] = episode
                trace_record["seed"] = episode_seed
                trace_record["trace_id"] = trace_id
                trace_file.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
            print(
                "episode={episode} reward={reward:.2f} ore={ore} "
                "llm_calls={calls} fallbacks={fallbacks}".format(
                    episode=episode,
                    reward=float(row["episode_reward"]),
                    ore=int(row["ore_delivered"]),
                    calls=int(row["llm_calls"]),
                    fallbacks=int(row["llm_fallbacks"]),
                )
            )

    metrics = {
        "schema_version": 1,
        "config": {
            "episodes": args.episodes,
            "seed": args.seed,
            "env_config": args.env_config,
            "deepseek_config": args.deepseek_config,
            **run_config,
            "mock": bool(mock_responses),
        },
        "preflight": preflight,
        "episodes": episodes,
        "summary": _summary(episodes),
        "artifacts": {
            "llm_trace": str(trace_path),
            "metrics": str(metrics_path),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {trace_path}")


def _build_run_config(args: argparse.Namespace, config: dict) -> dict:
    return {
        "mode": args.mode or _config_value(config.get("mode"), "planner"),
        "replan_interval": int(args.replan_interval or _config_value(config.get("replan_interval"), 20)),
        "temperature": float(args.temperature if args.temperature is not None else _config_value(config.get("temperature"), 0.2)),
        "timeout": int(args.timeout or _config_value(config.get("timeout"), 30)),
        "max_tokens": int(args.max_tokens or _config_value(config.get("max_tokens"), 512)),
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


def _mock_responses(args: argparse.Namespace):
    responses = list(args.mock_response)
    if args.mock_ok:
        ok_response = json.dumps(
            {
                "mode": "action",
                "action": "MOVE_RIGHT",
                "action_id": 3,
                "reason": "offline smoke test",
                "confidence": 1.0,
            }
        )
        if not responses:
            return itertools.repeat(ok_response)
        responses.append(ok_response)
    return responses or None


def _preflight_api(client: DeepSeekClient, temperature: float) -> dict:
    messages = [
        {"role": "system", "content": "Return JSON only. Do not include markdown."},
        {
            "role": "user",
            "content": (
                'Return {"ok": true, "mode": "action", '
                '"action": "NOOP", "action_id": 8} as JSON.'
            ),
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


def _episode_row(metrics: dict, trace: list[dict], episode: int, seed: int) -> dict:
    row = {key: value for key, value in metrics.items() if key != "map_summary"}
    llm_calls = len(trace)
    llm_fallbacks = sum(1 for item in trace if item.get("fallback_used"))
    llm_successes = llm_calls - llm_fallbacks
    row.update(
        {
            "group": "hybrid_deepseek_greedy",
            "policy_type": "llm_hybrid",
            "episode": episode,
            "seed": seed,
            "llm_calls": llm_calls,
            "llm_successes": llm_successes,
            "llm_fallbacks": llm_fallbacks,
            "llm_success_rate": llm_successes / llm_calls if llm_calls else 0.0,
        }
    )
    return row


def _summary(episodes: list[dict]) -> dict:
    summary = summarize_group(episodes)
    for key in ["llm_calls", "llm_successes", "llm_fallbacks", "llm_success_rate"]:
        values = [float(row.get(key, 0.0) or 0.0) for row in episodes]
        if not values:
            continue
        summary["metrics"][key] = {
            "mean": sum(values) / len(values),
            "std": 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def _config_value(value, default):
    if value is None:
        return default
    text = str(value).strip()
    if not text or (text.startswith("${") and text.endswith("}")):
        return default
    return value


if __name__ == "__main__":
    main()
