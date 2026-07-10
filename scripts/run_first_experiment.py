from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import itertools
import json
from pathlib import Path

from evogrid.agents.deepseek_agent import DeepSeekAgent
from evogrid.agents.greedy_agent import GreedyAgent
from evogrid.agents.hybrid_agent import HybridAgent
from evogrid.agents.random_agent import RandomAgent
from evogrid.envs import EvoGridMineEnv
from evogrid.envs.gym_wrapper import GymEvoGridMineEnv
from evogrid.evaluation.compare_groups import summarize_group
from evogrid.evaluation.summary import write_csv, write_json
from evogrid.llm.deepseek_client import DeepSeekClient
from evogrid.training.rollout import run_episode
from evogrid.training.train_ppo import train_ppo
from evogrid.utils.config import load_yaml

DEFAULT_GROUPS = ["full_shaping", "no_shaping", "random", "greedy", "hybrid_deepseek_greedy"]
PPO_GROUPS = {"full_shaping", "no_shaping"}
BASELINE_GROUPS = {"random", "greedy"}
LLM_GROUPS = {"deepseek_planner", "hybrid_deepseek_greedy"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first EvoGrid-Mine experiment.")
    parser.add_argument("--experiment-config", default="configs/experiment_first.yaml")
    parser.add_argument("--env-config", default="configs/env_fixed_map.yaml")
    parser.add_argument("--full-env-config", default="configs/env_full_shaping.yaml")
    parser.add_argument("--no-env-config", default="configs/env_no_shaping.yaml")
    parser.add_argument("--out")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--ppo-n-steps", type=int)
    parser.add_argument("--ppo-batch-size", type=int)
    parser.add_argument("--ppo-verbose", type=int)
    parser.add_argument("--deepseek-config", default="configs/deepseek.yaml")
    parser.add_argument("--deepseek-replan-interval", type=int)
    parser.add_argument("--deepseek-temperature", type=float)
    parser.add_argument("--deepseek-max-retries", type=int)
    parser.add_argument("--deepseek-max-steps", type=int)
    parser.add_argument("--deepseek-trace-prompts", action="store_true")
    parser.add_argument("--mock-deepseek", action="store_true")
    parser.add_argument("--mock-deepseek-response", action="append", default=[])
    parser.add_argument("--quiet-llm-calls", action="store_true")
    args = parser.parse_args()

    experiment_config = load_yaml(args.experiment_config)
    groups = args.groups or experiment_config.get("groups", DEFAULT_GROUPS)
    groups = [group for group in groups if group in PPO_GROUPS | BASELINE_GROUPS | LLM_GROUPS]
    seeds = args.seeds or experiment_config.get("seeds", [0])
    timesteps = args.timesteps or int(experiment_config.get("total_timesteps", 10_000))
    eval_episodes = args.eval_episodes or int(experiment_config.get("eval_episodes", 5))
    output_dir = args.out or experiment_config.get("output_dir", "outputs/first_experiment")
    ppo_config = experiment_config.get("ppo", {})
    ppo_n_steps = args.ppo_n_steps or int(ppo_config.get("n_steps", 512))
    ppo_batch_size = args.ppo_batch_size or int(ppo_config.get("batch_size", 64))
    ppo_verbose = args.ppo_verbose if args.ppo_verbose is not None else int(ppo_config.get("verbose", 0))
    config_paths = _config_paths(args, experiment_config)
    deepseek_config = load_yaml(args.deepseek_config).get("deepseek", {})
    deepseek_runtime = _build_deepseek_runtime(args, deepseek_config)
    mock_deepseek_responses = _mock_deepseek_responses(args)

    out_dir = Path(output_dir)
    model_dir = out_dir / "models"
    metrics_dir = out_dir / "metrics"
    rollout_dir = out_dir / "rollouts"
    logs_dir = out_dir / "logs"
    config_snapshot_dir = out_dir / "configs"
    llm_trace_dir = out_dir / "llm_traces"
    for path in [model_dir, metrics_dir, rollout_dir, logs_dir, config_snapshot_dir, llm_trace_dir]:
        path.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {
        "schema_version": 1,
        "config": {
            "groups": groups,
            "seeds": seeds,
            "timesteps": timesteps,
            "eval_episodes": eval_episodes,
            "experiment_config": args.experiment_config,
            "env_configs": config_paths,
            "ppo": {
                "n_steps": ppo_n_steps,
                "batch_size": ppo_batch_size,
                "verbose": ppo_verbose,
            },
            "deepseek": {
                **deepseek_runtime,
                "config_path": args.deepseek_config,
                "mock": bool(mock_deepseek_responses),
            },
        },
        "outputs": {
            "root": str(out_dir),
            "metrics_dir": str(metrics_dir),
            "models_dir": str(model_dir),
            "logs_dir": str(logs_dir),
            "rollouts_dir": str(rollout_dir),
            "configs_dir": str(config_snapshot_dir),
            "llm_traces_dir": str(llm_trace_dir),
        },
        "groups": {},
    }

    for group in groups:
        config = load_yaml(config_paths.get(group, config_paths["full_shaping"]))
        if group in PPO_GROUPS:
            rows = _run_ppo_group(
                group=group,
                config=config,
                seeds=seeds,
                timesteps=timesteps,
                eval_episodes=eval_episodes,
                model_dir=model_dir,
                logs_dir=logs_dir,
                config_snapshot_dir=config_snapshot_dir,
                n_steps=ppo_n_steps,
                batch_size=ppo_batch_size,
                verbose=ppo_verbose,
            )
        elif group in BASELINE_GROUPS:
            rows = _run_baseline_group(
                group=group,
                config=config,
                seeds=seeds,
                eval_episodes=eval_episodes,
            )
        elif group in LLM_GROUPS:
            rows = _run_llm_group(
                group=group,
                config=config,
                seeds=seeds,
                eval_episodes=eval_episodes,
                llm_trace_dir=llm_trace_dir,
                deepseek_runtime=deepseek_runtime,
                mock_responses=mock_deepseek_responses,
                log_llm_calls=not args.quiet_llm_calls,
                trace_prompts=args.deepseek_trace_prompts,
                max_steps=args.deepseek_max_steps,
            )
        else:
            raise ValueError(f"Unknown group: {group}")

        all_rows[group] = rows
        write_csv(metrics_dir / f"{group}_eval.csv", rows)
        summary["groups"][group] = summarize_group(rows)
        print(f"{group}: {summary['groups'][group]}")

    flat_rows = [row for rows in all_rows.values() for row in rows]
    write_csv(metrics_dir / "all_eval.csv", flat_rows)
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.pretty.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote experiment outputs to {out_dir}")


def _run_ppo_group(
    group: str,
    config: dict,
    seeds: list[int],
    timesteps: int,
    eval_episodes: int,
    model_dir: Path,
    logs_dir: Path,
    config_snapshot_dir: Path,
    n_steps: int,
    batch_size: int,
    verbose: int,
) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        model_name = f"{group}_seed{seed}"
        group_model_dir = model_dir / group
        seed_log_dir = logs_dir / group / f"seed{seed}"
        seed_config_path = config_snapshot_dir / f"{group}_seed{seed}.json"
        model_path = group_model_dir / f"{model_name}.zip"
        _write_config_snapshot(
            seed_config_path,
            group=group,
            seed=seed,
            timesteps=timesteps,
            n_steps=n_steps,
            batch_size=batch_size,
            model_path=str(model_path),
            train_log_dir=str(seed_log_dir),
            config=config,
        )
        model = train_ppo(
            config=config,
            total_timesteps=timesteps,
            output_dir=str(group_model_dir),
            seed=seed,
            model_name=model_name,
            n_steps=n_steps,
            batch_size=batch_size,
            verbose=verbose,
            monitor_log_dir=str(seed_log_dir),
        )
        rows.extend(
            _evaluate_ppo_model(
                model=model,
                config=config,
                group=group,
                seed=seed,
                eval_episodes=eval_episodes,
                model_path=str(model_path),
                train_log_dir=str(seed_log_dir),
                train_config_path=str(seed_config_path),
            )
        )
    return rows


def _run_baseline_group(
    group: str,
    config: dict,
    seeds: list[int],
    eval_episodes: int,
) -> list[dict]:
    rows: list[dict] = []
    agent_factory = RandomAgent if group == "random" else GreedyAgent
    for seed in seeds:
        for episode in range(eval_episodes):
            env = EvoGridMineEnv(config)
            agent = agent_factory()
            result = run_episode(env, agent, seed=seed * 10_000 + episode)
            row = _clean_row(result.metrics, group, seed, episode, policy_type="baseline")
            rows.append(row)
    return rows


def _run_llm_group(
    group: str,
    config: dict,
    seeds: list[int],
    eval_episodes: int,
    llm_trace_dir: Path,
    deepseek_runtime: dict,
    mock_responses,
    log_llm_calls: bool,
    trace_prompts: bool,
    max_steps: int | None,
) -> list[dict]:
    if max_steps is not None:
        config = json.loads(json.dumps(config))
        config.setdefault("env", {})["max_steps"] = int(max_steps)

    rows: list[dict] = []
    for seed in seeds:
        seed_trace_path = llm_trace_dir / group / f"seed{seed}" / "llm_trace.jsonl"
        seed_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with seed_trace_path.open("w", encoding="utf-8") as trace_file:
            for episode in range(eval_episodes):
                episode_seed = seed * 10_000 + episode
                env = EvoGridMineEnv(config)
                agent = _make_llm_agent(
                    group=group,
                    deepseek_runtime=deepseek_runtime,
                    mock_responses=mock_responses,
                    log_llm_calls=log_llm_calls,
                    trace_prompts=trace_prompts,
                    log_prefix=f"[{group} seed={seed} episode={episode}]",
                )
                result = run_episode(env, agent, seed=episode_seed)
                row = _clean_row(result.metrics, group, seed, episode, policy_type=_llm_policy_type(group))
                _add_llm_trace_metrics(row, result.trace)
                row["llm_trace_path"] = str(seed_trace_path)
                rows.append(row)
                for trace_id, trace in enumerate(result.trace):
                    record = dict(trace)
                    record.update(
                        {
                            "group": group,
                            "seed": seed,
                            "episode": episode,
                            "episode_seed": episode_seed,
                            "trace_id": trace_id,
                        }
                    )
                    trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(
                    "{group} seed={seed} episode={episode}: reward={reward:.2f} "
                    "ore={ore} llm_calls={calls} fallbacks={fallbacks}".format(
                        group=group,
                        seed=seed,
                        episode=episode,
                        reward=float(row["episode_reward"]),
                        ore=int(row["ore_delivered"]),
                        calls=int(row["llm_calls"]),
                        fallbacks=int(row["llm_fallbacks"]),
                    ),
                    flush=True,
                )
    return rows


def _evaluate_ppo_model(
    model,
    config: dict,
    group: str,
    seed: int,
    eval_episodes: int,
    model_path: str,
    train_log_dir: str,
    train_config_path: str,
) -> list[dict]:
    rows: list[dict] = []
    for episode in range(eval_episodes):
        env = GymEvoGridMineEnv(config)
        obs, info = env.reset(seed=seed * 10_000 + episode)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                break
        row = _clean_row(info, group, seed, episode, policy_type="ppo")
        row["model_path"] = model_path
        row["train_log_dir"] = train_log_dir
        row["train_config_path"] = train_config_path
        rows.append(row)
    return rows


def _clean_row(
    metrics: dict,
    group: str,
    seed: int,
    episode: int,
    policy_type: str,
) -> dict:
    row = {
        "group": group,
        "seed": seed,
        "episode": episode,
        "policy_type": policy_type,
        "model_path": "",
        "train_log_dir": "",
        "train_config_path": "",
    }
    for key, value in metrics.items():
        if key == "map_summary":
            continue
        row[key] = value
    return row


def _config_paths(args: argparse.Namespace, experiment_config: dict) -> dict[str, str]:
    env_configs = experiment_config.get("env_configs", {})
    full_config = args.full_env_config or env_configs.get("full_shaping") or args.env_config
    no_config = args.no_env_config or env_configs.get("no_shaping") or args.env_config
    return {
        "full_shaping": full_config,
        "no_shaping": no_config,
        "random": full_config,
        "greedy": full_config,
        "deepseek_planner": full_config,
        "hybrid_deepseek_greedy": full_config,
    }


def _write_config_snapshot(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_deepseek_runtime(args: argparse.Namespace, config: dict) -> dict:
    return {
        "mode": _config_value(config.get("mode"), "planner"),
        "replan_interval": int(args.deepseek_replan_interval or _config_value(config.get("replan_interval"), 20)),
        "temperature": float(
            args.deepseek_temperature
            if args.deepseek_temperature is not None
            else _config_value(config.get("temperature"), 0.2)
        ),
        "timeout": int(_config_value(config.get("timeout"), 30)),
        "max_tokens": int(_config_value(config.get("max_tokens"), 512)),
        "json_mode": bool(_config_value(config.get("json_mode"), True)),
        "max_retries": int(
            args.deepseek_max_retries
            if args.deepseek_max_retries is not None
            else _config_value(config.get("max_retries"), 0)
        ),
        "max_steps": args.deepseek_max_steps,
        "model": _config_value(config.get("model"), None),
        "base_url": _config_value(config.get("base_url"), None),
    }


def _make_llm_agent(
    group: str,
    deepseek_runtime: dict,
    mock_responses,
    log_llm_calls: bool,
    trace_prompts: bool,
    log_prefix: str,
):
    client = None if mock_responses else DeepSeekClient(
        base_url=deepseek_runtime["base_url"],
        model=deepseek_runtime["model"],
        timeout=deepseek_runtime["timeout"],
        max_tokens=deepseek_runtime["max_tokens"],
        json_mode=deepseek_runtime["json_mode"],
    )
    kwargs = {
        "client": client,
        "replan_interval": deepseek_runtime["replan_interval"],
        "mock_responses": mock_responses,
        "temperature": deepseek_runtime["temperature"],
        "max_retries": deepseek_runtime["max_retries"],
        "trace_prompts": trace_prompts,
        "log_llm_calls": log_llm_calls,
        "log_prefix": log_prefix,
    }
    if group == "hybrid_deepseek_greedy":
        return HybridAgent(**kwargs)
    return DeepSeekAgent(
        mode=deepseek_runtime["mode"],
        client=client,
        replan_interval=deepseek_runtime["replan_interval"],
        mock_responses=mock_responses,
        temperature=deepseek_runtime["temperature"],
        max_retries=deepseek_runtime["max_retries"],
        trace_prompts=trace_prompts,
        log_llm_calls=log_llm_calls,
        log_prefix=log_prefix,
    )


def _mock_deepseek_responses(args: argparse.Namespace):
    if not args.mock_deepseek and not args.mock_deepseek_response:
        return None
    responses = list(args.mock_deepseek_response)
    ok_response = json.dumps(
        {
            "mode": "action",
            "action": "MOVE_RIGHT",
            "action_id": 3,
            "reason": "offline first-experiment smoke test",
            "confidence": 1.0,
        }
    )
    if not responses:
        return itertools.repeat(ok_response)
    responses.append(ok_response)
    return responses


def _add_llm_trace_metrics(row: dict, trace: list[dict]) -> None:
    llm_calls = len(trace)
    llm_fallbacks = sum(1 for item in trace if item.get("fallback_used"))
    llm_successes = llm_calls - llm_fallbacks
    row["llm_calls"] = llm_calls
    row["llm_successes"] = llm_successes
    row["llm_fallbacks"] = llm_fallbacks
    row["llm_success_rate"] = llm_successes / llm_calls if llm_calls else 0.0


def _llm_policy_type(group: str) -> str:
    return "llm_hybrid" if group == "hybrid_deepseek_greedy" else "llm_planner"


def _config_value(value, default):
    if value is None:
        return default
    text = str(value).strip()
    if not text or (text.startswith("${") and text.endswith("}")):
        return default
    return value


if __name__ == "__main__":
    main()
