from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import copy
import json
from pathlib import Path

from evogrid.agents import GreedyAgent, RuleRoadOracleAgent
from evogrid.envs import EvoGridMineEnv
from evogrid.evaluation.compare_groups import summarize_group
from evogrid.evaluation.summary import write_csv, write_json
from evogrid.training.rollout import run_episode
from evogrid.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether rule-based road building has value.")
    parser.add_argument("--env-config", default="configs/env_road_sanity.yaml")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--out", default="outputs/runs/road_oracle_sanity")
    parser.add_argument("--oracle-build-ground", action="store_true")
    args = parser.parse_args()

    base_config = load_yaml(args.env_config)
    if args.max_steps is not None:
        base_config.setdefault("env", {})["max_steps"] = int(args.max_steps)

    out_dir = Path(args.out)
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    groups = {
        "no_road_greedy": (GreedyAgent, _no_road_config(base_config)),
        "rule_road_oracle": (
            lambda: RuleRoadOracleAgent(build_on_ground=args.oracle_build_ground),
            base_config,
        ),
    }
    all_rows: list[dict] = []
    summary = {
        "schema_version": 1,
        "config": {
            "env_config": args.env_config,
            "episodes": args.episodes,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "oracle_build_ground": bool(args.oracle_build_ground),
        },
        "groups": {},
        "artifacts": {
            "metrics_csv": str(out_dir / "metrics.csv"),
            "episodes_dir": str(episodes_dir),
        },
    }

    for group, (agent_factory, config) in groups.items():
        rows: list[dict] = []
        for episode in range(args.episodes):
            episode_seed = args.seed + episode
            env = EvoGridMineEnv(copy.deepcopy(config))
            agent = agent_factory()
            result = run_episode(env, agent, seed=episode_seed)
            row, detail = _episode_outputs(result.metrics, group, args.seed, episode)
            rows.append(row)
            all_rows.append(row)
            write_json(episodes_dir / f"{group}_episode_{episode:03d}.json", detail)
            print(
                "group={group} episode={episode} reward={reward:.2f} ore={ore} "
                "roads={roads} road_uses={uses} road_net={net:.3f}".format(
                    group=group,
                    episode=episode,
                    reward=float(row["episode_reward"]),
                    ore=int(row["ore_delivered"]),
                    roads=int(row["road_cells_built"]),
                    uses=int(row["road_total_usage_count"]),
                    net=float(row["road_net_payoff"]),
                ),
                flush=True,
            )
        summary["groups"][group] = summarize_group(rows)

    write_csv(out_dir / "metrics.csv", all_rows)
    write_json(out_dir / "summary.json", summary)
    print(f"Wrote {out_dir / 'metrics.csv'}")
    print(f"Wrote {out_dir / 'summary.json'}")


def _no_road_config(config: dict) -> dict:
    clean = copy.deepcopy(config)
    shaping = clean.setdefault("env", {}).setdefault("shaping", {})
    shaping["allow_build_road"] = False
    return clean


def _episode_outputs(metrics: dict, group: str, seed: int, episode: int) -> tuple[dict, dict]:
    road_records = metrics.get("road_credit_records", [])
    row = {
        "group": group,
        "seed": seed,
        "episode": episode,
        "policy_type": "oracle" if group == "rule_road_oracle" else "baseline",
        "model_path": "",
        "train_log_dir": "",
        "train_config_path": "",
    }
    for key, value in metrics.items():
        if key in {"map_summary", "road_credit_records"}:
            continue
        row[key] = value
    detail = {
        "group": group,
        "seed": seed,
        "episode": episode,
        "metrics": row,
        "road_credit_records": road_records,
    }
    return row, detail


if __name__ == "__main__":
    main()
