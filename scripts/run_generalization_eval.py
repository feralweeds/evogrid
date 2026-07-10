from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import copy
import csv
import json
import random
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


GROUPS = ["route_only", "random_road", "rough_rule_road", "exploration_road"]

LEARNED_THRESHOLD_PROFILES = {
    "no_threshold": {
        "min_contextual_evidence_count": 1,
        "positive_rate_threshold": 0.0,
        "learned_value_threshold": 0.0,
        "confidence_threshold": 0.0,
        "require_contextual_evidence": False,
        "require_on_route_learned_build": False,
    },
    "loose_threshold": {
        "min_contextual_evidence_count": 2,
        "positive_rate_threshold": 0.5,
        "learned_value_threshold": 0.05,
        "confidence_threshold": 0.2,
        "require_contextual_evidence": True,
        "require_on_route_learned_build": True,
    },
    "balanced_threshold": {
        "min_contextual_evidence_count": 2,
        "positive_rate_threshold": 0.6,
        "learned_value_threshold": 0.1,
        "confidence_threshold": 0.3,
        "require_contextual_evidence": True,
        "require_on_route_learned_build": True,
    },
    "medium_threshold": {
        "min_contextual_evidence_count": 3,
        "positive_rate_threshold": 0.7,
        "learned_value_threshold": 0.2,
        "confidence_threshold": 0.5,
        "require_contextual_evidence": True,
        "require_on_route_learned_build": True,
    },
    "strict_threshold": {
        "min_contextual_evidence_count": 5,
        "positive_rate_threshold": 0.8,
        "learned_value_threshold": 0.3,
        "confidence_threshold": 0.7,
        "require_contextual_evidence": True,
        "require_on_route_learned_build": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train/test generalization evaluation on random maps.")
    parser.add_argument("--env-config", default="configs/env_random_curriculum.yaml")
    parser.add_argument("--deepseek-config", default="configs/deepseek.yaml")
    parser.add_argument("--train-seeds", default="0:20")
    parser.add_argument("--test-seeds", default="1000:1020")
    parser.add_argument("--episodes-per-seed", type=int, default=2)
    parser.add_argument("--train-episodes-per-seed", type=int)
    parser.add_argument("--test-episodes-per-seed", type=int)
    parser.add_argument(
        "--test-context-scenarios",
        nargs="+",
        choices=["default", "positive", "negative", "mixed"],
        default=["default"],
        help="Run the test phase once per controlled-corridor scenario while keeping training unchanged.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--out", default="outputs/runs/generalization_eval")
    parser.add_argument("--groups", nargs="+", default=GROUPS)
    parser.add_argument("--random-road-probability", type=float, default=0.15)
    parser.add_argument("--epsilon-schedule", default="0.30,0.20,0.10,0.05")
    parser.add_argument("--epsilon-phase-length", type=int, default=20)
    parser.add_argument("--uncertainty-epsilon", type=float, default=0.60)
    parser.add_argument("--uncertainty-confidence-threshold", type=float, default=0.20)
    parser.add_argument("--max-exploratory-builds-per-episode", type=int, default=3)
    parser.add_argument("--test-exploration-budget", type=int)
    parser.add_argument("--max-learned-builds-per-episode", type=int, default=9)
    parser.add_argument("--learned-value-threshold", type=float, default=0.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--min-contextual-evidence-count", type=int, default=1)
    parser.add_argument("--positive-rate-threshold", type=float, default=0.0)
    parser.add_argument("--require-contextual-evidence", action="store_true")
    parser.add_argument("--require-on-route-learned-build", action="store_true")
    parser.add_argument("--mock-deepseek", action="store_true")
    parser.add_argument("--require-api", action="store_true")
    parser.add_argument("--skip-api-check", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--trace-prompts", action="store_true")
    parser.add_argument("--quiet-llm-calls", action="store_true")
    args = parser.parse_args()

    base_config = load_yaml(args.env_config)
    if args.max_steps is not None:
        base_config.setdefault("env", {})["max_steps"] = int(args.max_steps)

    train_seeds = _parse_seed_spec(args.train_seeds)
    test_seeds = _parse_seed_spec(args.test_seeds)
    train_episodes_per_seed = (
        int(args.train_episodes_per_seed)
        if args.train_episodes_per_seed is not None
        else int(args.episodes_per_seed)
    )
    test_episodes_per_seed = (
        int(args.test_episodes_per_seed)
        if args.test_episodes_per_seed is not None
        else int(args.episodes_per_seed)
    )
    epsilon_values = _parse_schedule(args.epsilon_schedule)
    use_llm = any(_group_uses_llm(group) for group in args.groups)
    deepseek_config = load_yaml(args.deepseek_config).get("deepseek", {}) if use_llm else {}
    run_config = _build_run_config(args, deepseek_config)
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

    all_rows: list[dict] = []
    summary = {
        "schema_version": 1,
        "config": {
            "env_config": args.env_config,
            "train_seeds": train_seeds,
            "test_seeds": test_seeds,
            "episodes_per_seed": args.episodes_per_seed,
            "train_episodes_per_seed": train_episodes_per_seed,
            "test_episodes_per_seed": test_episodes_per_seed,
            "test_context_scenarios": args.test_context_scenarios,
            "max_steps": args.max_steps,
            "groups": args.groups,
            "random_road_probability": args.random_road_probability,
            "epsilon_schedule": epsilon_values,
            "epsilon_phase_length": args.epsilon_phase_length,
            "uncertainty_epsilon": args.uncertainty_epsilon,
            "uncertainty_confidence_threshold": args.uncertainty_confidence_threshold,
            "max_exploratory_builds_per_episode": args.max_exploratory_builds_per_episode,
            "test_exploration_budget": args.test_exploration_budget,
            "max_learned_builds_per_episode": args.max_learned_builds_per_episode,
            "learned_value_threshold": args.learned_value_threshold,
            "confidence_threshold": args.confidence_threshold,
            "min_contextual_evidence_count": args.min_contextual_evidence_count,
            "positive_rate_threshold": args.positive_rate_threshold,
            "require_contextual_evidence": bool(args.require_contextual_evidence),
            "require_on_route_learned_build": bool(args.require_on_route_learned_build),
            "learned_threshold_profiles": LEARNED_THRESHOLD_PROFILES,
            "mock_deepseek": bool(args.mock_deepseek),
            "deepseek_config": args.deepseek_config if use_llm else None,
            **run_config,
        },
        "preflight": preflight,
        "groups": {},
        "artifacts": {
            "metrics_csv": str(out_dir / "metrics.csv"),
            "summary_json": str(out_dir / "summary.json"),
            "group_comparison_csv": str(out_dir / "group_comparison.csv"),
            "context_comparison_csv": str(out_dir / "context_comparison.csv"),
            "episodes_dir": str(episodes_dir),
            "traces_dir": str(traces_dir),
        },
    }

    for group in args.groups:
        rows = _run_group(
            group=group,
            base_config=base_config,
            train_seeds=train_seeds,
            test_seeds=test_seeds,
            train_episodes_per_seed=train_episodes_per_seed,
            test_episodes_per_seed=test_episodes_per_seed,
            test_context_scenarios=args.test_context_scenarios,
            epsilon_values=epsilon_values,
            epsilon_phase_length=args.epsilon_phase_length,
            random_road_probability=args.random_road_probability,
            uncertainty_epsilon=args.uncertainty_epsilon,
            uncertainty_confidence_threshold=args.uncertainty_confidence_threshold,
            max_exploratory_builds_per_episode=args.max_exploratory_builds_per_episode,
            test_exploration_budget=args.test_exploration_budget,
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
        summary["groups"][group] = {
            "train": _summarize_rows([row for row in rows if row["phase"] == "train"]),
            "test": _summarize_rows([row for row in rows if row["phase"] == "test"]),
        }

    comparison_rows = _group_comparison_rows(summary["groups"])
    context_rows = _context_comparison_rows(all_rows)
    _write_csv(out_dir / "metrics.csv", all_rows)
    _write_csv(out_dir / "group_comparison.csv", comparison_rows)
    _write_csv(out_dir / "context_comparison.csv", context_rows)
    _write_json(out_dir / "summary.json", summary)
    print(f"Wrote {out_dir / 'metrics.csv'}")
    print(f"Wrote {out_dir / 'group_comparison.csv'}")
    print(f"Wrote {out_dir / 'context_comparison.csv'}")
    print(f"Wrote {out_dir / 'summary.json'}")


def _run_group(
    group: str,
    base_config: dict,
    train_seeds: list[int],
    test_seeds: list[int],
    train_episodes_per_seed: int,
    test_episodes_per_seed: int,
    test_context_scenarios: list[str],
    epsilon_values: list[float],
    epsilon_phase_length: int,
    random_road_probability: float,
    uncertainty_epsilon: float,
    uncertainty_confidence_threshold: float,
    max_exploratory_builds_per_episode: int | None,
    test_exploration_budget: int | None,
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
    config = _config_for_group(base_config, group)
    learned_records = AgentMemory()
    rows: list[dict] = []

    rows.extend(
        _run_phase(
            phase="train",
            group=group,
            config=config,
            seeds=train_seeds,
            episodes_per_seed=train_episodes_per_seed,
            base_experience=learned_records,
            update_global_experience=True,
            epsilon_values=epsilon_values,
            epsilon_phase_length=epsilon_phase_length,
            random_road_probability=random_road_probability,
            uncertainty_epsilon=uncertainty_epsilon,
            uncertainty_confidence_threshold=uncertainty_confidence_threshold,
            max_exploratory_builds_per_episode=max_exploratory_builds_per_episode,
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
            episodes_dir=episodes_dir,
            traces_dir=traces_dir,
            context_scenario="train",
        )
    )
    frozen_train_experience = _experience_only_memory(learned_records.road_credit_records)
    for context_scenario in test_context_scenarios:
        rows.extend(
            _run_phase(
                phase="test",
                group=group,
                config=_config_for_test_context(config, context_scenario),
                seeds=test_seeds,
                episodes_per_seed=test_episodes_per_seed,
                base_experience=frozen_train_experience,
                update_global_experience=False,
                epsilon_values=epsilon_values,
                epsilon_phase_length=epsilon_phase_length,
                random_road_probability=random_road_probability,
                uncertainty_epsilon=uncertainty_epsilon,
                uncertainty_confidence_threshold=uncertainty_confidence_threshold,
                max_exploratory_builds_per_episode=(
                    test_exploration_budget
                    if test_exploration_budget is not None
                    else max_exploratory_builds_per_episode
                ),
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
                episodes_dir=episodes_dir,
                traces_dir=traces_dir,
                context_scenario=context_scenario,
            )
        )
    return rows


def _run_phase(
    phase: str,
    group: str,
    config: dict,
    seeds: list[int],
    episodes_per_seed: int,
    base_experience: AgentMemory,
    update_global_experience: bool,
    epsilon_values: list[float],
    epsilon_phase_length: int,
    random_road_probability: float,
    uncertainty_epsilon: float,
    uncertainty_confidence_threshold: float,
    max_exploratory_builds_per_episode: int | None,
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
    context_scenario: str,
) -> list[dict]:
    rows: list[dict] = []
    global_episode = 0
    for map_seed in seeds:
        map_memory = _experience_only_memory(base_experience.road_credit_records)
        for episode_in_seed in range(episodes_per_seed):
            epsilon = _scheduled_epsilon(global_episode, epsilon_values, epsilon_phase_length)
            env = EvoGridMineEnv(copy.deepcopy(config))
            agent_seed = map_seed * 1000 + episode_in_seed
            agent = _agent_for_group(
                group=group,
                memory=map_memory,
                seed=agent_seed,
                epsilon=epsilon,
                random_road_probability=random_road_probability,
                uncertainty_epsilon=uncertainty_epsilon,
                uncertainty_confidence_threshold=uncertainty_confidence_threshold,
                max_exploratory_builds_per_episode=max_exploratory_builds_per_episode,
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
                phase=phase,
                episode=global_episode,
            )
            result = _run_episode(env, agent, map_seed, agent_seed)
            row = _row_from_result(
                result=result,
                phase=phase,
                group=group,
                map_seed=map_seed,
                episode_in_seed=episode_in_seed,
                global_episode=global_episode,
                epsilon=epsilon,
                context_scenario=context_scenario,
            )
            rows.append(row)
            if update_global_experience:
                base_experience.add_road_credit_records(result["metrics"].get("road_credit_records", []))
            output_phase = _output_phase_name(phase, context_scenario)
            _write_json(
                episodes_dir / output_phase / group / f"seed_{map_seed}_episode_{episode_in_seed:03d}.json",
                result,
            )
            _write_json(
                traces_dir / output_phase / group / f"seed_{map_seed}_episode_{episode_in_seed:03d}_trace.json",
                {"trace": result["trace"]},
            )
            print(
                "phase={phase} context={context} group={group} seed={seed} episode={episode} eps={epsilon:.2f} "
                "reward={reward:.2f} ore={ore} roads={roads} llm_calls={llm_calls} "
                "llm_builds={llm_builds} road_net={road_net:.3f}".format(
                    phase=phase,
                    context=context_scenario,
                    group=group,
                    seed=map_seed,
                    episode=episode_in_seed,
                    epsilon=epsilon,
                    reward=float(row["episode_reward"]),
                    ore=int(row["ore_delivered"]),
                    roads=int(row["num_build_road"]),
                    llm_calls=int(row.get("llm_decision_count", 0) or 0),
                    llm_builds=int(row.get("llm_build_road_count", 0) or 0),
                    road_net=float(row["road_net_payoff"]),
                ),
                flush=True,
            )
            global_episode += 1
    return rows


class RandomRoadAgent(RouteOnlyAgent):
    def __init__(self, memory: AgentMemory, probability: float = 0.15, seed: int = 0):
        super().__init__(memory=memory)
        self.probability = float(probability)
        self.rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.rng.seed(seed)

    def act(self, obs: dict, info: dict) -> int:
        self.memory.update_from_observation(obs)
        route_action, route_plan = self._route_only_action(obs)
        if self._is_legal(int(Action.BUILD_ROAD), obs) and self.rng.random() < self.probability:
            action = int(Action.BUILD_ROAD)
            source = "random"
        else:
            action = int(route_action)
            source = "route"
        self._record_policy_trace(obs, action, route_plan, source)
        return action

    def _record_policy_trace(self, obs: dict, action: int, route_plan, source: str) -> None:
        self.trace.append(
            {
                "step": obs.get("step"),
                "agent_pos": obs.get("agent_pos"),
                "action": ACTION_IDS.get(int(action), str(action)),
                "action_id": int(action),
                "route_plan": _route_trace(route_plan),
                "build_decision_source": source if action == int(Action.BUILD_ROAD) else "route",
            }
        )


class RoughRuleRoadAgent(RouteOnlyAgent):
    def act(self, obs: dict, info: dict) -> int:
        self.memory.update_from_observation(obs)
        route_action, route_plan = self._route_only_action(obs)
        if _visible_tile(obs, _position(obs["agent_pos"])) == int(Tile.ROUGH) and self._is_legal(
            int(Action.BUILD_ROAD), obs
        ):
            action = int(Action.BUILD_ROAD)
            source = "rough_rule"
        else:
            action = int(route_action)
            source = "route"
        self.trace.append(
            {
                "step": obs.get("step"),
                "agent_pos": obs.get("agent_pos"),
                "action": ACTION_IDS.get(int(action), str(action)),
                "action_id": int(action),
                "route_plan": _route_trace(route_plan),
                "build_decision_source": source if action == int(Action.BUILD_ROAD) else "route",
            }
        )
        return action


def _policy_group(group: str) -> str:
    if group in LEARNED_THRESHOLD_PROFILES:
        return "llm_with_road_learning"
    for profile_name in LEARNED_THRESHOLD_PROFILES:
        suffix = f"_{profile_name}"
        if group.endswith(suffix):
            return group[: -len(suffix)]
    return group


def _learned_threshold_profile(group: str) -> dict | None:
    if group in LEARNED_THRESHOLD_PROFILES:
        return LEARNED_THRESHOLD_PROFILES[group]
    for profile_name, profile in LEARNED_THRESHOLD_PROFILES.items():
        if group.endswith(f"_{profile_name}"):
            return profile
    return None


def _group_uses_llm(group: str) -> bool:
    return _policy_group(group).startswith("llm_")


def _agent_for_group(
    group: str,
    memory: AgentMemory,
    seed: int,
    epsilon: float,
    random_road_probability: float,
    uncertainty_epsilon: float,
    uncertainty_confidence_threshold: float,
    max_exploratory_builds_per_episode: int | None,
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
    phase: str,
    episode: int,
):
    profile = _learned_threshold_profile(group)
    policy_group = _policy_group(group)
    if profile:
        learned_value_threshold = profile["learned_value_threshold"]
        confidence_threshold = profile["confidence_threshold"]
        min_contextual_evidence_count = profile["min_contextual_evidence_count"]
        positive_rate_threshold = profile["positive_rate_threshold"]
        require_contextual_evidence = profile["require_contextual_evidence"]
        require_on_route_learned_build = profile["require_on_route_learned_build"]

    if policy_group == "route_only":
        return RouteOnlyAgent(memory=memory)
    if policy_group == "random_road":
        return RandomRoadAgent(memory=memory, probability=random_road_probability, seed=seed)
    if policy_group == "rough_rule_road":
        return RoughRuleRoadAgent(memory=memory)
    if policy_group == "exploration_road":
        return ExplorationRoadAgent(
            memory=memory,
            epsilon=epsilon,
            uncertainty_epsilon=uncertainty_epsilon,
            uncertainty_confidence_threshold=uncertainty_confidence_threshold,
            max_exploratory_builds_per_episode=max_exploratory_builds_per_episode,
            max_learned_builds_per_episode=max_learned_builds_per_episode,
            learned_value_threshold=learned_value_threshold,
            confidence_threshold=confidence_threshold,
            min_contextual_evidence_count=min_contextual_evidence_count,
            positive_rate_threshold=positive_rate_threshold,
            require_contextual_evidence=require_contextual_evidence,
            require_on_route_learned_build=require_on_route_learned_build,
        )
    if policy_group == "llm_no_road_learning":
        return LLMRoadLearningAgent(
            client=client,
            memory=memory,
            use_road_learning=False,
            exploration_budget_per_episode=int(max_exploratory_builds_per_episode or 0),
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
            log_prefix=f"[{phase}:{group} episode={episode} seed={seed}]",
        )
    if policy_group == "llm_with_road_learning":
        return LLMRoadLearningAgent(
            client=client,
            memory=memory,
            use_road_learning=True,
            exploration_budget_per_episode=int(max_exploratory_builds_per_episode or 0),
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
            log_prefix=f"[{phase}:{group} episode={episode} seed={seed}]",
        )
    raise ValueError(f"unknown group: {group}")


def _config_for_group(base_config: dict, group: str) -> dict:
    config = copy.deepcopy(base_config)
    shaping = config.setdefault("env", {}).setdefault("shaping", {})
    shaping["allow_build_road"] = group != "route_only"
    return config


def _config_for_test_context(config: dict, context_scenario: str) -> dict:
    if context_scenario == "default":
        return copy.deepcopy(config)
    forced = copy.deepcopy(config)
    corridor = forced.setdefault("env", {}).setdefault("random_map", {}).setdefault("controlled_corridor", {})
    for scenario in ("positive", "mixed", "negative"):
        corridor[f"{scenario}_weight"] = 1.0 if scenario == context_scenario else 0.0
    return forced


def _output_phase_name(phase: str, context_scenario: str) -> str:
    if phase != "test" or context_scenario == "default":
        return phase
    return f"{phase}_{context_scenario}"


def _run_episode(env: EvoGridMineEnv, agent, map_seed: int, agent_seed: int) -> dict:
    obs, info = env.reset(seed=map_seed)
    agent.reset(agent_seed)
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


def _row_from_result(
    result: dict,
    phase: str,
    group: str,
    map_seed: int,
    episode_in_seed: int,
    global_episode: int,
    epsilon: float,
    context_scenario: str,
) -> dict:
    metrics = result["metrics"]
    trace = result["trace"]
    row = {
        "phase": phase,
        "group": group,
        "map_seed": map_seed,
        "episode_in_seed": episode_in_seed,
        "global_episode": global_episode,
        "epsilon": epsilon,
        "context_scenario": context_scenario,
    }
    for key, value in metrics.items():
        if key in {"map_summary", "road_credit_records"}:
            continue
        row[key] = value
    row.update(_road_quality_metrics(metrics, trace))
    row.update(_learned_influence_metrics(trace))
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
    off_route_build_count = max(0, road_count - route_build_count)
    road_net = float(metrics.get("road_net_payoff", 0.0) or 0.0)
    return {
        "positive_road_ratio": positive_count / road_count if road_count else 0.0,
        "avg_payoff_per_road": road_net / road_count if road_count else 0.0,
        "rough_road_ratio": rough_count / road_count if road_count else 0.0,
        "route_road_ratio": route_build_count / road_count if road_count else 0.0,
        "on_route_build_count": route_build_count,
        "off_route_build_count": off_route_build_count,
    }


def _learned_influence_metrics(trace: list[dict]) -> dict:
    seen = []
    positive = []
    nonpositive = []
    strong = []
    weak = []
    build_positive = 0
    build_nonpositive = 0
    build_strong = 0
    build_weak = 0
    for item in trace:
        opportunity = item.get("shaping_opportunity", {})
        estimate = opportunity.get("learned_estimate", {})
        if not opportunity.get("available") or int(estimate.get("evidence_count", 0) or 0) <= 0:
            continue
        seen.append(item)
        if _learned_evidence_strong(item):
            strong.append(item)
            if _trace_action(item) == "BUILD_ROAD":
                build_strong += 1
        else:
            weak.append(item)
            if _trace_action(item) == "BUILD_ROAD":
                build_weak += 1
        if float(estimate.get("learned_value", 0.0) or 0.0) > 0.0:
            positive.append(item)
            if _trace_action(item) == "BUILD_ROAD":
                build_positive += 1
        else:
            nonpositive.append(item)
            if _trace_action(item) == "BUILD_ROAD":
                build_nonpositive += 1
    return {
        "learned_estimate_seen_count": len(seen),
        "learned_estimate_positive_count": len(positive),
        "learned_estimate_nonpositive_count": len(nonpositive),
        "build_when_learned_positive_count": build_positive,
        "build_when_learned_nonpositive_count": build_nonpositive,
        "p_build_given_learned_positive": build_positive / len(positive) if positive else 0.0,
        "p_build_given_learned_nonpositive": build_nonpositive / len(nonpositive) if nonpositive else 0.0,
        "strong_learned_evidence_count": len(strong),
        "weak_learned_evidence_count": len(weak),
        "build_when_strong_learned_evidence_count": build_strong,
        "build_when_weak_learned_evidence_count": build_weak,
        "skip_when_strong_learned_evidence_count": len(strong) - build_strong,
        "skip_when_weak_learned_evidence_count": len(weak) - build_weak,
        "p_build_given_strong_learned_evidence": build_strong / len(strong) if strong else 0.0,
        "p_build_given_weak_learned_evidence": build_weak / len(weak) if weak else 0.0,
    }


def _llm_metrics(trace: list[dict]) -> dict:
    llm_opportunities = [item for item in trace if item.get("mode") == "llm_road_learning"]
    llm_decisions = [item for item in llm_opportunities if item.get("attempt_count", 0)]
    llm_builds = [item for item in llm_decisions if _trace_action(item) == "BUILD_ROAD"]
    positive = []
    nonpositive = []
    strong = []
    weak = []
    build_positive = 0
    build_nonpositive = 0
    build_strong = 0
    build_weak = 0
    for item in llm_decisions:
        opportunity = item.get("shaping_opportunity", {})
        estimate = opportunity.get("learned_estimate", {})
        if not opportunity.get("available") or int(estimate.get("evidence_count", 0) or 0) <= 0:
            continue
        if _learned_evidence_strong(item):
            strong.append(item)
            if _trace_action(item) == "BUILD_ROAD":
                build_strong += 1
        else:
            weak.append(item)
            if _trace_action(item) == "BUILD_ROAD":
                build_weak += 1
        if float(estimate.get("learned_value", 0.0) or 0.0) > 0.0:
            positive.append(item)
            if _trace_action(item) == "BUILD_ROAD":
                build_positive += 1
        else:
            nonpositive.append(item)
            if _trace_action(item) == "BUILD_ROAD":
                build_nonpositive += 1
    strong_opportunities = []
    weak_opportunities = []
    build_strong_opportunities = 0
    build_weak_opportunities = 0
    for item in llm_opportunities:
        opportunity = item.get("shaping_opportunity", {})
        estimate = opportunity.get("learned_estimate", {})
        if not opportunity.get("available") or int(estimate.get("evidence_count", 0) or 0) <= 0:
            continue
        if _learned_evidence_strong(item):
            strong_opportunities.append(item)
            if _trace_action(item) == "BUILD_ROAD":
                build_strong_opportunities += 1
        else:
            weak_opportunities.append(item)
            if _trace_action(item) == "BUILD_ROAD":
                build_weak_opportunities += 1

    return {
        "llm_decision_count": len(llm_decisions),
        "llm_build_road_count": len(llm_builds),
        "llm_exploration_build_count": sum(
            1 for item in llm_builds if item.get("build_decision_source") == "llm_exploration"
        ),
        "llm_learned_build_count": sum(
            1 for item in llm_builds if item.get("build_decision_source") == "llm_learned"
        ),
        "llm_rejected_candidate_count": sum(1 for item in llm_decisions if item.get("llm_rejected_candidate")),
        "llm_fallback_count": sum(1 for item in llm_decisions if item.get("fallback_used")),
        "p_llm_build_given_learned_positive": build_positive / len(positive) if positive else 0.0,
        "p_llm_build_given_learned_nonpositive": build_nonpositive / len(nonpositive) if nonpositive else 0.0,
        "llm_strong_learned_evidence_count": len(strong_opportunities),
        "llm_weak_learned_evidence_count": len(weak_opportunities),
        "llm_build_given_strong_evidence_count": build_strong_opportunities,
        "llm_skip_given_strong_evidence_count": len(strong_opportunities) - build_strong_opportunities,
        "llm_build_given_weak_evidence_count": build_weak_opportunities,
        "llm_skip_given_weak_evidence_count": len(weak_opportunities) - build_weak_opportunities,
        "p_llm_build_given_strong_learned_evidence": (
            build_strong_opportunities / len(strong_opportunities) if strong_opportunities else 0.0
        ),
        "p_llm_build_given_weak_learned_evidence": (
            build_weak_opportunities / len(weak_opportunities) if weak_opportunities else 0.0
        ),
    }


def _build_source_metrics(metrics: dict, trace: list[dict]) -> dict:
    source_by_key = {}
    for item in trace:
        if _trace_action(item) != "BUILD_ROAD":
            continue
        pos = _position(item.get("agent_pos") or item.get("shaping_opportunity", {}).get("position"))
        step = int(item.get("step", 0) or 0)
        source_by_key[(pos, step)] = item.get("build_decision_source") or "unknown"

    sources = ["random", "rough_rule", "exploratory", "learned", "llm_exploration", "llm_learned", "unknown"]
    counts = {f"{source}_build_count": 0 for source in sources}
    counts.update({f"{source}_positive_count": 0 for source in sources})
    for record in metrics.get("road_credit_records", []):
        pos = _position(record.get("position"))
        step = int(record.get("build_step", 0) or 0)
        source = source_by_key.get((pos, step), "unknown")
        if source not in sources:
            source = "unknown"
        counts[f"{source}_build_count"] += 1
        if float(record.get("net_payoff", 0.0) or 0.0) > 0.0:
            counts[f"{source}_positive_count"] += 1

    return {
        **counts,
        **{
            f"{source}_positive_rate": _ratio(
                counts[f"{source}_positive_count"], counts[f"{source}_build_count"]
            )
            for source in sources
        },
    }


def _summarize_rows(rows: list[dict]) -> dict:
    summary = {"episode_count": len(rows), "map_seeds": sorted({row["map_seed"] for row in rows}), "metrics": {}}
    if not rows:
        return summary
    numeric_keys = sorted(
        key
        for key, value in rows[0].items()
        if key not in {"phase", "group", "context_scenario"} and isinstance(value, (int, float, bool))
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


def _group_comparison_rows(group_summary: dict) -> list[dict]:
    rows = []
    keys = [
        "episode_reward",
        "ore_delivered",
        "road_net_payoff",
        "positive_road_ratio",
        "avg_payoff_per_road",
        "p_build_given_learned_positive",
        "p_build_given_learned_nonpositive",
        "p_build_given_strong_learned_evidence",
        "p_build_given_weak_learned_evidence",
        "llm_decision_count",
        "llm_build_road_count",
        "llm_learned_build_count",
        "p_llm_build_given_learned_positive",
        "p_llm_build_given_learned_nonpositive",
        "p_llm_build_given_strong_learned_evidence",
        "p_llm_build_given_weak_learned_evidence",
    ]
    for group, phases in group_summary.items():
        row = {"group": group}
        for phase in ("train", "test"):
            metrics = phases.get(phase, {}).get("metrics", {})
            for key in keys:
                row[f"{phase}_{key}_mean"] = metrics.get(key, {}).get("mean", 0.0)
        row["test_minus_train_reward_gap"] = (
            row["test_episode_reward_mean"] - row["train_episode_reward_mean"]
        )
        row["test_minus_train_road_payoff_gap"] = (
            row["test_road_net_payoff_mean"] - row["train_road_net_payoff_mean"]
        )
        rows.append(row)
    return rows


def _context_comparison_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (
            str(row.get("phase", "")),
            str(row.get("context_scenario", "default")),
            str(row.get("group", "")),
        )
        grouped.setdefault(key, []).append(row)

    comparison = []
    for (phase, context_scenario, group), group_rows in sorted(grouped.items()):
        road_count = _sum_metric(group_rows, "num_build_road")
        positive_count = _sum_metric(group_rows, "positive_road_payoff_count")
        negative_count = _sum_metric(group_rows, "negative_road_payoff_count")
        row = {
            "phase": phase,
            "context_scenario": context_scenario,
            "group": group,
            "episode_count": len(group_rows),
            "episode_reward_mean": _mean_metric(group_rows, "episode_reward"),
            "ore_delivered_mean": _mean_metric(group_rows, "ore_delivered"),
            "road_net_payoff_sum": _sum_metric(group_rows, "road_net_payoff"),
            "road_net_payoff_mean": _mean_metric(group_rows, "road_net_payoff"),
            "num_build_road_sum": road_count,
            "llm_exploration_build_count_sum": _sum_metric(group_rows, "llm_exploration_build_count"),
            "llm_learned_build_count_sum": _sum_metric(group_rows, "llm_learned_build_count"),
            "strong_learned_evidence_count_sum": _sum_metric(group_rows, "strong_learned_evidence_count"),
            "weak_learned_evidence_count_sum": _sum_metric(group_rows, "weak_learned_evidence_count"),
            "on_route_build_count_sum": _sum_metric(group_rows, "on_route_build_count"),
            "off_route_build_count_sum": _sum_metric(group_rows, "off_route_build_count"),
            "positive_road_payoff_count_sum": positive_count,
            "negative_road_payoff_count_sum": negative_count,
            "total_positive_road_ratio": _ratio(int(positive_count), int(road_count)),
            "avg_payoff_per_road": _sum_metric(group_rows, "road_net_payoff") / road_count if road_count else 0.0,
        }
        comparison.append(row)
    return comparison


def _sum_metric(rows: list[dict], key: str) -> float:
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows)


def _mean_metric(rows: list[dict], key: str) -> float:
    return _sum_metric(rows, key) / len(rows) if rows else 0.0


def _experience_only_memory(records: list[dict]) -> AgentMemory:
    memory = AgentMemory()
    memory.add_road_credit_records(records)
    return memory


class MockRoadLearningClient:
    """Deterministic stand-in for LLM-mediated road-learning decisions."""

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
                return _decision(Action.BUILD_ROAD, "learned_road", "positive learned road payoff evidence")
            action_id = int(route_action.get("action_id", int(Action.NOOP)))
            return _decision(Action(action_id), "route", "learned road build budget exhausted")
        if (
            opportunity.get("available")
            and int(exploration.get("exploration_budget_remaining", 0) or 0) > 0
            and float(opportunity.get("cost", {}).get("saving_per_use", 0.0) or 0.0) > 0.0
            and opportunity.get("route_context", {}).get("on_current_route")
        ):
            return _decision(Action.BUILD_ROAD, "explore_road", "sample road payoff under exploration budget")
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


def _route_trace(route_plan) -> dict:
    if route_plan is None:
        return {"has_route_plan": False}
    return {
        "has_route_plan": True,
        "target_pos": route_plan.target_pos,
        "next_pos": route_plan.next_pos,
        "path_length": len(route_plan.path),
        "planner_mode": route_plan.mode,
    }


def _visible_tile(obs: dict, pos: tuple[int, int]) -> int | None:
    for item in obs.get("visible_tiles", []):
        if _position(item["pos"]) == pos:
            return int(item["tile"])
    return None


def _trace_action(item: dict) -> str:
    if "action" in item:
        return str(item["action"])
    return str(item.get("chosen_action_name", ""))


def _learned_evidence_strong(item: dict) -> bool:
    if "learned_evidence_strong" in item:
        return bool(item.get("learned_evidence_strong"))
    return bool((item.get("exploration_state") or {}).get("learned_evidence_strong"))


def _position(value) -> tuple[int, int]:
    return int(value[0]), int(value[1])


def _parse_seed_spec(value: str) -> list[int]:
    text = str(value).strip()
    if ":" in text:
        start_text, end_text = text.split(":", 1)
        start = int(start_text)
        end = int(end_text)
        return list(range(start, end))
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_schedule(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("epsilon schedule cannot be empty")
    return values


def _scheduled_epsilon(episode: int, values: list[float], phase_length: int) -> float:
    phase_length = max(1, int(phase_length))
    index = min(len(values) - 1, int(episode) // phase_length)
    return float(values[index])


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _config_value(value, default):
    if value is None:
        return default
    text = str(value).strip()
    if not text or (text.startswith("${") and text.endswith("}")):
        return default
    return value


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
