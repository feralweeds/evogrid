from __future__ import annotations

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import yaml

from evogrid.agents.route_only_agent import RouteOnlyAgent
from evogrid.agents.skill_agent import SkillAgent
from evogrid.envs.evogrid_mine_env import EvoGridMineEnv
from evogrid.evaluation.skill_verifier import SkillVerifier, SkillVerificationProtocol
from evogrid.skills.registry import SkillRegistry
from evogrid.skills.runtime import SkillRuntime
from evogrid.skills.schemas import SkillSpec

REQUIRED_FORMAL_GATES = {
    "G0_data_integrity",
    "G1_effect",
    "G2_reliability",
    "G3_negative_safety",
    "G4_transfer",
    "G5_non_redundancy",
}
FORMAL_MIN_PAIRED_SEEDS_PER_STRATUM = 30


def run_skill_verification(candidate_path: str | Path, config_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    candidate_path = Path(candidate_path)
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    candidate_data = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    if "spec" in candidate_data:
        candidate_data = candidate_data["spec"]
    candidate = SkillSpec.from_dict(candidate_data)
    verification = config.get("verification", {})
    mode = str(verification.get("mode", "smoke"))
    paired_seeds = [int(seed) for seed in verification.get("paired_seeds", [])]
    environment_strata = [str(item) for item in verification.get("environment_strata", [])]
    protocol = SkillVerificationProtocol(
        protocol_id=str(verification.get("protocol_id", "skill_verification_v1")),
        primary_metric=str(verification.get("primary_metric", "road_net_payoff")),
        direction=str(verification.get("direction", "maximize")),
        min_effect=float(verification.get("min_effect", 0.0)),
        min_success_rate=float(verification.get("min_success_rate", 0.6)),
        max_false_trigger_rate=float(verification.get("max_false_trigger_rate", 0.1)),
        min_paired_samples=int(verification.get("min_paired_samples", 0)),
        bootstrap_iterations=int(verification.get("bootstrap_iterations", 1000)),
        ci_level=float(verification.get("ci_level", 0.95)),
        max_runtime_failure_rate=float(verification.get("max_runtime_failure_rate", 0.02)),
        max_invalid_action_rate_delta=float(verification.get("max_invalid_action_rate_delta", 0.0)),
        min_activation_rate=float(verification.get("min_activation_rate", 0.0)),
        requires_transfer=bool(verification.get("requires_transfer", False)),
        min_transfer_strata=int(verification.get("min_transfer_strata", 2)),
        max_redundancy_score=float(verification.get("max_redundancy_score", 1.0)),
        bootstrap_seed=int(verification.get("bootstrap_seed", 0)),
    )
    evaluator = _build_evaluator(
        str(verification.get("evaluator", "fixture_positive")),
        primary_metric=protocol.primary_metric,
        candidate=candidate,
        verification=verification,
        config=config,
        out_dir=out_dir,
    )
    report = SkillVerifier(protocol).verify(
        candidate,
        paired_seeds=paired_seeds,
        evaluator=evaluator,
        source_train_seeds={int(seed) for seed in verification.get("source_train_seeds", [])},
        environment_strata=environment_strata,
    )
    report_dir = out_dir / "skills" / "reports" / candidate.skill_id / candidate.version
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{report.verification_id}.json"
    report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    report_ref = str(report_path.relative_to(out_dir)).replace("\\", "/")
    formal_readiness = _formal_readiness(
        mode=mode,
        protocol=protocol,
        report=report.to_dict(),
        paired_seeds=paired_seeds,
        environment_strata=environment_strata,
    )
    formal_passed = formal_readiness["passed"] and report.decision == "verified" and not report.failure_reasons
    manifest = {
        "schema_version": 1,
        "run_id": out_dir.name,
        "experiment_type": "skill_verification",
        "mode": mode,
        "candidate": str(candidate_path),
        "config": str(config_path),
        "report": report_ref,
        "decision": report.decision,
        "candidate_hash": candidate.spec_hash,
        "sample_size": len(paired_seeds),
        "environment_strata": environment_strata,
        "formal_readiness": formal_readiness,
        "formal_acceptance": {
            "passed": formal_passed,
            "conclusion_level": "E2" if formal_passed else "E0",
            "gate_report": report_ref,
            "protocol_id": protocol.protocol_id,
        },
        "smoke_not_promoted": mode != "formal",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "completion_status": "completed",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _formal_readiness(
    *,
    mode: str,
    protocol: SkillVerificationProtocol,
    report: dict[str, Any],
    paired_seeds: list[int],
    environment_strata: list[str],
) -> dict[str, Any]:
    failures: list[str] = []
    if mode != "formal":
        failures.append("mode is not formal")
    if protocol.protocol_id != "skill_verification_v1":
        failures.append("unexpected protocol_id")
    if len(set(paired_seeds)) != len(paired_seeds):
        failures.append("paired_seeds are not unique")
    strata_counts = _stratum_counts(report, environment_strata, len(paired_seeds))
    if not strata_counts:
        failures.append("no environment strata recorded")
    underpowered = {
        stratum: count
        for stratum, count in strata_counts.items()
        if count < FORMAL_MIN_PAIRED_SEEDS_PER_STRATUM
    }
    if underpowered:
        failures.append("formal stratum sample size below 30 paired seeds")
    observed_gates = {str(gate.get("gate", gate.get("gate_id", ""))) for gate in report.get("gates", [])}
    missing_gates = sorted(REQUIRED_FORMAL_GATES - observed_gates)
    if missing_gates:
        failures.append("missing formal G0-G5 gates")
    if report.get("protocol_id") != protocol.protocol_id:
        failures.append("report protocol_id does not match config")
    return {
        "schema_version": 1,
        "passed": not failures,
        "failures": failures,
        "required_min_paired_seeds_per_stratum": FORMAL_MIN_PAIRED_SEEDS_PER_STRATUM,
        "strata_counts": strata_counts,
        "underpowered_strata": underpowered,
        "missing_gates": missing_gates,
        "protocol_id": protocol.protocol_id,
    }


def _stratum_counts(report: dict[str, Any], environment_strata: list[str], sample_size: int) -> dict[str, int]:
    by_stratum = report.get("metrics", {}).get("paired_deltas_by_stratum", {})
    if isinstance(by_stratum, dict) and by_stratum:
        return {str(stratum): len(values) for stratum, values in by_stratum.items() if isinstance(values, list)}
    if not environment_strata:
        return {"default": int(sample_size)} if sample_size else {}
    counts = {stratum: 0 for stratum in environment_strata}
    for index in range(sample_size):
        counts[environment_strata[index % len(environment_strata)]] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired Skill verification.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--config", default="configs/skill_verification.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = run_skill_verification(args.candidate, args.config, args.out)
    print(json.dumps({"run_manifest": str(Path(args.out) / "run_manifest.json"), "decision": manifest["decision"]}))


def _build_evaluator(
    name: str,
    *,
    primary_metric: str,
    candidate: SkillSpec,
    verification: dict[str, Any],
    config: dict[str, Any],
    out_dir: Path,
):
    if name == "rollout_route_skill":
        return _rollout_route_skill_evaluator(
            candidate=candidate,
            verification=verification,
            config=config,
            out_dir=out_dir,
            primary_metric=primary_metric,
        )
    return _fixture_evaluator(name, primary_metric)


def _fixture_evaluator(name: str, primary_metric: str):
    def evaluator(seed: int, enabled: bool) -> dict[str, Any]:
        base = float(seed % 10)
        if name == "fixture_negative":
            value = base - (1.0 if enabled else 0.0)
            false_trigger = 0.0
        elif name == "fixture_unsafe":
            value = base + (1.0 if enabled else 0.0)
            false_trigger = 0.5 if enabled else 0.0
        else:
            value = base + (1.0 if enabled else 0.0)
            false_trigger = 0.0
        return {
            primary_metric: value,
            "false_trigger_rate": false_trigger,
            "runtime_failure_rate": 0.0,
            "invalid_action_rate": 0.0,
            "activation_rate": 1.0,
            "redundancy_score": 0.0,
            "stratum": "fixture_positive" if seed % 2 else "fixture_negative",
        }

    return evaluator


def _rollout_route_skill_evaluator(
    *,
    candidate: SkillSpec,
    verification: dict[str, Any],
    config: dict[str, Any],
    out_dir: Path,
    primary_metric: str,
):
    env_config = _load_rollout_env_config(verification, config)
    registry = SkillRegistry(out_dir / "evaluator_registry")
    registry.register_candidate(candidate)
    estimators = _rollout_estimators(verification)
    runtime = SkillRuntime(estimators=estimators, skill_resolver=None)

    def evaluator(seed: int, enabled: bool) -> dict[str, Any]:
        fallback = RouteOnlyAgent()
        agent = (
            SkillAgent(registry, fallback, runtime=runtime, allow_candidates=True)
            if enabled
            else fallback
        )
        metrics, trace = _run_episode(env_config, agent, seed)
        steps = max(1, int(metrics.get("steps", 0) or 0))
        skill_traces = [item for item in trace if item.get("source") == "skill"]
        runtime_failures = [
            item
            for item in skill_traces
            if item.get("runtime", {}).get("termination")
            not in {"completed", "completed_no_action", "returned", "not_applicable"}
        ]
        road_builds = int(metrics.get("num_build_road", 0) or 0)
        road_net_payoff = float(metrics.get(primary_metric, 0.0) or 0.0)
        return {
            primary_metric: road_net_payoff,
            "false_trigger_rate": 1.0 if enabled and road_builds > 0 and road_net_payoff <= 0.0 else 0.0,
            "runtime_failure_rate": len(runtime_failures) / max(1, len(skill_traces)),
            "invalid_action_rate": float(metrics.get("invalid_actions", 0) or 0) / steps,
            "activation_rate": 1.0 if skill_traces else 0.0,
            "redundancy_score": 0.0,
            "stratum": _rollout_stratum(metrics),
            "episode_reward": float(metrics.get("episode_reward", 0.0) or 0.0),
            "ore_delivered": int(metrics.get("ore_delivered", 0) or 0),
            "num_build_road": road_builds,
            "road_usage_rate": float(metrics.get("road_usage_rate", 0.0) or 0.0),
        }

    return evaluator


def _load_rollout_env_config(verification: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(verification.get("env_config"), dict):
        env_config = dict(verification["env_config"])
    else:
        env_config_path = verification.get("env_config_path", config.get("env_config_path"))
        if not env_config_path:
            raise ValueError("rollout_route_skill evaluator requires env_config_path or env_config")
        env_config = yaml.safe_load(Path(str(env_config_path)).read_text(encoding="utf-8-sig"))
    if "episode_max_steps" in verification:
        env_config = dict(env_config)
        env_section = dict(env_config.get("env", env_config))
        env_section["max_steps"] = int(verification["episode_max_steps"])
        if "env" in env_config:
            env_config["env"] = env_section
        else:
            env_config = env_section
    return env_config


def _rollout_estimators(verification: dict[str, Any]) -> dict[str, Any]:
    values = verification.get("estimator_values", {})
    future_route_uses = int(values.get("future_route_uses", 5) if isinstance(values, dict) else 5)
    road_break_even_uses = int(values.get("road_break_even_uses", 3) if isinstance(values, dict) else 3)
    return {
        "future_route_uses": lambda context, variables: future_route_uses,
        "road_break_even_uses": lambda context, variables: road_break_even_uses,
    }


def _run_episode(env_config: dict[str, Any], agent: Any, seed: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = EvoGridMineEnv(env_config)
    obs, info = env.reset(seed=seed)
    if hasattr(agent, "reset"):
        agent.reset(seed)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        previous_info = info
        action = int(agent.act(obs, info))
        obs, reward, terminated, truncated, info = env.step(action)
        observe = getattr(agent, "observe_result", None)
        if observe is not None:
            observe(action, reward, obs, info, previous_info=previous_info)
    metrics = env.get_audit_snapshot()
    trace = list(getattr(agent, "trace", []))
    return metrics, trace


def _rollout_stratum(metrics: dict[str, Any]) -> str:
    if int(metrics.get("positive_road_opportunity_count", 0) or 0) > 0:
        return "rollout_road_positive"
    return "rollout_road_negative"


if __name__ == "__main__":
    main()
