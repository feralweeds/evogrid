from __future__ import annotations

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evogrid.agents.route_only_agent import RouteOnlyAgent
from evogrid.agents.skill_agent import SkillAgent
from evogrid.evaluation.skill_verifier import bootstrap_mean_ci
from evogrid.skills.registry import SkillRegistry
from evogrid.skills.runtime import SkillRuntime
from evogrid.skills.schemas import SkillSpec
from scripts.run_handcrafted_return_gate_rollout import handcrafted_return_gate_candidate
from scripts.run_handcrafted_route_target_rollout import (
    PRIMARY_METRIC,
    _rollout_stratum,
    _verification_config,
    handcrafted_route_target_candidate,
)
from scripts.run_skill_verification import _load_rollout_env_config, _run_episode


NO_SKILL = "no_skill"
UNGATED_SKILL = "ungated_skill"
RETURN_GATED_SKILL = "return_gated_skill"
GROUPS = [NO_SKILL, UNGATED_SKILL, RETURN_GATED_SKILL]

METRIC_FIELDS = [
    PRIMARY_METRIC,
    "episode_reward",
    "ore_delivered",
    "steps",
    "num_build_road",
    "road_total_usage_count",
    "road_usage_rate",
    "transport_steps_per_ore",
    "num_dig",
    "num_mine",
    "invalid_actions",
]


def run_r5_road_stop_audit(out_dir: str | Path, seeds: list[int] | None = None) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds or range(5300, 5320))
    env_config = _load_rollout_env_config(_verification_config(), {"verification": _verification_config()})
    candidates = {
        UNGATED_SKILL: handcrafted_route_target_candidate(),
        RETURN_GATED_SKILL: handcrafted_return_gate_candidate(),
    }
    _write_candidate_specs(out_dir, candidates)

    rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for group in GROUPS:
            agent = _agent_for_group(group, out_dir, candidates)
            metrics, trace = _run_episode(env_config, agent, seed)
            row = _episode_row(seed, group, metrics, trace)
            rows.append(row)
            for item in trace:
                if item.get("source") == "skill":
                    trace_rows.append({"seed": seed, "group": group, **item})

    group_summaries = summarize_groups(rows)
    comparisons = compare_groups(rows)
    no_harm = task_no_harm_check(comparisons)
    decision = stop_audit_decision(group_summaries, comparisons, no_harm)

    metrics_path = out_dir / "episodes" / "metrics.csv"
    pairwise_path = out_dir / "comparisons" / "pairwise_summary.csv"
    trace_path = out_dir / "skills" / "skill_trace.jsonl"
    _write_metrics_csv(metrics_path, rows)
    _write_pairwise_csv(pairwise_path, comparisons)
    _write_trace_jsonl(trace_path, trace_rows)

    manifest = {
        "schema_version": 1,
        "experiment_type": "r5_road_stop_audit",
        "mode": "development_stop_audit",
        "question": "On the same 5300-5319 seeds, does return gating improve the route-target road Skill without harming task performance?",
        "frozen_boundaries": [
            "Skill Runtime",
            "Skill DSL",
            "environment config",
            "Verifier protocol",
            "route target selection procedure",
        ],
        "groups": GROUPS,
        "paired_seeds": seeds,
        "candidate_specs": {
            group: str((out_dir / "candidates" / f"{group}.json").as_posix())
            for group in candidates
        },
        "group_summaries": group_summaries,
        "comparisons": comparisons,
        "task_no_harm_check": no_harm,
        "decision": decision,
        "outputs": {
            "metrics_csv": str(metrics_path.as_posix()),
            "pairwise_summary_csv": str(pairwise_path.as_posix()),
            "skill_trace_jsonl": str(trace_path.as_posix()),
            "audit_report": str((out_dir / "audit_report.md").as_posix()),
        },
        "scope": "same-seed development audit; not formal independent verification and not a final road Skill claim",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "audit_report.md").write_text(_audit_report(manifest), encoding="utf-8")
    return manifest


def summarize_groups(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        group_rows = [row for row in rows if row["group"] == group]
        build_rows = [row for row in group_rows if int(row.get("num_build_road", 0) or 0) > 0]
        positive_build_rows = [row for row in build_rows if float(row.get(PRIMARY_METRIC, 0.0) or 0.0) > 0.0]
        nonpositive_build_rows = [row for row in build_rows if float(row.get(PRIMARY_METRIC, 0.0) or 0.0) <= 0.0]
        road_build_count = sum(int(row.get("num_build_road", 0) or 0) for row in group_rows)
        summaries[group] = {
            "episode_count": len(group_rows),
            "road_build_count": road_build_count,
            "build_episode_count": len(build_rows),
            "positive_build_episode_count": len(positive_build_rows),
            "nonpositive_build_episode_count": len(nonpositive_build_rows),
            "positive_build_episode_ratio": len(positive_build_rows) / len(build_rows) if build_rows else 0.0,
            "road_net_sum": _sum(group_rows, PRIMARY_METRIC),
            "road_net_mean": _mean(group_rows, PRIMARY_METRIC),
            "road_total_usage_count": sum(int(row.get("road_total_usage_count", 0) or 0) for row in group_rows),
            "episode_reward_mean": _mean(group_rows, "episode_reward"),
            "ore_delivered_mean": _mean(group_rows, "ore_delivered"),
            "steps_mean": _mean(group_rows, "steps"),
            "activation_rate_mean": _mean(group_rows, "activation_rate"),
            "false_trigger_rate_mean": _mean(group_rows, "false_trigger_rate"),
        }
    return summaries


def compare_groups(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_group_seed = {(row["group"], int(row["seed"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    pairs = [
        (NO_SKILL, UNGATED_SKILL),
        (NO_SKILL, RETURN_GATED_SKILL),
        (UNGATED_SKILL, RETURN_GATED_SKILL),
    ]
    comparisons: dict[str, dict[str, Any]] = {}
    for base, treatment in pairs:
        key = f"{treatment}_minus_{base}"
        metric_summaries: dict[str, Any] = {}
        for metric in METRIC_FIELDS:
            deltas = [
                float(by_group_seed[(treatment, seed)].get(metric, 0.0) or 0.0)
                - float(by_group_seed[(base, seed)].get(metric, 0.0) or 0.0)
                for seed in seeds
                if (treatment, seed) in by_group_seed and (base, seed) in by_group_seed
            ]
            metric_summaries[metric] = _delta_summary(deltas)
        comparisons[key] = {
            "base": base,
            "treatment": treatment,
            "paired_seed_count": len(seeds),
            "metrics": metric_summaries,
        }
    return comparisons


def task_no_harm_check(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gated_vs_ungated = comparisons[f"{RETURN_GATED_SKILL}_minus_{UNGATED_SKILL}"]["metrics"]
    road_net = gated_vs_ungated[PRIMARY_METRIC]
    reward = gated_vs_ungated["episode_reward"]
    ore = gated_vs_ungated["ore_delivered"]
    invalid = gated_vs_ungated["invalid_actions"]
    reward_significant_decline = reward["bootstrap_ci"]["high"] < 0.0
    ore_significant_decline = ore["bootstrap_ci"]["high"] < 0.0
    invalid_increase = invalid["mean_delta"] > 0.0
    return {
        "road_net_improved_vs_ungated": road_net["mean_delta"] > 0.0,
        "episode_reward_not_significantly_down": not reward_significant_decline,
        "ore_delivered_not_significantly_down": not ore_significant_decline,
        "invalid_actions_not_up": not invalid_increase,
        "episode_reward_mean_delta": reward["mean_delta"],
        "episode_reward_bootstrap_ci": reward["bootstrap_ci"],
        "ore_delivered_mean_delta": ore["mean_delta"],
        "ore_delivered_bootstrap_ci": ore["bootstrap_ci"],
        "invalid_actions_mean_delta": invalid["mean_delta"],
        "invalid_actions_bootstrap_ci": invalid["bootstrap_ci"],
        "road_net_mean_delta": road_net["mean_delta"],
        "road_net_bootstrap_ci": road_net["bootstrap_ci"],
        "passed": (
            road_net["mean_delta"] > 0.0
            and not reward_significant_decline
            and not ore_significant_decline
            and not invalid_increase
        ),
    }


def stop_audit_decision(
    group_summaries: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    no_harm: dict[str, Any],
) -> dict[str, Any]:
    ungated = group_summaries[UNGATED_SKILL]
    gated = group_summaries[RETURN_GATED_SKILL]
    gated_vs_ungated = comparisons[f"{RETURN_GATED_SKILL}_minus_{UNGATED_SKILL}"]["metrics"]
    clearly_reduces_bad_builds = (
        gated["nonpositive_build_episode_count"] < ungated["nonpositive_build_episode_count"]
        and gated["positive_build_episode_ratio"] > ungated["positive_build_episode_ratio"]
    )
    keeps_enough_activation = gated["build_episode_count"] > 0 and gated["activation_rate_mean"] >= 0.05
    freeze_baseline = bool(no_harm["passed"] and clearly_reduces_bad_builds and keeps_enough_activation)
    return {
        "freeze_handcrafted_return_gated_road_baseline": freeze_baseline,
        "label_if_frozen": "development_verified",
        "do_not_claim": "formally_verified",
        "next_step": (
            "Run one constrained DeepSeek return-gate regression, then stop the road line and move to DIG."
            if freeze_baseline
            else "Do not spend another road rule-edit round; record instability or no attribution and move to DIG."
        ),
        "evidence": {
            "gated_minus_ungated_road_net_mean_delta": gated_vs_ungated[PRIMARY_METRIC]["mean_delta"],
            "ungated_nonpositive_build_episode_count": ungated["nonpositive_build_episode_count"],
            "gated_nonpositive_build_episode_count": gated["nonpositive_build_episode_count"],
            "ungated_positive_build_episode_ratio": ungated["positive_build_episode_ratio"],
            "gated_positive_build_episode_ratio": gated["positive_build_episode_ratio"],
            "task_no_harm_passed": no_harm["passed"],
        },
    }


def _agent_for_group(group: str, out_dir: Path, candidates: dict[str, SkillSpec]):
    fallback = RouteOnlyAgent()
    if group == NO_SKILL:
        return fallback
    registry = SkillRegistry(out_dir / "registries" / group)
    registry.register_candidate(candidates[group])
    return SkillAgent(registry, fallback, runtime=SkillRuntime(), allow_candidates=True)


def _episode_row(seed: int, group: str, metrics: dict[str, Any], trace: list[dict[str, Any]]) -> dict[str, Any]:
    steps = max(1, int(metrics.get("steps", 0) or 0))
    skill_traces = [item for item in trace if item.get("source") == "skill"]
    runtime_failures = [
        item
        for item in skill_traces
        if item.get("runtime", {}).get("termination")
        not in {
            "completed",
            "completed_no_action",
            "returned",
            "not_applicable",
            "episode_use_limit_reached",
            "episode_stop_after_success",
        }
    ]
    road_builds = int(metrics.get("num_build_road", 0) or 0)
    road_net = float(metrics.get(PRIMARY_METRIC, 0.0) or 0.0)
    return {
        "seed": int(seed),
        "group": group,
        "stratum": _rollout_stratum(metrics),
        PRIMARY_METRIC: road_net,
        "episode_reward": float(metrics.get("episode_reward", 0.0) or 0.0),
        "ore_delivered": int(metrics.get("ore_delivered", 0) or 0),
        "steps": int(metrics.get("steps", 0) or 0),
        "num_build_road": road_builds,
        "road_total_usage_count": int(metrics.get("road_total_usage_count", 0) or 0),
        "road_usage_rate": float(metrics.get("road_usage_rate", 0.0) or 0.0),
        "transport_steps_per_ore": float(metrics.get("transport_steps_per_ore", 0.0) or 0.0),
        "num_dig": int(metrics.get("num_dig", 0) or 0),
        "num_mine": int(metrics.get("num_mine", 0) or 0),
        "invalid_actions": int(metrics.get("invalid_actions", 0) or 0),
        "activation_rate": 1.0 if skill_traces else 0.0,
        "false_trigger_rate": 1.0 if group != NO_SKILL and road_builds > 0 and road_net <= 0.0 else 0.0,
        "runtime_failure_rate": len(runtime_failures) / max(1, len(skill_traces)),
        "positive_road_payoff_count": int(metrics.get("positive_road_payoff_count", 0) or 0),
        "negative_road_payoff_count": int(metrics.get("negative_road_payoff_count", 0) or 0),
    }


def _write_candidate_specs(out_dir: Path, candidates: dict[str, SkillSpec]) -> None:
    candidate_dir = out_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for group, candidate in candidates.items():
        (candidate_dir / f"{group}.json").write_text(
            json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "group",
        "stratum",
        *METRIC_FIELDS,
        "activation_rate",
        "false_trigger_rate",
        "runtime_failure_rate",
        "positive_road_payoff_count",
        "negative_road_payoff_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_pairwise_csv(path: Path, comparisons: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "comparison",
        "metric",
        "mean_delta",
        "sum_delta",
        "positive_count",
        "zero_count",
        "negative_count",
        "ci_low",
        "ci_high",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for comparison, payload in comparisons.items():
            for metric, summary in payload["metrics"].items():
                writer.writerow(
                    {
                        "comparison": comparison,
                        "metric": metric,
                        "mean_delta": summary["mean_delta"],
                        "sum_delta": summary["sum_delta"],
                        "positive_count": summary["positive_count"],
                        "zero_count": summary["zero_count"],
                        "negative_count": summary["negative_count"],
                        "ci_low": summary["bootstrap_ci"]["low"],
                        "ci_high": summary["bootstrap_ci"]["high"],
                    }
                )


def _write_trace_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _audit_report(manifest: dict[str, Any]) -> str:
    summaries = manifest["group_summaries"]
    no_harm = manifest["task_no_harm_check"]
    decision = manifest["decision"]
    lines = [
        "# WP-R5-STOP Road Stop Audit",
        "",
        "## Question",
        "",
        manifest["question"],
        "",
        "## Group Summary",
        "",
        "| group | builds | positive build episodes | nonpositive build episodes | positive ratio | road net sum | road usage | reward mean | ore mean | steps mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in GROUPS:
        item = summaries[group]
        lines.append(
            "| {group} | {builds} | {pos} | {nonpos} | {ratio:.3f} | {road_net:.3f} | {usage} | {reward:.3f} | {ore:.3f} | {steps:.3f} |".format(
                group=group,
                builds=item["road_build_count"],
                pos=item["positive_build_episode_count"],
                nonpos=item["nonpositive_build_episode_count"],
                ratio=item["positive_build_episode_ratio"],
                road_net=item["road_net_sum"],
                usage=item["road_total_usage_count"],
                reward=item["episode_reward_mean"],
                ore=item["ore_delivered_mean"],
                steps=item["steps_mean"],
            )
        )
    lines.extend(
        [
            "",
            "## Gated vs Ungated",
            "",
            f"- road_net mean delta: {no_harm['road_net_mean_delta']:.4f}",
            f"- episode_reward mean delta: {no_harm['episode_reward_mean_delta']:.4f}",
            f"- ore_delivered mean delta: {no_harm['ore_delivered_mean_delta']:.4f}",
            f"- invalid_actions mean delta: {no_harm['invalid_actions_mean_delta']:.4f}",
            f"- task no-harm check passed: {str(no_harm['passed']).lower()}",
            "",
            "## Decision",
            "",
            f"- freeze handcrafted return-gated road baseline: {str(decision['freeze_handcrafted_return_gated_road_baseline']).lower()}",
            f"- label if frozen: {decision['label_if_frozen']}",
            f"- do not claim: {decision['do_not_claim']}",
            f"- next step: {decision['next_step']}",
            "",
            "## Scope",
            "",
            manifest["scope"],
            "",
        ]
    )
    return "\n".join(lines)


def _delta_summary(deltas: list[float]) -> dict[str, Any]:
    return {
        "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        "sum_delta": sum(deltas),
        "positive_count": sum(1 for value in deltas if value > 0.0),
        "zero_count": sum(1 for value in deltas if value == 0.0),
        "negative_count": sum(1 for value in deltas if value < 0.0),
        "bootstrap_ci": bootstrap_mean_ci(deltas, iterations=300, ci_level=0.95, seed=59),
        "paired_deltas": deltas,
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return _sum(rows, key) / len(rows) if rows else 0.0


def _sum(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0.0) or 0.0) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WP-R5-STOP same-seed road Skill stop audit.")
    parser.add_argument("--out", default="outputs/r5_road_stop_audit")
    parser.add_argument("--seed-start", type=int, default=5300)
    parser.add_argument("--seed-count", type=int, default=20)
    args = parser.parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    manifest = run_r5_road_stop_audit(args.out, seeds=seeds)
    print(
        json.dumps(
            {
                "run_manifest": str(Path(args.out) / "run_manifest.json"),
                "freeze_baseline": manifest["decision"]["freeze_handcrafted_return_gated_road_baseline"],
                "task_no_harm_passed": manifest["task_no_harm_check"]["passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
