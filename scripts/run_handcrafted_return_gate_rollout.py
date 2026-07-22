from __future__ import annotations

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evogrid.evaluation.skill_verifier import SkillVerifier
from evogrid.skills.registry import SkillRegistry
from evogrid.skills.schemas import SkillSpec
from scripts.run_handcrafted_route_target_rollout import (
    _protocol,
    _recording_rollout_evaluator,
    _rollout_summary,
    _verification_config,
    _write_metrics_csv,
    _write_trace_jsonl,
    handcrafted_route_target_candidate,
)


def run_handcrafted_return_gate_rollout(out_dir: str | Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate = handcrafted_return_gate_candidate()
    candidate_path = out_dir / "candidate.json"
    candidate_path.write_text(
        json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    registry = SkillRegistry(out_dir / "registry")
    registry.register_candidate(candidate)
    evaluator, episode_rows, trace_rows = _recording_rollout_evaluator(
        candidate=candidate,
        out_dir=out_dir,
        verification=_verification_config(),
    )
    paired_seeds = list(range(5300, 5320))
    protocol = _protocol()
    report = SkillVerifier(protocol).verify(
        candidate,
        paired_seeds=paired_seeds,
        evaluator=evaluator,
        source_train_seeds=set(),
        environment_strata=["rollout_road_positive", "rollout_road_negative"],
    )
    lease = registry.begin_verification(candidate.skill_id, candidate.version)
    record = registry.apply_verification(report, lease_id=lease.lease_id)
    metrics_path = out_dir / "episodes" / "metrics.csv"
    trace_path = out_dir / "skills" / "skill_trace.jsonl"
    _write_metrics_csv(metrics_path, episode_rows)
    _write_trace_jsonl(trace_path, trace_rows)
    manifest = {
        "schema_version": 1,
        "experiment_type": "handcrafted_return_gate_rollout",
        "hypothesis": "Requiring cargo.has_ore == true reduces premature low-reuse road investment while preserving positive road opportunities.",
        "mode": "pilot",
        "candidate": str(candidate_path.as_posix()),
        "candidate_hash": candidate.spec_hash,
        "only_candidate_change": "Added applicability leaf: cargo.has_ore eq true.",
        "unchanged_boundaries": [
            "Skill Runtime",
            "environment config",
            "Verifier protocol",
            "route target DSL",
        ],
        "decision": report.decision,
        "promoted_status": record.spec.status,
        "report": record.verification_reports[-1].replace("\\", "/") if record.verification_reports else "",
        "paired_seeds": paired_seeds,
        "seed_policy": "new development pilot seeds; not used in the 5200-5219 diagnostic that discovered the return-phase hypothesis",
        "sample_size": report.sample_size,
        "environment_strata": report.environment_strata,
        "metrics": {
            "paired_delta_mean": report.metrics.get("paired_delta_mean", 0.0),
            "success_rate": report.metrics.get("success_rate", 0.0),
            "false_trigger_rate": report.metrics.get("false_trigger_rate", 0.0),
            "activation_rate": report.metrics.get("activation_rate", 0.0),
            "runtime_failure_rate": report.metrics.get("runtime_failure_rate", 0.0),
        },
        "rollout_summary": _rollout_summary(episode_rows),
        "outputs": {
            "metrics_csv": str(metrics_path.as_posix()),
            "skill_trace_jsonl": str(trace_path.as_posix()),
            "registry": str((out_dir / "registry").as_posix()),
        },
        "scope": "real EvoGrid rollout hypothesis test; not a final/test conclusion and not a verified road Skill claim",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def handcrafted_return_gate_candidate() -> SkillSpec:
    base = handcrafted_route_target_candidate().to_dict()
    base["version"] = "1.0.1"
    base["description"] = f"{base['description']} Return-phase gated pilot."
    base["source"] = {
        **base["source"],
        "revision_of_spec_hash": base["spec_hash"],
        "hypothesis": "return_phase_gate",
    }
    base["applicability"]["all"].append({"feature": "cargo.has_ore", "op": "eq", "value": True})
    base["rationale"] = "One-factor R5-04A hypothesis test: gate road building to return-to-base phase."
    base.pop("spec_hash", None)
    return SkillSpec.from_dict(base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the R5 return-phase gated road Skill pilot.")
    parser.add_argument("--out", default="outputs/r5_handcrafted_return_gate_rollout")
    args = parser.parse_args()
    manifest = run_handcrafted_return_gate_rollout(args.out)
    print(json.dumps({"run_manifest": str(Path(args.out) / "run_manifest.json"), "decision": manifest["decision"]}))


if __name__ == "__main__":
    main()
