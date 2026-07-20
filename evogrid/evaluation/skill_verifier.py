"""Paired Skill verification helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import random
import uuid
from typing import Any, Callable

from evogrid.skills.schemas import SkillSpec, VerificationReport


PairEvaluator = Callable[[int, bool], dict[str, Any]]


@dataclass(frozen=True)
class SkillVerificationProtocol:
    protocol_id: str = "skill_verification_v1"
    primary_metric: str = "road_net_payoff"
    direction: str = "maximize"
    min_effect: float = 0.0
    min_success_rate: float = 0.6
    max_false_trigger_rate: float = 0.1
    min_paired_samples: int = 0
    bootstrap_iterations: int = 1000
    ci_level: float = 0.95
    max_runtime_failure_rate: float = 0.02
    max_invalid_action_rate_delta: float = 0.0
    min_activation_rate: float = 0.0
    requires_transfer: bool = False
    min_transfer_strata: int = 2
    max_redundancy_score: float = 1.0
    bootstrap_seed: int = 0


@dataclass(frozen=True)
class SkillVerifier:
    protocol: SkillVerificationProtocol = field(default_factory=SkillVerificationProtocol)

    def verify(
        self,
        candidate: SkillSpec,
        paired_seeds: list[int],
        evaluator: PairEvaluator,
        source_train_seeds: set[int] | None = None,
        environment_strata: list[str] | None = None,
    ) -> VerificationReport:
        source_train_seeds = source_train_seeds or set()
        overlap = sorted(set(paired_seeds) & set(source_train_seeds))
        if overlap:
            return self._invalid_report(candidate, paired_seeds, environment_strata or [], f"verify seed leakage: {overlap}")

        disabled_rows = []
        enabled_rows = []
        deltas = []
        deltas_by_stratum: dict[str, list[float]] = {}
        failures = []
        strata = environment_strata or []
        for index, seed in enumerate(paired_seeds):
            disabled = evaluator(seed, False)
            enabled = evaluator(seed, True)
            disabled_rows.append(disabled)
            enabled_rows.append(enabled)
            stratum = str(enabled.get("stratum", disabled.get("stratum", _stratum_for_index(strata, index))))
            try:
                delta = float(enabled[self.protocol.primary_metric]) - float(disabled[self.protocol.primary_metric])
            except KeyError as exc:
                failures.append(f"missing metric {exc}")
                delta = 0.0
            deltas.append(delta)
            deltas_by_stratum.setdefault(stratum, []).append(delta)

        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        positive_rate = sum(1 for delta in deltas if delta > 0.0) / len(deltas) if deltas else 0.0
        false_trigger_rate = _mean(enabled_rows, "false_trigger_rate", default=0.0)
        runtime_failure_rate = _mean(enabled_rows, "runtime_failure_rate", default=0.0)
        enabled_invalid_rate = _mean(enabled_rows, "invalid_action_rate", default=0.0)
        disabled_invalid_rate = _mean(disabled_rows, "invalid_action_rate", default=0.0)
        invalid_action_rate_delta = enabled_invalid_rate - disabled_invalid_rate
        activation_rate = _mean(enabled_rows, "activation_rate", default=1.0)
        redundancy_score = _mean(enabled_rows, "redundancy_score", default=0.0)
        bootstrap_ci = bootstrap_mean_ci(
            deltas,
            iterations=self.protocol.bootstrap_iterations,
            ci_level=self.protocol.ci_level,
            seed=self.protocol.bootstrap_seed,
        )
        sample_count_pass = len(paired_seeds) >= self.protocol.min_paired_samples
        if self.protocol.direction == "maximize":
            effect_pass = (
                mean_delta >= self.protocol.min_effect
                and bootstrap_ci["low"] > 0.0
            )
        else:
            effect_pass = (
                mean_delta <= -self.protocol.min_effect
                and bootstrap_ci["high"] < 0.0
            )
        transfer_summary = _transfer_summary(deltas_by_stratum, self.protocol.direction)
        transfer_pass = (
            not self.protocol.requires_transfer
            or (
                transfer_summary["consistent_strata"] >= self.protocol.min_transfer_strata
                and transfer_summary["direction_consistent"]
            )
        )
        non_redundant = redundancy_score <= self.protocol.max_redundancy_score
        gates = [
            {
                "gate": "G0_data_integrity",
                "passed": not failures and sample_count_pass,
                "details": failures,
                "sample_size": len(paired_seeds),
                "min_paired_samples": self.protocol.min_paired_samples,
            },
            {
                "gate": "G1_effect",
                "passed": effect_pass,
                "mean_delta": mean_delta,
                "min_effect": self.protocol.min_effect,
                "bootstrap_ci": bootstrap_ci,
                "direction": self.protocol.direction,
            },
            {
                "gate": "G2_reliability",
                "passed": (
                    positive_rate >= self.protocol.min_success_rate
                    and runtime_failure_rate <= self.protocol.max_runtime_failure_rate
                    and invalid_action_rate_delta <= self.protocol.max_invalid_action_rate_delta
                ),
                "positive_rate": positive_rate,
                "runtime_failure_rate": runtime_failure_rate,
                "max_runtime_failure_rate": self.protocol.max_runtime_failure_rate,
                "enabled_invalid_action_rate": enabled_invalid_rate,
                "disabled_invalid_action_rate": disabled_invalid_rate,
                "invalid_action_rate_delta": invalid_action_rate_delta,
                "max_invalid_action_rate_delta": self.protocol.max_invalid_action_rate_delta,
            },
            {
                "gate": "G3_negative_safety",
                "passed": (
                    false_trigger_rate <= self.protocol.max_false_trigger_rate
                    and activation_rate >= self.protocol.min_activation_rate
                ),
                "false_trigger_rate": false_trigger_rate,
                "max_false_trigger_rate": self.protocol.max_false_trigger_rate,
                "activation_rate": activation_rate,
                "min_activation_rate": self.protocol.min_activation_rate,
            },
            {
                "gate": "G4_transfer",
                "passed": transfer_pass,
                "requires_transfer": self.protocol.requires_transfer,
                "min_transfer_strata": self.protocol.min_transfer_strata,
                **transfer_summary,
            },
            {
                "gate": "G5_non_redundancy",
                "passed": non_redundant,
                "redundancy_score": redundancy_score,
                "max_redundancy_score": self.protocol.max_redundancy_score,
            },
        ]
        decision = "verified" if all(gate["passed"] for gate in gates) else _failure_decision(gates)
        failure_reasons = [gate["gate"] for gate in gates if not gate["passed"]]
        return VerificationReport.from_dict(
            {
                "schema_version": 1,
                "verification_id": f"verify_{candidate.skill_id}_{uuid.uuid4().hex[:8]}",
                "skill_id": candidate.skill_id,
                "skill_version": candidate.version,
                "spec_hash": candidate.spec_hash,
                "protocol_id": self.protocol.protocol_id,
                "candidate_source_partitions": ["train"],
                "verification_partition": "verify",
                "paired_seeds": paired_seeds,
                "environment_strata": environment_strata or [],
                "baseline": "same_agent_skill_disabled",
                "sample_size": len(paired_seeds),
                "metrics": {
                    "primary_metric": self.protocol.primary_metric,
                    "paired_deltas": deltas,
                    "paired_deltas_by_stratum": deltas_by_stratum,
                    "paired_delta_mean": mean_delta,
                    "paired_delta_bootstrap_ci": bootstrap_ci,
                    "success_rate": positive_rate,
                    "false_trigger_rate": false_trigger_rate,
                    "activation_rate": activation_rate,
                    "runtime_failure_rate": runtime_failure_rate,
                    "enabled_invalid_action_rate": enabled_invalid_rate,
                    "disabled_invalid_action_rate": disabled_invalid_rate,
                    "invalid_action_rate_delta": invalid_action_rate_delta,
                    "redundancy_score": redundancy_score,
                    "transfer": transfer_summary,
                    "disabled": disabled_rows,
                    "enabled": enabled_rows,
                    "bootstrap_seed": self.protocol.bootstrap_seed,
                },
                "gates": gates,
                "decision": decision,
                "failure_reasons": failure_reasons,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _invalid_report(
        self,
        candidate: SkillSpec,
        paired_seeds: list[int],
        environment_strata: list[str],
        reason: str,
    ) -> VerificationReport:
        return VerificationReport.from_dict(
            {
                "schema_version": 1,
                "verification_id": f"verify_{candidate.skill_id}_{uuid.uuid4().hex[:8]}",
                "skill_id": candidate.skill_id,
                "skill_version": candidate.version,
                "spec_hash": candidate.spec_hash,
                "protocol_id": self.protocol.protocol_id,
                "candidate_source_partitions": ["train"],
                "verification_partition": "verify",
                "paired_seeds": paired_seeds,
                "environment_strata": environment_strata,
                "baseline": "same_agent_skill_disabled",
                "sample_size": len(paired_seeds),
                "metrics": {},
                "gates": [{"gate": "G0_data_integrity", "passed": False, "details": [reason]}],
                "decision": "verification_invalid",
                "failure_reasons": [reason],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )


def _mean(rows: list[dict[str, Any]], key: str, default: float) -> float:
    values = [float(row.get(key, default) or 0.0) for row in rows]
    return sum(values) / len(values) if values else default


def _failure_decision(gates: list[dict[str, Any]]) -> str:
    failed = {gate["gate"] for gate in gates if not gate["passed"]}
    if "G0_data_integrity" in failed:
        return "verification_invalid"
    if "G1_effect" in failed:
        return "rejected"
    return "revision_required"


def bootstrap_mean_ci(
    values: list[float],
    *,
    iterations: int,
    ci_level: float,
    seed: int,
) -> dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0, "level": float(ci_level), "iterations": int(iterations)}
    if len(values) == 1 or iterations <= 0:
        mean = sum(values) / len(values)
        return {"low": mean, "high": mean, "level": float(ci_level), "iterations": int(iterations)}
    rng = random.Random(int(seed))
    means = []
    for _ in range(int(iterations)):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    alpha = max(0.0, min(1.0, 1.0 - float(ci_level)))
    low_index = min(len(means) - 1, max(0, int(math.floor((alpha / 2.0) * len(means)))))
    high_index = min(len(means) - 1, max(0, int(math.ceil((1.0 - alpha / 2.0) * len(means))) - 1))
    return {
        "low": means[low_index],
        "high": means[high_index],
        "level": float(ci_level),
        "iterations": int(iterations),
    }


def _stratum_for_index(strata: list[str], index: int) -> str:
    if not strata:
        return "default"
    return strata[index % len(strata)]


def _transfer_summary(deltas_by_stratum: dict[str, list[float]], direction: str) -> dict[str, Any]:
    means = {
        stratum: (sum(values) / len(values) if values else 0.0)
        for stratum, values in sorted(deltas_by_stratum.items())
    }
    if direction == "maximize":
        positive = [stratum for stratum, value in means.items() if value > 0.0]
        direction_consistent = len(positive) == len(means) if means else False
    else:
        positive = [stratum for stratum, value in means.items() if value < 0.0]
        direction_consistent = len(positive) == len(means) if means else False
    return {
        "stratum_mean_deltas": means,
        "consistent_strata": len(positive),
        "direction_consistent": direction_consistent,
    }
