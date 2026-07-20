"""Predicate DSL evaluation for Skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evogrid.skills.context import SkillContext


ALLOWED_FEATURES = {
    "current.terrain_band",
    "current.tile_type",
    "cargo.has_ore",
    "route.exists",
    "route.is_known_transport_route",
    "route.remaining_length_bucket",
    "memory.visit_count_bucket",
    "memory.similar_outcome_count",
    "memory.similar_mean_payoff",
    "local.adjacent_obstacle_count",
    "local.frontier_count",
    "episode_budget.steps_remaining",
}
ALLOWED_OPS = {"eq", "ne", "lt", "lte", "gt", "gte", "in", "not_in"}


@dataclass(frozen=True)
class PredicateResult:
    applicable: bool
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def evaluate_predicate(predicate: dict[str, Any], context: SkillContext, max_depth: int = 8) -> PredicateResult:
    try:
        applicable = _eval_node(predicate, context.feature_root(), depth=0, max_depth=max_depth)
    except PredicateError as exc:
        return PredicateResult(False, (str(exc),))
    return PredicateResult(bool(applicable), ())


class PredicateError(ValueError):
    pass


def _eval_node(node: dict[str, Any], root: dict[str, Any], depth: int, max_depth: int) -> bool:
    if depth > max_depth:
        raise PredicateError("predicate: maximum nesting depth exceeded")
    if not isinstance(node, dict):
        raise PredicateError("predicate: expected object")
    if "all" in node:
        return all(_eval_node(child, root, depth + 1, max_depth) for child in _list(node["all"], "all"))
    if "any" in node:
        return any(_eval_node(child, root, depth + 1, max_depth) for child in _list(node["any"], "any"))
    if "not" in node:
        return not _eval_node(node["not"], root, depth + 1, max_depth)
    return _eval_leaf(node, root)


def _eval_leaf(node: dict[str, Any], root: dict[str, Any]) -> bool:
    feature = str(node.get("feature"))
    if feature not in ALLOWED_FEATURES:
        raise PredicateError(f"feature not allowed: {feature}")
    op = str(node.get("op"))
    if op not in ALLOWED_OPS:
        raise PredicateError(f"op not allowed: {op}")
    if "value" not in node:
        raise PredicateError("predicate.value: required")
    actual = _resolve_feature(root, feature)
    expected = node["value"]
    return _compare(actual, op, expected)


def _resolve_feature(root: dict[str, Any], feature: str) -> Any:
    current: Any = root
    for part in feature.split("."):
        if not isinstance(current, dict) or part not in current:
            raise PredicateError(f"feature missing: {feature}")
        current = current[part]
    if current is None:
        raise PredicateError(f"feature missing: {feature}")
    return current


def _compare(actual: Any, op: str, expected: Any) -> bool:
    if op == "eq":
        return actual == expected
    if op == "ne":
        return actual != expected
    if op == "lt":
        return actual < expected
    if op == "lte":
        return actual <= expected
    if op == "gt":
        return actual > expected
    if op == "gte":
        return actual >= expected
    if op == "in":
        return actual in expected
    if op == "not_in":
        return actual not in expected
    raise PredicateError(f"op not allowed: {op}")


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise PredicateError(f"{path}: expected list")
    return value
