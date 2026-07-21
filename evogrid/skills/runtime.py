"""Restricted Skill DSL runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from evogrid.constants import ACTION_NAMES
from evogrid.skills.context import SkillContext
from evogrid.skills.predicates import evaluate_predicate
from evogrid.skills.schemas import SkillSpec, canonical_json, compute_spec_hash
from evogrid.skills.traces import SkillTrace


Estimator = Callable[[SkillContext, dict[str, Any]], Any]
ActionValidator = Callable[[str, SkillContext], bool]
SkillResolver = Callable[[str, str | None], SkillSpec | None]


@dataclass
class SkillRuntimeResult:
    chosen_action: str | None
    termination: str
    trace: SkillTrace
    variables: dict[str, Any] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.termination == "completed"


@dataclass
class SkillEpisodeState:
    """Mutable per-episode execution state shared across Skill invocations."""

    use_counts: dict[str, int] = field(default_factory=dict)
    stopped_skills: set[str] = field(default_factory=set)

    def use_count(self, spec: SkillSpec) -> int:
        return int(self.use_counts.get(_skill_key(spec), 0))

    def is_stopped(self, spec: SkillSpec) -> bool:
        return _skill_key(spec) in self.stopped_skills

    def record_success(self, spec: SkillSpec, *, stop_after_success: bool = False) -> None:
        key = _skill_key(spec)
        self.use_counts[key] = int(self.use_counts.get(key, 0)) + 1
        if stop_after_success:
            self.stopped_skills.add(key)


class SkillRuntime:
    def __init__(
        self,
        estimators: dict[str, Estimator] | None = None,
        action_validator: ActionValidator | None = None,
        skill_resolver: SkillResolver | None = None,
    ):
        self.estimators = estimators or {}
        self.action_validator = action_validator or _default_action_validator
        self.skill_resolver = skill_resolver

    def execute(
        self,
        spec: SkillSpec,
        context: SkillContext,
        run_id: str = "",
        episode_id: str = "",
        step: int = 0,
        allow_candidate: bool = False,
        max_nested_depth: int = 0,
        episode_state: SkillEpisodeState | None = None,
        _call_stack: tuple[str, ...] = (),
    ) -> SkillRuntimeResult:
        if spec.spec_hash != compute_spec_hash(spec):
            raise ValueError("spec_hash does not match SkillSpec content")
        if spec.status == "candidate" and not allow_candidate:
            return self._result(spec, context, run_id, episode_id, step, False, "candidate_not_allowed")
        if spec.status not in {"verified", "candidate"}:
            return self._result(spec, context, run_id, episode_id, step, False, "status_not_executable")

        predicate = evaluate_predicate(spec.applicability, context)
        if not predicate.ok:
            trace = self._trace(spec, context, run_id, episode_id, step, False)
            trace.operations.append({"op": "PREDICATE", "error": predicate.errors[0]})
            trace.termination = "predicate_error"
            return SkillRuntimeResult(None, trace.termination, trace)
        if not predicate.applicable:
            return self._result(spec, context, run_id, episode_id, step, False, "not_applicable")
        if episode_state is not None and episode_state.is_stopped(spec):
            return self._result(spec, context, run_id, episode_id, step, True, "episode_stop_after_success")
        max_uses = int(spec.budget.get("max_uses_per_episode", 0) or 0)
        if episode_state is not None and max_uses and episode_state.use_count(spec) >= max_uses:
            return self._result(spec, context, run_id, episode_id, step, True, "episode_use_limit_reached")

        trace = self._trace(spec, context, run_id, episode_id, step, True)
        state = _RuntimeState(
            variables={},
            operations_used=0,
            environment_actions_used=0,
            max_runtime_steps=int(spec.budget.get("max_runtime_steps", 0) or 0),
            max_environment_actions=int(spec.budget.get("max_environment_actions", 0) or 0),
            max_nested_depth=int(spec.budget.get("max_nested_skill_depth", 0) or 0),
            run_id=run_id,
            episode_id=episode_id,
            step=step,
            allow_candidate=allow_candidate,
            call_stack=(*_call_stack, _skill_key(spec)),
            episode_state=episode_state,
        )
        try:
            action = self._execute_nodes(spec.procedure, context, trace, state, max_nested_depth)
            trace.chosen_action = action
            trace.termination = "completed" if action is not None else "completed_no_action"
            if action is not None and episode_state is not None:
                episode_state.record_success(
                    spec,
                    stop_after_success=bool(spec.budget.get("stop_after_success", False)),
                )
        except RuntimeStop as stop:
            trace.termination = stop.termination
        return SkillRuntimeResult(trace.chosen_action, trace.termination, trace, dict(state.variables))

    def _execute_nodes(
        self,
        nodes: list[dict[str, Any]],
        context: SkillContext,
        trace: SkillTrace,
        state: "_RuntimeState",
        nested_depth: int,
    ) -> str | None:
        for node in nodes:
            action = self._execute_node(node, context, trace, state, nested_depth)
            if action is not None:
                return action
            if trace.termination == "returned":
                return None
        return None

    def _execute_node(
        self,
        node: dict[str, Any],
        context: SkillContext,
        trace: SkillTrace,
        state: "_RuntimeState",
        nested_depth: int,
    ) -> str | None:
        state.consume_operation()
        op = node.get("op")
        trace.operations.append({"op": op, "node": _compact_node(node)})
        if op == "ESTIMATE":
            estimator_name = str(node.get("estimator"))
            estimator = self.estimators.get(estimator_name)
            if estimator is None:
                raise RuntimeStop("unknown_estimator")
            state.variables[str(node.get("store_as"))] = estimator(context, dict(state.variables))
            return None
        if op == "IF":
            branch = node.get("then", []) if _eval_condition(node.get("condition", {}), state.variables) else node.get("else", [])
            return self._execute_nodes(branch, context, trace, state, nested_depth)
        if op == "ACT":
            action = str(node.get("action"))
            if action not in ACTION_NAMES:
                raise RuntimeStop("unknown_action")
            state.consume_action()
            if not self.action_validator(action, context):
                raise RuntimeStop("illegal_action")
            return action
        if op == "RETURN":
            trace.operations[-1]["result"] = node.get("result")
            trace.termination = "returned"
            return None
        if op == "SELECT_TARGET":
            target = _select_target(node, context, state.variables)
            state.variables[str(node.get("store_as", "target"))] = target
            trace.operations[-1]["result"] = target
            return None
        if op == "PLAN_ROUTE":
            route = _plan_route(node, context, state.variables)
            state.variables[str(node.get("store_as", "route"))] = route
            trace.operations[-1]["result"] = route
            return None
        if op == "FOLLOW_ROUTE":
            action = _follow_route(node, state.variables)
            trace.operations[-1]["result"] = action
            if action is None:
                return None
            if action not in ACTION_NAMES:
                raise RuntimeStop("unknown_action")
            state.consume_action()
            if not self.action_validator(action, context):
                raise RuntimeStop("illegal_action")
            return action
        if op == "CALL_SKILL":
            action = self._call_skill(node, context, trace, state, nested_depth)
            trace.operations[-1]["result"] = action
            if action is not None:
                state.consume_action()
            return action
        raise RuntimeStop("unknown_op")

    def _call_skill(
        self,
        node: dict[str, Any],
        context: SkillContext,
        trace: SkillTrace,
        state: "_RuntimeState",
        nested_depth: int,
    ) -> str | None:
        if nested_depth >= state.max_nested_depth:
            raise RuntimeStop("nested_skill_depth_exceeded")
        if self.skill_resolver is None:
            raise RuntimeStop("nested_skill_registry_unavailable")
        skill_id = str(node.get("skill_id"))
        version = None if node.get("version") is None else str(node.get("version"))
        child = self.skill_resolver(skill_id, version)
        if child is None:
            raise RuntimeStop("nested_skill_not_found")
        child_key = _skill_key(child)
        if child_key in state.call_stack:
            raise RuntimeStop("nested_skill_cycle_detected")
        result = self.execute(
            child,
            context,
            run_id=state.run_id,
            episode_id=state.episode_id,
            step=state.step,
            allow_candidate=state.allow_candidate,
            max_nested_depth=nested_depth + 1,
            episode_state=state.episode_state,
            _call_stack=state.call_stack,
        )
        trace.operations[-1]["child_trace"] = result.trace.to_dict()
        if result.chosen_action is None:
            return None
        if result.chosen_action not in ACTION_NAMES:
            raise RuntimeStop("unknown_action")
        if not self.action_validator(result.chosen_action, context):
            raise RuntimeStop("illegal_action")
        return result.chosen_action

    def _trace(self, spec: SkillSpec, context: SkillContext, run_id: str, episode_id: str, step: int, applicable: bool) -> SkillTrace:
        return SkillTrace(
            schema_version=1,
            run_id=run_id,
            episode_id=episode_id,
            step=int(step),
            skill_id=spec.skill_id,
            skill_version=spec.version,
            spec_hash=spec.spec_hash,
            applicable=applicable,
            observable_context_hash=_context_hash(context),
        )

    def _result(
        self,
        spec: SkillSpec,
        context: SkillContext,
        run_id: str,
        episode_id: str,
        step: int,
        applicable: bool,
        termination: str,
    ) -> SkillRuntimeResult:
        trace = self._trace(spec, context, run_id, episode_id, step, applicable)
        trace.termination = termination
        return SkillRuntimeResult(None, termination, trace)


@dataclass
class _RuntimeState:
    variables: dict[str, Any]
    operations_used: int
    environment_actions_used: int
    max_runtime_steps: int
    max_environment_actions: int
    max_nested_depth: int
    run_id: str
    episode_id: str
    step: int
    allow_candidate: bool
    call_stack: tuple[str, ...]
    episode_state: SkillEpisodeState | None

    def consume_operation(self) -> None:
        self.operations_used += 1
        if self.max_runtime_steps and self.operations_used > self.max_runtime_steps:
            raise RuntimeStop("runtime_budget_exceeded")

    def consume_action(self) -> None:
        self.environment_actions_used += 1
        if self.max_environment_actions and self.environment_actions_used > self.max_environment_actions:
            raise RuntimeStop("environment_action_budget_exceeded")


class RuntimeStop(Exception):
    def __init__(self, termination: str):
        super().__init__(termination)
        self.termination = termination


def _eval_condition(condition: dict[str, Any], variables: dict[str, Any]) -> bool:
    left = _value(condition.get("left"), variables)
    right = _value(condition.get("right"), variables)
    op = condition.get("op")
    if op == "eq":
        return left == right
    if op == "ne":
        return left != right
    if op == "lt":
        return left < right
    if op == "lte":
        return left <= right
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    raise RuntimeStop("invalid_condition")


def _value(node: Any, variables: dict[str, Any]) -> Any:
    if isinstance(node, dict) and "var" in node:
        key = str(node["var"])
        if key not in variables:
            raise RuntimeStop("missing_variable")
        return variables[key]
    return node


def _select_target(node: dict[str, Any], context: SkillContext, variables: dict[str, Any]) -> dict[str, Any]:
    source = str(node.get("source", "visible_tiles"))
    strategy = str(node.get("strategy", "nearest"))
    if source == "memory":
        memory_key = str(node.get("memory_key", "targets"))
        candidates = context.memory_summary.get(memory_key, [])
    elif source == "variable":
        candidates = _value({"var": str(node.get("var"))}, variables)
    else:
        candidates = context.observation.get("visible_tiles", [])
    if not isinstance(candidates, list):
        raise RuntimeStop("target_source_not_list")
    filtered = [_normalize_target(item) for item in candidates if _target_matches(item, node)]
    if not filtered:
        if "default" in node:
            return _normalize_target(node["default"])
        raise RuntimeStop("target_not_found")
    agent_pos = _position(context.observation.get("agent_pos"))
    if strategy == "first":
        return filtered[0]
    if strategy == "farthest":
        return max(filtered, key=lambda item: _manhattan(agent_pos, _position(item.get("pos"))))
    if strategy == "nearest":
        return min(filtered, key=lambda item: _manhattan(agent_pos, _position(item.get("pos"))))
    raise RuntimeStop("unknown_target_strategy")


def _target_matches(item: Any, node: dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    if "tile_types" in node and item.get("tile") not in node["tile_types"]:
        return False
    if "terrain_bands" in node and item.get("terrain_band") not in node["terrain_bands"]:
        return False
    if "tags" in node:
        tags = set(item.get("tags", []))
        if not set(node["tags"]).issubset(tags):
            return False
    return item.get("pos") is not None


def _normalize_target(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise RuntimeStop("invalid_target")
    pos = _position(item.get("pos"))
    target = {key: value for key, value in item.items() if key in {"tile", "terrain_band", "tags", "score"}}
    target["pos"] = list(pos)
    return target


def _plan_route(node: dict[str, Any], context: SkillContext, variables: dict[str, Any]) -> dict[str, Any]:
    if "unknown_cell_policy" not in node:
        raise RuntimeStop("missing_unknown_cell_policy")
    policy = str(node["unknown_cell_policy"])
    if policy not in {"avoid", "allow", "frontier"}:
        raise RuntimeStop("invalid_unknown_cell_policy")
    start = _position(node.get("start", context.observation.get("agent_pos")))
    target_node = node.get("target", {"var": str(node.get("target_var", "target"))})
    target = _value(target_node, variables)
    if not isinstance(target, dict):
        raise RuntimeStop("invalid_route_target")
    goal = _position(target.get("pos"))
    max_length = int(node.get("max_length", node.get("max_steps", 64)) or 0)
    if max_length <= 0:
        raise RuntimeStop("invalid_route_length")
    actions = _manhattan_actions(start, goal)[:max_length]
    return {
        "start": list(start),
        "goal": list(goal),
        "actions": actions,
        "unknown_cell_policy": policy,
        "complete": len(actions) == _manhattan(start, goal),
    }


def _follow_route(node: dict[str, Any], variables: dict[str, Any]) -> str | None:
    if "max_steps" not in node:
        raise RuntimeStop("missing_follow_route_max_steps")
    max_steps = int(node.get("max_steps", 0) or 0)
    if max_steps <= 0:
        raise RuntimeStop("invalid_follow_route_max_steps")
    route = _value({"var": str(node.get("route_var", "route"))}, variables)
    if not isinstance(route, dict):
        raise RuntimeStop("invalid_route")
    actions = route.get("actions", [])
    if not isinstance(actions, list):
        raise RuntimeStop("invalid_route_actions")
    if not actions:
        return None
    return str(actions[0])


def _position(value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise RuntimeStop("invalid_position")
    return (int(value[0]), int(value[1]))


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _manhattan_actions(start: tuple[int, int], goal: tuple[int, int]) -> list[str]:
    actions: list[str] = []
    row, col = start
    goal_row, goal_col = goal
    while row != goal_row:
        if goal_row < row:
            actions.append("MOVE_UP")
            row -= 1
        else:
            actions.append("MOVE_DOWN")
            row += 1
    while col != goal_col:
        if goal_col < col:
            actions.append("MOVE_LEFT")
            col -= 1
        else:
            actions.append("MOVE_RIGHT")
            col += 1
    return actions


def _skill_key(spec: SkillSpec) -> str:
    return f"{spec.skill_id}@{spec.version}@{spec.spec_hash}"


def _default_action_validator(action: str, context: SkillContext) -> bool:
    if action == "BUILD_ROAD":
        return context.feature_root()["current"].get("tile_type") in {0, 4}
    return action in ACTION_NAMES


def _context_hash(context: SkillContext) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(context.feature_root()).encode("utf-8")).hexdigest()


def _compact_node(node: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in node.items() if key not in {"then", "else"}}
