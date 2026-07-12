"""Partial-observation route baseline agents for road-learning ablations."""

from __future__ import annotations

import random
from copy import copy

from evogrid.agents.base_agent import BaseAgent
from evogrid.agents.memory import AgentMemory
from evogrid.agents.memory_route_planner import MemoryMapRoutePlanner, RoutePlan
from evogrid.agents.road_evidence import learned_road_evidence_gate
from evogrid.agents.shaping_opportunity import ShapingOpportunityBuilder
from evogrid.constants import ACTION_IDS, MOVE_DELTAS, Action, Tile

Position = tuple[int, int]


class RouteOnlyAgent(BaseAgent):
    """Uses memory routing but never builds roads from shaping opportunities."""

    def __init__(
        self,
        memory: AgentMemory | None = None,
        route_planner: MemoryMapRoutePlanner | None = None,
    ):
        self.memory = memory or AgentMemory()
        self.route_planner = route_planner or MemoryMapRoutePlanner()
        self.trace: list[dict] = []
        self._explore_offset = 0

    def reset(self, seed: int | None = None) -> None:
        self.trace.clear()
        self._explore_offset = int(seed or 0) % 4

    def act(self, obs: dict, info: dict) -> int:
        self.memory.update_from_observation(obs)
        action, route_plan = self._route_only_action(obs)
        self._record_trace(obs, action, route_plan)
        return int(action)

    def observe_result(
        self,
        action: int,
        reward: float,
        obs: dict,
        info: dict,
        previous_info: dict | None = None,
    ) -> None:
        self.memory.update_from_result(action, reward, obs, info, previous_info)

    def _route_only_action(self, obs: dict) -> tuple[int, RoutePlan | None]:
        agent_pos = _position(obs["agent_pos"])
        base_pos = _position(obs["base_pos"])
        if obs.get("has_ore") and agent_pos == base_pos:
            return int(Action.DROPOFF), None
        if not obs.get("has_ore") and self._can_mine(obs):
            return int(Action.MINE), None

        target = base_pos if obs.get("has_ore") else self._nearest_known_ore(agent_pos)
        if target is not None:
            plan = self.route_planner.plan_next_action(
                obs=obs,
                memory=self.memory,
                target=target,
                allow_dig=True,
                allow_unknown=True,
            )
            if plan is not None and self._is_legal(plan.action_id, obs):
                return int(plan.action_id), plan

        return self._least_visited_move(obs) or int(Action.NOOP), None

    def _record_trace(self, obs: dict, action: int, route_plan: RoutePlan | None) -> None:
        self.trace.append(
            {
                "step": obs.get("step"),
                "agent_pos": obs.get("agent_pos"),
                "action": ACTION_IDS.get(int(action), str(action)),
                "action_id": int(action),
                "route_plan": _route_trace(route_plan),
            }
        )

    def _can_mine(self, obs: dict) -> bool:
        agent_pos = _position(obs["agent_pos"])
        if _visible_tile(obs, agent_pos) == int(Tile.ORE):
            return True
        return self._has_adjacent_tile(obs, Tile.ORE)

    def _has_adjacent_tile(self, obs: dict, tile: Tile) -> bool:
        agent_pos = _position(obs["agent_pos"])
        for delta in MOVE_DELTAS.values():
            if _visible_tile(obs, _add(agent_pos, delta)) == int(tile):
                return True
        return False

    def _nearest_known_ore(self, agent_pos: Position) -> Position | None:
        if not self.memory.seen_ore_locations:
            return None
        return min(self.memory.seen_ore_locations, key=lambda pos: _manhattan(agent_pos, pos))

    def _least_visited_move(self, obs: dict) -> int | None:
        legal_moves = self._legal_moves(obs)
        if not legal_moves:
            return None
        agent_pos = _position(obs["agent_pos"])
        preferred = _rotated_moves(self._explore_offset + int(obs.get("step", 0)))
        return min(
            legal_moves,
            key=lambda action_id: (
                self.memory.visited_counts.get(_add(agent_pos, MOVE_DELTAS[Action(action_id)]), 0),
                preferred.index(Action(action_id)),
            ),
        )

    def _legal_moves(self, obs: dict) -> list[int]:
        return [int(action) for action in _rotated_moves(self._explore_offset) if self._is_legal(int(action), obs)]

    def _is_legal(self, action_id: int, obs: dict) -> bool:
        try:
            action = Action(action_id)
        except ValueError:
            return False
        if action in MOVE_DELTAS:
            target = _add(_position(obs["agent_pos"]), MOVE_DELTAS[action])
            tile = _visible_tile(obs, target)
            return tile is not None and tile != int(Tile.OBSTACLE)
        if action == Action.MINE:
            return not bool(obs.get("has_ore")) and self._can_mine(obs)
        if action == Action.DIG:
            return self._has_adjacent_tile(obs, Tile.OBSTACLE)
        if action == Action.BUILD_ROAD:
            current_tile = _visible_tile(obs, _position(obs["agent_pos"]))
            return current_tile in {int(Tile.GROUND), int(Tile.ROUGH)}
        if action == Action.DROPOFF:
            return bool(obs.get("has_ore")) and _position(obs["agent_pos"]) == _position(obs["base_pos"])
        if action == Action.NOOP:
            return True
        return False


class LearnedRoadAgent(RouteOnlyAgent):
    """Uses learned road value estimates to decide whether to build a candidate road."""

    def __init__(
        self,
        memory: AgentMemory | None = None,
        route_planner: MemoryMapRoutePlanner | None = None,
        shaping_opportunity_builder: ShapingOpportunityBuilder | None = None,
        learned_value_threshold: float = 0.0,
        confidence_threshold: float = 0.0,
        min_contextual_evidence_count: int = 1,
        positive_rate_threshold: float = 0.0,
        require_contextual_evidence: bool = False,
        require_on_route_learned_build: bool = False,
        require_future_use_break_even: bool = False,
        future_use_margin: int = 0,
    ):
        super().__init__(memory=memory, route_planner=route_planner)
        self.shaping_opportunity_builder = shaping_opportunity_builder or ShapingOpportunityBuilder()
        self.learned_value_threshold = float(learned_value_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.min_contextual_evidence_count = int(min_contextual_evidence_count)
        self.positive_rate_threshold = float(positive_rate_threshold)
        self.require_contextual_evidence = bool(require_contextual_evidence)
        self.require_on_route_learned_build = bool(require_on_route_learned_build)
        self.require_future_use_break_even = bool(require_future_use_break_even)
        self.future_use_margin = int(future_use_margin)

    def act(self, obs: dict, info: dict) -> int:
        self.memory.update_from_observation(obs)
        route_action, route_plan = self._route_only_action(obs)
        opportunity = self.shaping_opportunity_builder.build(
            obs=obs,
            info=info,
            memory=self.memory,
            route_plan=route_plan,
            mode="RETURN_BASE" if obs.get("has_ore") else "GO_TO_ORE",
        )
        action = int(Action.BUILD_ROAD) if self._should_build(opportunity) else route_action
        self._record_learned_trace(obs, action, route_plan, opportunity)
        return int(action)

    def _should_build(self, opportunity: dict) -> bool:
        if not opportunity.get("available"):
            return False
        if opportunity.get("candidate_action") != Action.BUILD_ROAD.name:
            return False
        return self._learned_evidence_gate(opportunity)["passes"]

    def _learned_evidence_gate(self, opportunity: dict) -> dict:
        return learned_road_evidence_gate(
            opportunity=opportunity,
            learned_value_threshold=self.learned_value_threshold,
            confidence_threshold=self.confidence_threshold,
            min_contextual_evidence_count=self.min_contextual_evidence_count,
            positive_rate_threshold=self.positive_rate_threshold,
            require_contextual_evidence=self.require_contextual_evidence,
            require_on_route=self.require_on_route_learned_build,
            require_future_use_break_even=self.require_future_use_break_even,
            future_use_margin=self.future_use_margin,
        )

    def _record_learned_trace(
        self,
        obs: dict,
        action: int,
        route_plan: RoutePlan | None,
        opportunity: dict,
    ) -> None:
        estimate = opportunity.get("learned_estimate", {})
        gate = self._learned_evidence_gate(opportunity)
        self.trace.append(
            {
                "step": obs.get("step"),
                "agent_pos": obs.get("agent_pos"),
                "action": ACTION_IDS.get(int(action), str(action)),
                "action_id": int(action),
                "route_plan": _route_trace(route_plan),
                "shaping_opportunity": opportunity,
                "learned_value": float(estimate.get("learned_value", 0.0) or 0.0),
                "learned_positive": float(estimate.get("learned_value", 0.0) or 0.0) > 0.0,
                "learned_evidence_count": int(estimate.get("evidence_count", 0) or 0),
                "learned_evidence_strong": bool(gate["passes"]),
                "learned_evidence_gate": gate,
            }
        )


class ExplorationRoadAgent(LearnedRoadAgent):
    """Learns road preferences from its own controlled road-building attempts."""

    def __init__(
        self,
        memory: AgentMemory | None = None,
        route_planner: MemoryMapRoutePlanner | None = None,
        shaping_opportunity_builder: ShapingOpportunityBuilder | None = None,
        learned_value_threshold: float = 0.0,
        confidence_threshold: float = 0.0,
        epsilon: float = 0.3,
        uncertainty_epsilon: float = 0.6,
        uncertainty_confidence_threshold: float = 0.2,
        min_saving_per_use: float = 0.0,
        route_only_exploration: bool = True,
        learn_from_current_episode: bool = False,
        max_exploratory_builds_per_episode: int | None = 3,
        max_learned_builds_per_episode: int | None = None,
        min_contextual_evidence_count: int = 1,
        positive_rate_threshold: float = 0.0,
        require_contextual_evidence: bool = False,
        require_on_route_learned_build: bool = False,
        require_future_use_break_even: bool = False,
        future_use_margin: int = 0,
    ):
        super().__init__(
            memory=memory,
            route_planner=route_planner,
            shaping_opportunity_builder=shaping_opportunity_builder,
            learned_value_threshold=learned_value_threshold,
            confidence_threshold=confidence_threshold,
            min_contextual_evidence_count=min_contextual_evidence_count,
            positive_rate_threshold=positive_rate_threshold,
            require_contextual_evidence=require_contextual_evidence,
            require_on_route_learned_build=require_on_route_learned_build,
            require_future_use_break_even=require_future_use_break_even,
            future_use_margin=future_use_margin,
        )
        self.epsilon = float(epsilon)
        self.uncertainty_epsilon = float(uncertainty_epsilon)
        self.uncertainty_confidence_threshold = float(uncertainty_confidence_threshold)
        self.min_saving_per_use = float(min_saving_per_use)
        self.route_only_exploration = bool(route_only_exploration)
        self.learn_from_current_episode = bool(learn_from_current_episode)
        self.max_exploratory_builds_per_episode = max_exploratory_builds_per_episode
        self.max_learned_builds_per_episode = max_learned_builds_per_episode
        self._rng = random.Random(0)
        self._episode_start_record_count = len(self.memory.road_credit_records)
        self._exploratory_builds_this_episode = 0
        self._learned_builds_this_episode = 0

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._rng = random.Random(int(seed or 0))
        self._episode_start_record_count = len(self.memory.road_credit_records)
        self._exploratory_builds_this_episode = 0
        self._learned_builds_this_episode = 0

    def act(self, obs: dict, info: dict) -> int:
        self.memory.update_from_observation(obs)
        route_action, route_plan = self._route_only_action(obs)
        opportunity = self.shaping_opportunity_builder.build(
            obs=obs,
            info=info,
            memory=self._memory_for_opportunity(),
            route_plan=route_plan,
            mode="RETURN_BASE" if obs.get("has_ore") else "GO_TO_ORE",
        )
        decision = self._road_decision(opportunity)
        action = int(Action.BUILD_ROAD) if decision["source"] in {"learned", "exploratory"} else route_action
        if decision["source"] == "exploratory":
            self._exploratory_builds_this_episode += 1
        if decision["source"] == "learned":
            self._learned_builds_this_episode += 1
        self._record_exploration_trace(obs, action, route_plan, opportunity, decision)
        return int(action)

    def _memory_for_opportunity(self) -> AgentMemory:
        if self.learn_from_current_episode:
            return self.memory
        memory_view = copy(self.memory)
        memory_view.road_credit_records = list(self.memory.road_credit_records[: self._episode_start_record_count])
        return memory_view

    def _road_decision(self, opportunity: dict) -> dict:
        if self._should_build(opportunity) and not self._learned_budget_exhausted():
            return {
                "source": "learned",
                "reason": f"{opportunity.get('learned_estimate', {}).get('source', 'learned')}_positive_estimate",
                "roll": None,
                "probability": 1.0,
            }
        if not self._is_exploration_candidate(opportunity):
            return {
                "source": "route",
                "reason": "no_exploration_candidate",
                "roll": None,
                "probability": 0.0,
            }

        probability = self.epsilon
        reason = "epsilon"
        if self._is_uncertain_candidate(opportunity):
            probability = max(probability, self.uncertainty_epsilon)
            reason = "uncertainty"
        roll = self._rng.random()
        if roll < probability:
            return {
                "source": "exploratory",
                "reason": reason,
                "roll": roll,
                "probability": probability,
            }
        return {
            "source": "route",
            "reason": f"{reason}_roll_not_selected",
            "roll": roll,
            "probability": probability,
        }

    def _learned_budget_exhausted(self) -> bool:
        return (
            self.max_learned_builds_per_episode is not None
            and self._learned_builds_this_episode >= self.max_learned_builds_per_episode
        )

    def _is_exploration_candidate(self, opportunity: dict) -> bool:
        if (
            self.max_exploratory_builds_per_episode is not None
            and self._exploratory_builds_this_episode >= self.max_exploratory_builds_per_episode
        ):
            return False
        if not opportunity.get("available"):
            return False
        if opportunity.get("candidate_action") != Action.BUILD_ROAD.name:
            return False
        saving_per_use = float(opportunity.get("cost", {}).get("saving_per_use", 0.0) or 0.0)
        if saving_per_use <= self.min_saving_per_use:
            return False
        if not self.route_only_exploration:
            return True
        return bool(opportunity.get("route_context", {}).get("on_current_route")) or bool(
            opportunity.get("memory_evidence", {}).get("known_as_transport_corridor")
        )

    def _is_uncertain_candidate(self, opportunity: dict) -> bool:
        estimate = opportunity.get("learned_estimate", {})
        confidence = float(estimate.get("confidence", 0.0) or 0.0)
        evidence_count = int(estimate.get("evidence_count", 0) or 0)
        return confidence <= self.uncertainty_confidence_threshold or evidence_count == 0

    def _record_exploration_trace(
        self,
        obs: dict,
        action: int,
        route_plan: RoutePlan | None,
        opportunity: dict,
        decision: dict,
    ) -> None:
        estimate = opportunity.get("learned_estimate", {})
        gate = self._learned_evidence_gate(opportunity)
        self.trace.append(
            {
                "step": obs.get("step"),
                "agent_pos": obs.get("agent_pos"),
                "action": ACTION_IDS.get(int(action), str(action)),
                "action_id": int(action),
                "route_plan": _route_trace(route_plan),
                "shaping_opportunity": opportunity,
                "learned_value": float(estimate.get("learned_value", 0.0) or 0.0),
                "learned_positive": float(estimate.get("learned_value", 0.0) or 0.0) > 0.0,
                "learned_evidence_count": int(estimate.get("evidence_count", 0) or 0),
                "learned_evidence_strong": bool(gate["passes"]),
                "learned_evidence_gate": gate,
                "build_decision_source": decision["source"],
                "build_decision_reason": decision["reason"],
                "exploration_probability": decision["probability"],
                "exploration_roll": decision["roll"],
            }
        )


def _route_trace(route_plan: RoutePlan | None) -> dict:
    if route_plan is None:
        return {"has_route_plan": False}
    return {
        "has_route_plan": True,
        "target_pos": route_plan.target_pos,
        "next_pos": route_plan.next_pos,
        "path_length": len(route_plan.path),
        "planner_mode": route_plan.mode,
    }


def _position(value) -> Position:
    return int(value[0]), int(value[1])


def _add(pos: Position, delta: tuple[int, int]) -> Position:
    return pos[0] + delta[0], pos[1] + delta[1]


def _visible_tile(obs: dict, pos: Position) -> int | None:
    if obs.get("visible_tiles"):
        for item in obs["visible_tiles"]:
            if _position(item["pos"]) == pos:
                return int(item["tile"])
        return None
    if "local_view" in obs:
        origin_row, origin_col = obs.get("local_view_origin", [0, 0])
        row_idx = pos[0] - int(origin_row)
        col_idx = pos[1] - int(origin_col)
        local_view = obs.get("local_view", [])
        if 0 <= row_idx < len(local_view) and 0 <= col_idx < len(local_view[row_idx]):
            value = local_view[row_idx][col_idx]
            return None if value is None else int(value)
        return None
    if "grid" in obs:
        grid = obs["grid"]
        row, col = pos
        if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
            return int(grid[row][col])
    return None


def _rotated_moves(offset: int) -> list[Action]:
    moves = [Action.MOVE_RIGHT, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_UP]
    offset = offset % len(moves)
    return moves[offset:] + moves[:offset]


def _manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])
