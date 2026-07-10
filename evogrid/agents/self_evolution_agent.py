"""DeepSeek-backed agent for partial-observation self-evolution experiments."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from evogrid.agents.base_agent import BaseAgent
from evogrid.agents.memory import AgentMemory
from evogrid.agents.memory_route_planner import MemoryMapRoutePlanner, RoutePlan
from evogrid.agents.shaping_opportunity import ShapingOpportunityBuilder
from evogrid.constants import ACTION_IDS, MOVE_DELTAS, Action, Tile
from evogrid.llm.deepseek_client import DeepSeekClient
from evogrid.llm.parser import extract_json_object
from evogrid.llm.planner import decision_to_action_id
from evogrid.llm.prompts import build_self_evolution_messages
from evogrid.llm.schemas import LLMDecision

Position = tuple[int, int]


class SelfEvolutionAgent(BaseAgent):
    def __init__(
        self,
        client: DeepSeekClient | None = None,
        memory: AgentMemory | None = None,
        reflection: dict | None = None,
        replan_interval: int = 20,
        temperature: float = 0.2,
        mock_responses: Iterable[str] | None = None,
        max_retries: int = 0,
        trace_prompts: bool = False,
        log_llm_calls: bool = False,
        log_prefix: str = "",
        route_planner: MemoryMapRoutePlanner | None = None,
        shaping_opportunity_builder: ShapingOpportunityBuilder | None = None,
    ):
        self.client = client
        self.memory = memory or AgentMemory()
        self.reflection = reflection or {}
        self.replan_interval = max(1, int(replan_interval))
        self.temperature = temperature
        self.mock_responses = iter(mock_responses) if mock_responses is not None else None
        self.max_retries = max(0, int(max_retries))
        self.trace_prompts = trace_prompts
        self.log_llm_calls = log_llm_calls
        self.log_prefix = log_prefix
        self.route_planner = route_planner or MemoryMapRoutePlanner()
        self.shaping_opportunity_builder = shaping_opportunity_builder or ShapingOpportunityBuilder()
        self.trace: list[dict] = []
        self.route_trace: list[dict] = []
        self._last_plan_step = -10**9
        self._explore_offset = 0

    def reset(self, seed: int | None = None) -> None:
        self.trace.clear()
        self.route_trace.clear()
        self._last_plan_step = -10**9
        self._explore_offset = int(seed or 0) % 4

    def act(self, obs: dict, info: dict) -> int:
        self.memory.update_from_observation(obs)
        step = int(obs.get("step", 0))
        should_call = (step - self._last_plan_step) >= self.replan_interval
        route_plan = self._plan_to_base(obs) if obs.get("has_ore") else None
        shaping_opportunity = self.shaping_opportunity_builder.build(
            obs=obs,
            info=info,
            memory=self.memory,
            route_plan=route_plan,
            mode="RETURN_BASE" if obs.get("has_ore") else "EXPLORE",
        )

        if obs.get("has_ore") and _position(obs.get("agent_pos")) == _position(obs.get("base_pos")):
            return int(Action.DROPOFF)

        if obs.get("has_ore") and (
            not _should_consider_shaping_opportunity(shaping_opportunity) or not should_call
        ):
            route_action = self._route_action_from_plan(obs, route_plan, "return_base_mode")
            if route_action is not None:
                return route_action

        if not should_call:
            return self._fallback(obs, "between_replans")

        messages = build_self_evolution_messages(
            obs,
            info,
            self.memory.summary(),
            self.reflection,
            shaping_opportunity=shaping_opportunity,
        )
        raw_response = ""
        fallback_used = False
        fallback_reason = ""
        legality_filter_used = False
        attempts: list[dict] = []
        parsed_decision = {}
        chosen_action: int | None = None

        try:
            for attempt in range(self.max_retries + 1):
                raw_response = ""
                try:
                    raw_response = self._next_response(messages)
                    parsed = extract_json_object(raw_response)
                    decision = LLMDecision.from_dict(parsed)
                    action_id = decision_to_action_id(decision)
                    if action_id is None:
                        raise ValueError("DeepSeek decision did not contain an executable action.")
                    chosen_action, legality_filter_used, fallback_reason = self._legal_or_fallback(
                        int(action_id),
                        obs,
                    )
                    parsed_decision = asdict(decision)
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "raw_response": raw_response,
                            "parsed_decision": parsed_decision,
                            "error": "",
                        }
                    )
                    break
                except Exception as exc:
                    attempts.append(
                        {
                            "attempt": attempt + 1,
                            "raw_response": raw_response,
                            "parsed_decision": {},
                            "error": str(exc),
                        }
                    )
            if chosen_action is None:
                raise RuntimeError(attempts[-1]["error"] if attempts else "DeepSeek returned no action.")
            fallback_used = legality_filter_used
        except Exception as exc:
            fallback_used = True
            fallback_reason = f"llm_error:{exc}"
            chosen_action = self._fallback(obs, fallback_reason)
            parsed_decision = {"error": str(exc)}

        self._last_plan_step = step
        trace_entry = {
            "step": step,
            "mode": "self_evolution",
            "replan_interval": self.replan_interval,
            "raw_response": raw_response,
            "parsed_decision": parsed_decision,
            "chosen_action": chosen_action,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "legality_filter_used": legality_filter_used,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "memory_summary": self.memory.summary(max_items=5),
            "shaping_opportunity": shaping_opportunity,
        }
        self.trace.append(trace_entry)
        if self.trace_prompts:
            self.trace[-1]["messages"] = messages
        if self.log_llm_calls:
            print(self._format_llm_log_line(trace_entry), flush=True)
        return int(chosen_action)

    def observe_result(
        self,
        action: int,
        reward: float,
        obs: dict,
        info: dict,
        previous_info: dict | None = None,
    ) -> None:
        self.memory.update_from_result(action, reward, obs, info, previous_info)

    def _next_response(self, messages: list[dict]) -> str:
        if self.mock_responses is not None:
            try:
                return next(self.mock_responses)
            except StopIteration as exc:
                raise RuntimeError("mock response exhausted") from exc
        if self.client is None:
            self.client = DeepSeekClient()
        return self.client.chat(messages, temperature=self.temperature)

    def _legal_or_fallback(self, action_id: int, obs: dict) -> tuple[int, bool, str]:
        if self._is_legal(action_id, obs):
            return action_id, False, ""
        return self._fallback(obs, f"illegal_action:{_action_name(action_id)}"), True, (
            f"illegal_action:{_action_name(action_id)}"
        )

    def _fallback(self, obs: dict, reason: str) -> int:
        if obs.get("has_ore") and _position(obs.get("agent_pos")) == _position(obs.get("base_pos")):
            return int(Action.DROPOFF)
        if not obs.get("has_ore") and self._can_mine(obs):
            return int(Action.MINE)
        if obs.get("has_ore"):
            route_action = self._route_to_base(obs, reason)
            if route_action is not None:
                return route_action
            toward_base = self._move_toward_base(obs)
            if toward_base is not None:
                return toward_base
        move = self._least_visited_move(obs)
        if move is not None:
            return move
        return int(Action.NOOP)

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

    def _move_toward_base(self, obs: dict) -> int | None:
        agent_pos = _position(obs["agent_pos"])
        base_pos = _position(obs["base_pos"])
        legal_moves = self._legal_moves(obs)
        if not legal_moves:
            return None
        current_distance = _manhattan(agent_pos, base_pos)
        best_action = None
        best_distance = current_distance
        for action in legal_moves:
            target = _add(agent_pos, MOVE_DELTAS[Action(action)])
            distance = _manhattan(target, base_pos)
            if distance < best_distance:
                best_action = action
                best_distance = distance
        return best_action

    def _least_visited_move(self, obs: dict) -> int | None:
        legal_moves = self._legal_moves(obs)
        if not legal_moves:
            return None
        agent_pos = _position(obs["agent_pos"])
        preferred = _rotated_moves(self._explore_offset + int(obs.get("step", 0)))
        ordered = sorted(
            legal_moves,
            key=lambda action: (
                self.memory.visited_counts.get(_add(agent_pos, MOVE_DELTAS[Action(action)]), 0),
                preferred.index(action) if action in preferred else len(preferred),
            ),
        )
        return ordered[0]

    def _legal_moves(self, obs: dict) -> list[int]:
        return [int(action) for action in _rotated_moves(self._explore_offset) if self._is_legal(int(action), obs)]

    def _route_to_base(self, obs: dict, reason: str) -> int | None:
        return self._route_action_from_plan(obs, self._plan_to_base(obs), reason)

    def _plan_to_base(self, obs: dict) -> RoutePlan | None:
        if _position(obs["agent_pos"]) == _position(obs["base_pos"]):
            return None
        return self.route_planner.plan_next_action(
            obs=obs,
            memory=self.memory,
            target=_position(obs["base_pos"]),
            allow_dig=True,
            allow_unknown=True,
        )

    def _route_action_from_plan(self, obs: dict, plan: RoutePlan | None, reason: str) -> int | None:
        if _position(obs["agent_pos"]) == _position(obs["base_pos"]):
            return int(Action.DROPOFF)
        if plan is None:
            return None
        if not self._is_legal(plan.action_id, obs):
            self._record_route_plan(obs, plan, reason, rejected=True)
            return None
        self._record_route_plan(obs, plan, reason, rejected=False)
        return int(plan.action_id)

    def _record_route_plan(self, obs: dict, plan: RoutePlan, reason: str, rejected: bool) -> None:
        self.route_trace.append(
            {
                "step": obs.get("step"),
                "agent_pos": obs.get("agent_pos"),
                "base_pos": obs.get("base_pos"),
                "has_ore": obs.get("has_ore"),
                "mode": "return_base",
                "trigger_reason": reason,
                "action_id": plan.action_id,
                "action": ACTION_IDS.get(plan.action_id, str(plan.action_id)),
                "planner_mode": plan.mode,
                "next_pos": plan.next_pos,
                "target_pos": plan.target_pos,
                "path_prefix": plan.path[:8],
                "path_length": len(plan.path),
                "cost": plan.cost,
                "reason": plan.reason,
                "rejected": rejected,
            }
        )

    def _format_llm_log_line(self, trace_entry: dict) -> str:
        parsed = trace_entry.get("parsed_decision", {})
        action_id = int(trace_entry.get("chosen_action", Action.NOOP))
        action_name = ACTION_IDS.get(action_id, str(action_id))
        reason = parsed.get("reason") or trace_entry.get("fallback_reason") or ""
        prefix = f"{self.log_prefix} " if self.log_prefix else ""
        return (
            f"{prefix}llm_call step={trace_entry.get('step')} "
            f"mode=self_evolution action={action_name} "
            f"fallback={int(bool(trace_entry.get('fallback_used')))} "
            f"attempts={trace_entry.get('attempt_count', 0)} "
            f"reason={_short_text(reason)}"
        )


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


def _action_name(action_id: int) -> str:
    try:
        return Action(action_id).name
    except ValueError:
        return str(action_id)


def _should_consider_shaping_opportunity(opportunity: dict) -> bool:
    if not opportunity.get("available"):
        return False
    history = opportunity.get("history_stats", {})
    learned = opportunity.get("learned_estimate", {})
    return (
        opportunity.get("current_tile") == "ROUGH"
        or float(history.get("similar_tile_positive_rate", 0.0) or 0.0) > 0.0
        or float(learned.get("learned_value", 0.0) or 0.0) > 0.0
    )


def _short_text(value: str, limit: int = 120) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
