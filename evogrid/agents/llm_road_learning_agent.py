"""LLM-mediated road-learning agent."""

from __future__ import annotations

from copy import copy
from dataclasses import asdict
from typing import Iterable

from evogrid.agents.memory import AgentMemory
from evogrid.agents.road_evidence import learned_road_evidence_gate
from evogrid.agents.route_only_agent import RouteOnlyAgent
from evogrid.agents.shaping_opportunity import ShapingOpportunityBuilder
from evogrid.constants import ACTION_IDS, Action
from evogrid.llm.deepseek_client import DeepSeekClient
from evogrid.llm.parser import extract_json_object
from evogrid.llm.planner import decision_to_action_id
from evogrid.llm.prompts import build_llm_road_learning_messages
from evogrid.llm.schemas import LLMDecision


class LLMRoadLearningAgent(RouteOnlyAgent):
    """Lets an LLM decide whether to build roads from opportunity evidence."""

    def __init__(
        self,
        client: DeepSeekClient | None = None,
        memory: AgentMemory | None = None,
        shaping_opportunity_builder: ShapingOpportunityBuilder | None = None,
        mock_responses: Iterable[str] | None = None,
        temperature: float = 0.2,
        max_retries: int = 0,
        trace_prompts: bool = False,
        log_llm_calls: bool = False,
        log_prefix: str = "",
        use_road_learning: bool = True,
        learn_from_current_episode: bool = False,
        exploration_budget_per_episode: int = 3,
        learned_value_threshold: float = 0.0,
        confidence_threshold: float = 0.0,
        max_learned_builds_per_episode: int | None = None,
        min_contextual_evidence_count: int = 1,
        positive_rate_threshold: float = 0.0,
        require_contextual_evidence: bool = False,
        require_on_route_learned_build: bool = False,
    ):
        super().__init__(memory=memory)
        self.client = client
        self.shaping_opportunity_builder = shaping_opportunity_builder or ShapingOpportunityBuilder()
        self.mock_responses = iter(mock_responses) if mock_responses is not None else None
        self.temperature = float(temperature)
        self.max_retries = max(0, int(max_retries))
        self.trace_prompts = bool(trace_prompts)
        self.log_llm_calls = bool(log_llm_calls)
        self.log_prefix = log_prefix
        self.use_road_learning = bool(use_road_learning)
        self.learn_from_current_episode = bool(learn_from_current_episode)
        self.exploration_budget_per_episode = int(exploration_budget_per_episode)
        self.learned_value_threshold = float(learned_value_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.max_learned_builds_per_episode = max_learned_builds_per_episode
        self.min_contextual_evidence_count = int(min_contextual_evidence_count)
        self.positive_rate_threshold = float(positive_rate_threshold)
        self.require_contextual_evidence = bool(require_contextual_evidence)
        self.require_on_route_learned_build = bool(require_on_route_learned_build)
        self._episode_start_record_count = len(self.memory.road_credit_records)
        self._exploration_builds_this_episode = 0
        self._learned_builds_this_episode = 0

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._episode_start_record_count = len(self.memory.road_credit_records)
        self._exploration_builds_this_episode = 0
        self._learned_builds_this_episode = 0

    def act(self, obs: dict, info: dict) -> int:
        self.memory.update_from_observation(obs)
        route_action, route_plan = self._route_only_action(obs)
        if int(route_action) in {int(Action.MINE), int(Action.DROPOFF)}:
            self._record_route_only_llm_trace(obs, route_action, route_plan, "task_action")
            return int(route_action)

        memory_view = self._memory_for_prompt()
        shaping_opportunity = self.shaping_opportunity_builder.build(
            obs=obs,
            info=info,
            memory=memory_view,
            route_plan=route_plan,
            mode="RETURN_BASE" if obs.get("has_ore") else "GO_TO_ORE",
        )
        if not shaping_opportunity.get("available"):
            self._record_route_only_llm_trace(obs, route_action, route_plan, "no_shaping_opportunity")
            return int(route_action)

        exploration_state = self._exploration_state(shaping_opportunity)
        if not self._should_call_llm(shaping_opportunity, exploration_state):
            self._record_route_only_llm_trace(
                obs,
                route_action,
                route_plan,
                "no_budget_or_learned_signal",
                shaping_opportunity=shaping_opportunity,
                exploration_state=exploration_state,
            )
            return int(route_action)
        messages = build_llm_road_learning_messages(
            obs=obs,
            info=info,
            memory_summary=memory_view.summary(),
            shaping_opportunity=shaping_opportunity,
            route_action_id=int(route_action),
            exploration_state=exploration_state,
        )
        action, trace_entry = self._call_llm_for_action(
            messages=messages,
            obs=obs,
            route_action=int(route_action),
            shaping_opportunity=shaping_opportunity,
            exploration_state=exploration_state,
        )
        if trace_entry["build_decision_source"] == "llm_exploration" and action == int(Action.BUILD_ROAD):
            self._exploration_builds_this_episode += 1
        if trace_entry["build_decision_source"] == "llm_learned" and action == int(Action.BUILD_ROAD):
            self._learned_builds_this_episode += 1
        if self.trace_prompts:
            trace_entry["messages"] = messages
        self.trace.append(trace_entry)
        if self.log_llm_calls:
            print(self._format_llm_log_line(trace_entry), flush=True)
        return int(action)

    def _memory_for_prompt(self) -> AgentMemory:
        memory_view = copy(self.memory)
        if not self.use_road_learning:
            memory_view.road_credit_records = []
            return memory_view
        if not self.learn_from_current_episode:
            memory_view.road_credit_records = list(
                self.memory.road_credit_records[: self._episode_start_record_count]
            )
        return memory_view

    def _exploration_state(self, shaping_opportunity: dict) -> dict:
        remaining = max(0, self.exploration_budget_per_episode - self._exploration_builds_this_episode)
        learned_remaining = None
        if self.max_learned_builds_per_episode is not None:
            learned_remaining = max(0, self.max_learned_builds_per_episode - self._learned_builds_this_episode)
        estimate = shaping_opportunity.get("learned_estimate", {})
        evidence_gate = self._learned_evidence_gate(shaping_opportunity)
        return {
            "exploration_budget_per_episode": self.exploration_budget_per_episode,
            "exploration_builds_used": self._exploration_builds_this_episode,
            "exploration_budget_remaining": remaining,
            "max_learned_builds_per_episode": self.max_learned_builds_per_episode,
            "learned_builds_used": self._learned_builds_this_episode,
            "learned_builds_remaining": learned_remaining,
            "has_tile_specific_evidence": estimate.get("source") == "tile_specific",
            "has_actionable_learned_evidence": estimate.get("source")
            in {"contextual", "contextual_route", "tile_specific"},
            "learned_evidence_strong": bool(evidence_gate["passes"]),
            "learned_evidence_gate": evidence_gate,
            "learned_value_positive": float(estimate.get("learned_value", 0.0) or 0.0)
            > self.learned_value_threshold,
        }

    def _should_call_llm(self, shaping_opportunity: dict, exploration_state: dict) -> bool:
        if exploration_state.get("learned_evidence_strong"):
            return self._learned_budget_remaining(exploration_state)
        if int(exploration_state.get("exploration_budget_remaining", 0) or 0) <= 0:
            return False
        return bool(shaping_opportunity.get("route_context", {}).get("on_current_route"))

    def _learned_evidence_gate(self, shaping_opportunity: dict) -> dict:
        return learned_road_evidence_gate(
            opportunity=shaping_opportunity,
            learned_value_threshold=self.learned_value_threshold,
            confidence_threshold=self.confidence_threshold,
            min_contextual_evidence_count=self.min_contextual_evidence_count,
            positive_rate_threshold=self.positive_rate_threshold,
            require_contextual_evidence=self.require_contextual_evidence,
            require_on_route=self.require_on_route_learned_build,
        )

    def _call_llm_for_action(
        self,
        messages: list[dict],
        obs: dict,
        route_action: int,
        shaping_opportunity: dict,
        exploration_state: dict,
    ) -> tuple[int, dict]:
        raw_response = ""
        attempts: list[dict] = []
        parsed_decision = {}
        fallback_used = False
        fallback_reason = ""
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
                        raise ValueError("LLM decision did not contain an executable action.")
                    parsed_decision = asdict(decision)
                    chosen_action, fallback_used, fallback_reason = self._legal_or_route_action(
                        action_id=int(action_id),
                        route_action=route_action,
                        obs=obs,
                        shaping_opportunity=shaping_opportunity,
                        exploration_state=exploration_state,
                    )
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
                raise RuntimeError(attempts[-1]["error"] if attempts else "LLM returned no action.")
        except Exception as exc:
            fallback_used = True
            fallback_reason = f"llm_error:{exc}"
            chosen_action = route_action
            parsed_decision = {"error": str(exc)}

        source = self._build_source(chosen_action, shaping_opportunity)
        if chosen_action != int(Action.BUILD_ROAD):
            source = "route"
        return int(chosen_action), {
            "step": obs.get("step"),
            "mode": "llm_road_learning",
            "raw_response": raw_response,
            "parsed_decision": parsed_decision,
            "chosen_action": int(chosen_action),
            "chosen_action_name": ACTION_IDS.get(int(chosen_action), str(chosen_action)),
            "route_action": int(route_action),
            "route_action_name": ACTION_IDS.get(int(route_action), str(route_action)),
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "shaping_opportunity": shaping_opportunity,
            "prompt_shaping_opportunity": shaping_opportunity,
            "prompt_learned_estimate": shaping_opportunity.get("learned_estimate", {}),
            "exploration_state": exploration_state,
            "build_decision_source": source,
            "llm_rejected_candidate": bool(
                shaping_opportunity.get("available") and int(chosen_action) != int(Action.BUILD_ROAD)
            ),
        }

    def _next_response(self, messages: list[dict]) -> str:
        if self.mock_responses is not None:
            try:
                return next(self.mock_responses)
            except StopIteration as exc:
                raise RuntimeError("mock response exhausted") from exc
        if self.client is None:
            self.client = DeepSeekClient()
        return self.client.chat(messages, temperature=self.temperature)

    def _legal_or_route_action(
        self,
        action_id: int,
        route_action: int,
        obs: dict,
        shaping_opportunity: dict,
        exploration_state: dict,
    ) -> tuple[int, bool, str]:
        if not self._is_legal(action_id, obs):
            return route_action, True, f"illegal_action:{ACTION_IDS.get(action_id, action_id)}"
        if action_id != int(Action.BUILD_ROAD):
            return int(action_id), False, ""
        if self._build_source(action_id, shaping_opportunity) == "llm_learned":
            if not self._learned_budget_remaining(exploration_state):
                return route_action, True, "learned_build_budget_exhausted"
            return int(action_id), False, ""
        if int(exploration_state.get("exploration_budget_remaining", 0) or 0) <= 0:
            return route_action, True, "exploration_budget_exhausted"
        return int(action_id), False, ""

    def _learned_budget_remaining(self, exploration_state: dict) -> bool:
        remaining = exploration_state.get("learned_builds_remaining")
        return remaining is None or int(remaining) > 0

    def _build_source(self, action_id: int, shaping_opportunity: dict) -> str:
        if int(action_id) != int(Action.BUILD_ROAD):
            return "route"
        estimate = shaping_opportunity.get("learned_estimate", {})
        if (
            self._learned_evidence_gate(shaping_opportunity)["passes"]
        ):
            return "llm_learned"
        return "llm_exploration"

    def _record_route_only_llm_trace(
        self,
        obs: dict,
        action: int,
        route_plan,
        reason: str,
        shaping_opportunity: dict | None = None,
        exploration_state: dict | None = None,
    ) -> None:
        shaping_opportunity = shaping_opportunity or {"available": False, "reason": reason}
        exploration_state = exploration_state or self._exploration_state({"learned_estimate": {}})
        self.trace.append(
            {
                "step": obs.get("step"),
                "mode": "llm_road_learning",
                "chosen_action": int(action),
                "chosen_action_name": ACTION_IDS.get(int(action), str(action)),
                "route_action": int(action),
                "route_action_name": ACTION_IDS.get(int(action), str(action)),
                "fallback_used": False,
                "fallback_reason": reason,
                "attempt_count": 0,
                "attempts": [],
                "shaping_opportunity": shaping_opportunity,
                "prompt_shaping_opportunity": shaping_opportunity,
                "prompt_learned_estimate": shaping_opportunity.get("learned_estimate", {}),
                "exploration_state": exploration_state,
                "build_decision_source": "route",
                "llm_rejected_candidate": False,
                "route_plan": {
                    "has_route_plan": route_plan is not None,
                    "path_length": len(route_plan.path) if route_plan is not None else None,
                },
            }
        )

    def _format_llm_log_line(self, trace_entry: dict) -> str:
        action_name = trace_entry.get("chosen_action_name", "")
        parsed = trace_entry.get("parsed_decision", {})
        reason = parsed.get("reason") or trace_entry.get("fallback_reason") or ""
        prefix = f"{self.log_prefix} " if self.log_prefix else ""
        return (
            f"{prefix}llm_road_call step={trace_entry.get('step')} action={action_name} "
            f"source={trace_entry.get('build_decision_source')} "
            f"fallback={int(bool(trace_entry.get('fallback_used')))} "
            f"reason={_short_text(reason)}"
        )


def _short_text(value: str, limit: int = 120) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
