"""DeepSeek-backed LLM agent."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from evogrid.agents.base_agent import BaseAgent
from evogrid.agents.greedy_agent import GreedyAgent
from evogrid.constants import ACTION_IDS, Action
from evogrid.llm.deepseek_client import DeepSeekClient
from evogrid.llm.parser import extract_json_object
from evogrid.llm.planner import decision_to_action_id
from evogrid.llm.prompts import build_action_messages, build_planner_messages
from evogrid.llm.schemas import LLMDecision


class DeepSeekAgent(BaseAgent):
    def __init__(
        self,
        client: DeepSeekClient | None = None,
        mode: str = "planner",
        replan_interval: int = 20,
        temperature: float = 0.2,
        fallback_agent: BaseAgent | None = None,
        mock_responses: Iterable[str] | None = None,
        max_retries: int = 0,
        trace_prompts: bool = False,
        log_llm_calls: bool = False,
        log_prefix: str = "",
    ):
        self.client = client
        self.mode = mode
        self.replan_interval = max(1, int(replan_interval))
        self.temperature = temperature
        self.fallback_agent = fallback_agent or GreedyAgent()
        self.mock_responses = iter(mock_responses) if mock_responses is not None else None
        self.max_retries = max(0, int(max_retries))
        self.trace_prompts = trace_prompts
        self.log_llm_calls = log_llm_calls
        self.log_prefix = log_prefix
        self.trace: list[dict] = []
        self._last_plan_step = -10**9

    def reset(self, seed: int | None = None) -> None:
        self.trace.clear()
        self._last_plan_step = -10**9
        self.fallback_agent.reset(seed)

    def act(self, obs: dict, info: dict) -> int:
        step = int(obs.get("step", 0))
        should_call = self.mode == "action" or (step - self._last_plan_step) >= self.replan_interval
        if not should_call:
            return self._fallback(obs, info, "between_replans")

        messages = (
            build_action_messages(obs, info)
            if self.mode == "action"
            else build_planner_messages(obs, info)
        )
        raw_response = ""
        fallback_used = False
        fallback_reason = ""
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
                    chosen_action = int(action_id)
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
        except Exception as exc:
            fallback_used = True
            fallback_reason = f"llm_error:{exc}"
            chosen_action = self._fallback(obs, info, fallback_reason)
            parsed_decision = {"error": str(exc)}

        self._last_plan_step = step
        trace_entry = {
            "step": step,
            "mode": self.mode,
            "replan_interval": self.replan_interval,
            "raw_response": raw_response,
            "parsed_decision": parsed_decision,
            "chosen_action": chosen_action,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "attempt_count": len(attempts),
            "attempts": attempts,
        }
        self.trace.append(trace_entry)
        if self.trace_prompts:
            self.trace[-1]["messages"] = messages
        if self.log_llm_calls:
            print(self._format_llm_log_line(trace_entry), flush=True)
        return chosen_action

    def _next_response(self, messages: list[dict]) -> str:
        if self.mock_responses is not None:
            try:
                return next(self.mock_responses)
            except StopIteration as exc:
                raise RuntimeError("mock response exhausted") from exc
        if self.client is None:
            self.client = DeepSeekClient()
        return self.client.chat(messages, temperature=self.temperature)

    def _fallback(self, obs: dict, info: dict, reason: str) -> int:
        try:
            return int(self.fallback_agent.act(obs, info))
        except Exception:
            return int(Action.NOOP)

    def _format_llm_log_line(self, trace_entry: dict) -> str:
        parsed = trace_entry.get("parsed_decision", {})
        action_id = int(trace_entry.get("chosen_action", Action.NOOP))
        action_name = ACTION_IDS.get(action_id, str(action_id))
        reason = parsed.get("reason") or trace_entry.get("fallback_reason") or ""
        prefix = f"{self.log_prefix} " if self.log_prefix else ""
        return (
            f"{prefix}llm_call step={trace_entry.get('step')} "
            f"mode={trace_entry.get('mode')} action={action_name} "
            f"fallback={int(bool(trace_entry.get('fallback_used')))} "
            f"attempts={trace_entry.get('attempt_count', 0)} "
            f"reason={_short_text(reason)}"
        )


class DeepSeekStepAgent(DeepSeekAgent):
    def __init__(self, **kwargs):
        super().__init__(mode="action", replan_interval=1, **kwargs)


def _short_text(value: str, limit: int = 120) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
