"""Hybrid LLM planner with rule-based fallback execution."""

from __future__ import annotations

from evogrid.agents.deepseek_agent import DeepSeekAgent
from evogrid.agents.greedy_agent import GreedyAgent
from evogrid.llm.deepseek_client import DeepSeekClient


class HybridAgent(DeepSeekAgent):
    def __init__(
        self,
        client: DeepSeekClient | None = None,
        replan_interval: int = 20,
        mock_responses=None,
        temperature: float = 0.2,
        max_retries: int = 0,
        trace_prompts: bool = False,
        log_llm_calls: bool = False,
        log_prefix: str = "",
    ):
        super().__init__(
            client=client,
            mode="planner",
            replan_interval=replan_interval,
            temperature=temperature,
            fallback_agent=GreedyAgent(),
            mock_responses=mock_responses,
            max_retries=max_retries,
            trace_prompts=trace_prompts,
            log_llm_calls=log_llm_calls,
            log_prefix=log_prefix,
        )
