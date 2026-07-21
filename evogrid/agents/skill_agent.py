"""Agent wrapper that can invoke verified Skills before falling back."""

from __future__ import annotations

from evogrid.agents.base_agent import BaseAgent
from evogrid.constants import ACTION_NAMES, Action
from evogrid.skills.context import SkillContext
from evogrid.skills.registry import SkillRegistry
from evogrid.skills.runtime import SkillEpisodeState, SkillRuntime
from evogrid.skills.selector import SkillSelector


class SkillAgent(BaseAgent):
    def __init__(
        self,
        registry: SkillRegistry,
        fallback_agent: BaseAgent,
        runtime: SkillRuntime | None = None,
        allow_candidates: bool = False,
    ):
        self.registry = registry
        self.fallback_agent = fallback_agent
        self.runtime = runtime or SkillRuntime(skill_resolver=self._resolve_skill)
        if self.runtime.skill_resolver is None:
            self.runtime.skill_resolver = self._resolve_skill
        self.selector = SkillSelector(allow_candidates=allow_candidates)
        self.allow_candidates = allow_candidates
        self.trace: list[dict] = []
        self.episode_state = SkillEpisodeState()

    def reset(self, seed: int | None = None) -> None:
        self.fallback_agent.reset(seed)
        self.trace.clear()
        self.episode_state = SkillEpisodeState()

    def act(self, obs: dict, info: dict) -> int:
        hints = self._context_hints(obs, info)
        context = SkillContext.from_observable_inputs(
            observation=obs,
            info=info,
            memory_summary=hints.get("memory_summary", info.get("memory_summary", {})),
            route_plan=hints.get("route_plan", info.get("route_plan")),
            episode_budget={"steps_remaining": info.get("steps_remaining")},
        )
        records = self.registry.list_available(status="verified")
        if self.allow_candidates:
            records.extend(self.registry.list_available(status="candidate"))
        selection = self.selector.select(records, context)
        if selection.record is None:
            action = self._fallback_action(obs, info, hints)
            self.trace.append({"source": "fallback", "reason": selection.reason, "action_id": int(action)})
            return int(action)

        result = self.runtime.execute(
            selection.record.spec,
            context,
            allow_candidate=self.allow_candidates,
            step=int(obs.get("step", 0) or 0),
            episode_state=self.episode_state,
        )
        self.trace.append({"source": "skill", "selection": selection.reason, "runtime": result.trace.to_dict()})
        if result.chosen_action is None:
            action = self._fallback_action(obs, info, hints)
            self.trace.append({"source": "fallback", "reason": result.termination, "action_id": int(action)})
            return int(action)
        return int(ACTION_NAMES.get(result.chosen_action, int(Action.NOOP)))

    def observe_result(
        self,
        action: int,
        reward: float,
        obs: dict,
        info: dict,
        previous_info: dict | None = None,
    ) -> None:
        observe = getattr(self.fallback_agent, "observe_result", None)
        if observe is not None:
            observe(action, reward, obs, info, previous_info=previous_info)

    def _resolve_skill(self, skill_id: str, version: str | None = None):
        records = self.registry.list_available(status="verified")
        if self.allow_candidates:
            records.extend(self.registry.list_available(status="candidate"))
        for record in records:
            if record.spec.skill_id == skill_id and (version is None or record.spec.version == version):
                return record.spec
        return None

    def _context_hints(self, obs: dict, info: dict) -> dict:
        hints = getattr(self.fallback_agent, "skill_context_hints", None)
        if hints is None:
            return {}
        return dict(hints(obs, info) or {})

    def _fallback_action(self, obs: dict, info: dict, hints: dict) -> int:
        if "fallback_action" not in hints:
            return int(self.fallback_agent.act(obs, info))
        action = int(hints["fallback_action"])
        record_precomputed = getattr(self.fallback_agent, "record_precomputed_action", None)
        if record_precomputed is not None:
            record_precomputed(obs, action, hints.get("_fallback_route_plan"))
        return action
