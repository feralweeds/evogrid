"""Optional Stable-Baselines3 PPO wrapper."""

from __future__ import annotations

from evogrid.agents.base_agent import BaseAgent


class PPOAgent(BaseAgent):
    def __init__(self, model_path: str):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required for PPOAgent. "
                "Install requirements.txt before PPO training."
            ) from exc
        self.model = PPO.load(model_path)

    def act(self, obs: dict, info: dict) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

