"""Random baseline agent."""

from __future__ import annotations

import random

from evogrid.agents.base_agent import BaseAgent
from evogrid.constants import Action


class RandomAgent(BaseAgent):
    def __init__(self, actions: list[int] | None = None):
        self.actions = actions or [int(action) for action in Action]
        self.rng = random.Random()

    def reset(self, seed: int | None = None) -> None:
        self.rng.seed(seed)

    def act(self, obs: dict, info: dict) -> int:
        return self.rng.choice(self.actions)

