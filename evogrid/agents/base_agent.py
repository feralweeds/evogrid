"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def reset(self, seed: int | None = None) -> None:
        return None

    @abstractmethod
    def act(self, obs: dict, info: dict) -> int:
        raise NotImplementedError

