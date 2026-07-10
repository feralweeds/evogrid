"""Reward configuration and helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    dropoff: float = 10.0
    move_ground: float = -0.01
    move_rough: float = -0.05
    move_road: float = 0.0
    dig: float = -0.2
    build_road: float = -0.1
    mine: float = -0.01
    dropoff_action: float = -0.01
    noop: float = -0.01
    invalid_action: float = -0.05

    @classmethod
    def from_config(cls, config: dict | None = None) -> "RewardConfig":
        config = config or {}
        env_config = config.get("env", config)
        rewards = env_config.get("rewards", {})
        known = {field.name for field in cls.__dataclass_fields__.values()}
        clean = {key: value for key, value in rewards.items() if key in known}
        return cls(**clean)

