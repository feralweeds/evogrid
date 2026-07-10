"""Optional Gymnasium wrapper for PPO training."""

from __future__ import annotations

from typing import Any

import numpy as np

from evogrid.envs.evogrid_mine_env import EvoGridMineEnv

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - exercised only in missing optional dependency setups
    gym = None


_BaseEnv = gym.Env if gym is not None else object


class GymEvoGridMineEnv(_BaseEnv):
    """Gymnasium-compatible wrapper.

    The wrapper is defined without inheriting from gym.Env so importing the package
    does not require Gymnasium. When Gymnasium is installed, the spaces are created
    and Stable-Baselines3 can use this object.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, config: dict | None = None):
        if gym is None:
            raise ImportError("gymnasium is required for GymEvoGridMineEnv.")
        super().__init__()
        from gymnasium import spaces

        self.env = EvoGridMineEnv(config)
        self.config = config or {}
        env_config = self.config.get("env", self.config)
        height, width = env_config.get("grid_size", [32, 32])
        self.observation_space = spaces.Box(
            low=0.0,
            high=8.0,
            shape=(height * width + 4,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(9)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed)
        return self._flatten(obs), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        return self._flatten(obs), float(reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return None

    def _flatten(self, obs: dict):
        grid = np.array(obs["grid"], dtype=np.float32).reshape(-1)
        agent_row, agent_col = obs["agent_pos"]
        step = float(obs["step"])
        has_ore = 1.0 if obs["has_ore"] else 0.0
        extra = np.array([agent_row, agent_col, step, has_ore], dtype=np.float32)
        return np.concatenate([grid, extra])
