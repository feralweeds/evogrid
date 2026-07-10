"""Optional PPO training entry points."""

from __future__ import annotations

from pathlib import Path

from evogrid.envs.gym_wrapper import GymEvoGridMineEnv


def train_ppo(
    config: dict | None = None,
    total_timesteps: int = 300_000,
    output_dir: str = "outputs/models/ppo",
    seed: int = 0,
    model_name: str | None = None,
    n_steps: int = 512,
    batch_size: int = 64,
    verbose: int = 0,
    monitor_log_dir: str | None = None,
):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise ImportError(
            "PPO training requires gymnasium and stable-baselines3. "
            "The environment, baselines, and DeepSeek agents can be smoke-tested without them."
        ) from exc

    env = GymEvoGridMineEnv(config)
    if monitor_log_dir is not None:
        monitor_path = Path(monitor_log_dir)
        monitor_path.mkdir(parents=True, exist_ok=True)
        env = Monitor(env, filename=str(monitor_path / "monitor.csv"))
    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
    )
    model.learn(total_timesteps=total_timesteps)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_name = model_name or f"ppo_seed{seed}"
    model.save(output_path / model_name)
    return model
