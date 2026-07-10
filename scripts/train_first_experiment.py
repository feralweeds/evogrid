from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

from evogrid.training.train_ppo import train_ppo
from evogrid.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/env_fixed_map.yaml")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/models/ppo")
    args = parser.parse_args()

    config = load_yaml(args.env_config)
    train_ppo(config=config, total_timesteps=args.timesteps, output_dir=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
