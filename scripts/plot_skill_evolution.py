from __future__ import annotations

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from evogrid.visualization.plot_skill_evolution import plot_skill_evolution


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved Skill-evolution experiment outputs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    outputs = plot_skill_evolution(args.run_dir, args.out)
    print(json.dumps({"figures": [str(path) for path in outputs]}))


if __name__ == "__main__":
    main()
