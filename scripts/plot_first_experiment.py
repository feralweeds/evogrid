from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

from evogrid.visualization.plot_curves import plot_first_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot first experiment metrics.")
    parser.add_argument("--input", default="outputs/first_experiment/summary.json")
    parser.add_argument("--metrics-csv")
    parser.add_argument("--out")
    args = parser.parse_args()

    outputs = plot_first_experiment(args.input, metrics_csv=args.metrics_csv, output_dir=args.out)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
