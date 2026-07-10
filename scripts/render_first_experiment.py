from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import itertools
import json
from pathlib import Path

from evogrid.evaluation.qualitative import capture_baseline_rollout, capture_deepseek_rollout, capture_ppo_rollout
from evogrid.utils.config import load_yaml
from evogrid.visualization.make_video import save_frames_text
from evogrid.visualization.plot_heatmaps import cells_to_matrix, counts_to_matrix, save_heatmap
from evogrid.visualization.render_map import save_ascii_map, save_before_after_image, save_map_image

PPO_GROUPS = {"full_shaping", "no_shaping"}
BASELINE_GROUPS = {"random", "greedy"}
LLM_GROUPS = {"deepseek_planner", "hybrid_deepseek_greedy"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render qualitative first-experiment rollouts.")
    parser.add_argument("--summary", default="outputs/first_experiment/summary.json")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--figures-out")
    parser.add_argument("--deepseek-config", default="configs/deepseek.yaml")
    parser.add_argument("--mock-deepseek", action="store_true")
    parser.add_argument("--quiet-llm-calls", action="store_true")
    parser.add_argument("--deepseek-trace-prompts", action="store_true")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    groups = args.groups or list(summary.get("groups", {}).keys())
    out_dir = Path(args.out or summary.get("outputs", {}).get("rollouts_dir") or summary_path.parent / "rollouts")
    figures_dir = Path(args.figures_out) if args.figures_out else Path(summary.get("outputs", {}).get("root", summary_path.parent)) / "figures" / "qualitative"
    config_paths = summary.get("config", {}).get("env_configs", {})
    deepseek_runtime = _deepseek_runtime(summary, args.deepseek_config)
    mock_deepseek_responses = _mock_deepseek_responses(args, summary)

    outputs = []
    for group in groups:
        config_path = config_paths.get(group) or config_paths.get("full_shaping") or "configs/env_full_shaping.yaml"
        config = load_yaml(config_path)
        group_dir = out_dir / group
        group_figures_dir = figures_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        group_figures_dir.mkdir(parents=True, exist_ok=True)

        if group in PPO_GROUPS:
            model_path = _first_artifact(summary, group, "model_paths")
            if not model_path:
                print(f"Skip {group}: no model path in summary")
                continue
            rollout = capture_ppo_rollout(group, config, model_path, seed=args.seed)
        elif group in BASELINE_GROUPS:
            rollout = capture_baseline_rollout(group, config, seed=args.seed)
        elif group in LLM_GROUPS:
            max_steps = args.max_steps or deepseek_runtime.get("max_steps")
            if max_steps is not None:
                config.setdefault("env", {})["max_steps"] = int(max_steps)
            rollout = capture_deepseek_rollout(
                group,
                config,
                deepseek_runtime,
                mock_responses=mock_deepseek_responses,
                seed=args.seed,
                log_llm_calls=not args.quiet_llm_calls,
                trace_prompts=args.deepseek_trace_prompts,
            )
        else:
            print(f"Skip {group}: unsupported qualitative renderer")
            continue

        height = len(rollout.after_grid)
        width = len(rollout.after_grid[0]) if height else 0
        changed_cells = rollout.changed_cells | rollout.built_roads | rollout.dug_cells
        frames_path = save_frames_text(group_dir / "frames.txt", rollout.frames)
        rollout_text = "\n\n".join(
            [
                f"Metrics: {rollout.metrics}",
                f"LLM trace count: {len(rollout.trace)}",
                f"Frames file: {frames_path}",
            ]
        )
        if rollout.trace:
            (group_dir / "llm_trace.jsonl").write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in rollout.trace) + "\n",
                encoding="utf-8",
            )

        paths = [
            frames_path,
            save_ascii_map(group_dir / "rollout.txt", rollout_text),
            save_map_image(rollout.before_grid, group_figures_dir / "map_before.png", title=f"{group} Before"),
            save_map_image(rollout.after_grid, group_figures_dir / "map_after.png", agent_pos=rollout.agent_pos, title=f"{group} After"),
            save_before_after_image(rollout.before_grid, rollout.after_grid, group_figures_dir / "map_before_after.png", title=f"{group} Before / After"),
            save_heatmap(
                counts_to_matrix(rollout.visited_counts, height, width),
                group_figures_dir / "trajectory_heatmap.png",
                title=f"{group} Trajectory Heatmap",
                cmap="magma",
            ),
            save_heatmap(
                cells_to_matrix(changed_cells, height, width),
                group_figures_dir / "shaping_heatmap.png",
                title=f"{group} Shaping Heatmap",
                cmap="viridis",
            ),
        ]
        outputs.extend(paths)
        print(f"{group}: wrote {group_dir} and {group_figures_dir}")

    for output in outputs:
        print(output)


def _first_artifact(summary: dict, group: str, key: str) -> str | None:
    values = summary.get("groups", {}).get(group, {}).get("artifacts", {}).get(key, [])
    return values[0] if values else None


def _deepseek_runtime(summary: dict, config_path: str) -> dict:
    runtime = dict(summary.get("config", {}).get("deepseek", {}))
    config = load_yaml(runtime.get("config_path") or config_path).get("deepseek", {})
    return {
        "mode": _config_value(runtime.get("mode", config.get("mode")), "planner"),
        "replan_interval": int(_config_value(runtime.get("replan_interval", config.get("replan_interval")), 20)),
        "temperature": float(_config_value(runtime.get("temperature", config.get("temperature")), 0.2)),
        "timeout": int(_config_value(runtime.get("timeout", config.get("timeout")), 30)),
        "max_tokens": int(_config_value(runtime.get("max_tokens", config.get("max_tokens")), 512)),
        "json_mode": bool(_config_value(runtime.get("json_mode", config.get("json_mode")), True)),
        "max_retries": int(_config_value(runtime.get("max_retries", config.get("max_retries")), 0)),
        "max_steps": _config_value(runtime.get("max_steps"), None),
        "model": _config_value(runtime.get("model", config.get("model")), None),
        "base_url": _config_value(runtime.get("base_url", config.get("base_url")), None),
    }


def _mock_deepseek_responses(args: argparse.Namespace, summary: dict):
    if not args.mock_deepseek and not summary.get("config", {}).get("deepseek", {}).get("mock", False):
        return None
    return itertools.repeat(
        json.dumps(
            {
                "mode": "action",
                "action": "MOVE_RIGHT",
                "action_id": 3,
                "reason": "offline qualitative smoke test",
                "confidence": 1.0,
            }
        )
    )


def _config_value(value, default):
    if value is None:
        return default
    text = str(value).strip()
    if not text or (text.startswith("${") and text.endswith("}")):
        return default
    return value


if __name__ == "__main__":
    main()
