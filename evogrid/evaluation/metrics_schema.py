"""Canonical evaluation metric schema."""

from __future__ import annotations

META_COLUMNS = [
    "group",
    "seed",
    "episode",
    "policy_type",
    "model_path",
    "train_log_dir",
    "train_config_path",
    "llm_trace_path",
]

METRIC_COLUMNS = [
    "episode_reward",
    "ore_delivered",
    "num_dig",
    "num_build_road",
    "invalid_actions",
    "road_cells_built",
    "dug_cells",
    "road_usage_rate",
    "road_total_usage_count",
    "road_saved_cost",
    "road_build_cost",
    "road_net_payoff",
    "positive_road_payoff_count",
    "negative_road_payoff_count",
    "transport_steps_per_ore",
    "early_shaping_ratio",
    "late_delivery_rate",
    "steps",
    "llm_calls",
    "llm_successes",
    "llm_fallbacks",
    "llm_success_rate",
]

EVAL_COLUMNS = META_COLUMNS + METRIC_COLUMNS

INTEGER_METRICS = {
    "ore_delivered",
    "num_dig",
    "num_build_road",
    "invalid_actions",
    "road_cells_built",
    "dug_cells",
    "road_total_usage_count",
    "positive_road_payoff_count",
    "negative_road_payoff_count",
    "steps",
    "llm_calls",
    "llm_successes",
    "llm_fallbacks",
}

FLOAT_METRICS = set(METRIC_COLUMNS) - INTEGER_METRICS


def standardize_eval_row(row: dict) -> dict:
    clean = {}
    for column in EVAL_COLUMNS:
        if column in META_COLUMNS:
            clean[column] = row.get(column, "")
        elif column in INTEGER_METRICS:
            clean[column] = int(row.get(column, 0) or 0)
        else:
            clean[column] = float(row.get(column, 0.0) or 0.0)
    return clean


def standardize_eval_rows(rows: list[dict]) -> list[dict]:
    return [standardize_eval_row(row) for row in rows]
