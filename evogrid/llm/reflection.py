"""Reflection prompts and deterministic fallback reflection for self-evolution runs."""

from __future__ import annotations

import json


REFLECTION_SYSTEM_PROMPT = (
    "You analyze one partially observed mining episode. "
    "Return JSON only. Do not include markdown. "
    "Write general lessons grounded in observed evidence and memory, not hidden map truth."
)


def build_reflection_messages(
    episode_metrics: dict,
    memory_summary: dict,
    previous_reflection: dict | None = None,
) -> list[dict]:
    payload = {
        "episode_metrics": episode_metrics,
        "memory_summary": memory_summary,
        "previous_reflection": previous_reflection or {},
        "required_json": {
            "lessons": "list of short lessons",
            "try_next": "list of next-episode behavioral adjustments",
            "avoid": "list of repeated mistakes to avoid",
            "confidence": "0.0-1.0",
        },
    }
    return [
        {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def generate_rule_reflection(episode_metrics: dict, memory) -> dict:
    memory_summary = memory.summary()
    ore_delivered = int(episode_metrics.get("ore_delivered", 0) or 0)
    invalid_actions = int(episode_metrics.get("invalid_actions", 0) or 0)
    known_ore_count = int(memory_summary.get("known_ore_count", 0) or 0)
    visited_cell_count = int(memory_summary.get("visited_cell_count", 0) or 0)
    num_build_road = int(episode_metrics.get("num_build_road", 0) or 0)
    num_dig = int(episode_metrics.get("num_dig", 0) or 0)

    lessons: list[str] = []
    try_next: list[str] = []
    avoid: list[str] = []

    if ore_delivered > 0:
        lessons.append("Observed behavior can complete at least one mine-and-return cycle.")
        try_next.append("When carrying ore, prioritize reliable return behavior using only observed traversable cells.")
    else:
        lessons.append("No ore was delivered in this episode.")
        try_next.append("Increase local exploration and preserve remembered discoveries across episodes.")

    if known_ore_count > 0:
        lessons.append("At least one ore tile has been observed and stored in memory.")
        try_next.append("When an observed ore tile is visible or remembered nearby, test actions that confirm reachability.")
    else:
        try_next.append("Search for ore by expanding the observed frontier rather than assuming a hidden target.")

    if invalid_actions > 0:
        lessons.append("Some attempted actions were invalid in the observed state.")
        avoid.append("Avoid repeating actions recorded in recent_failed_actions under the same local conditions.")
    else:
        lessons.append("No invalid actions were recorded.")

    if num_build_road > 0 or num_dig > 0:
        lessons.append("Environment-shaping actions were tested during the episode.")
        try_next.append("Compare shaped cells against later travel reward before repeating the same shaping pattern.")
    else:
        try_next.append("Only test shaping actions when local evidence suggests repeated travel or a blocked passage.")

    if visited_cell_count < 10:
        try_next.append("Reduce local loops by selecting legal moves with lower visit counts.")

    return {
        "source": "rule_reflection",
        "episode": episode_metrics.get("episode"),
        "lessons": lessons,
        "try_next": try_next,
        "avoid": avoid,
        "confidence": 0.7,
    }
