"""Prompt builders for DeepSeek-backed agents."""

from __future__ import annotations

import json

from evogrid.constants import ACTION_IDS, MOVE_DELTAS, TILE_CHARS, Action, Tile


SYSTEM_PROMPT = (
    "You are controlling an agent in a 2D mining grid. "
    "Return JSON only. Do not include markdown. "
    "The goal is to collect ore and deliver it to base. "
    "DIG and BUILD_ROAD have short-term costs but can improve future transport."
)

SELF_EVOLUTION_SYSTEM_PROMPT = (
    "You are controlling an agent in a partially observed 2D mining grid. "
    "Return JSON only. Do not include markdown. "
    "Use only the current local observation and the provided memory summary. "
    "Do not assume ore, obstacle, or road locations that have not been observed. "
    "The research goal is self-evolution: improve through observation, memory, and reflection, "
    "not through a hard-coded path-planning strategy."
)

LLM_ROAD_LEARNING_SYSTEM_PROMPT = (
    "You are controlling road-building choices in a partially observed 2D mining grid. "
    "Return JSON only. Do not include markdown. "
    "Use only the current local observation, memory summary, route context, shaping opportunity, "
    "learned road payoff estimates, and exploration budget. "
    "Do not assume hidden map facts. "
    "BUILD_ROAD is allowed only when justified as either exploration under the budget or exploitation "
    "of positive learned payoff evidence."
)


def summarize_observation(obs: dict) -> dict:
    mode = obs.get("observation_mode") or ("full_obs" if "grid" in obs else "partial_obs")
    summary = {
        "observation_mode": mode,
        "agent_pos": obs["agent_pos"],
        "base_pos": obs["base_pos"],
        "has_ore": obs["has_ore"],
        "step": obs["step"],
        "ore_delivered": obs.get("ore_delivered"),
        "action_space": ACTION_IDS,
    }
    if "grid" in obs:
        grid = obs["grid"]
        summary["tile_counts"] = _count_grid_tiles(grid)
        summary["ore_positions"] = obs.get("ore_positions", [])
        summary["local_view"] = render_local_view(obs, radius=int(obs.get("local_view_radius", 4)))
        return summary

    visible_tiles = obs.get("visible_tiles", [])
    summary.update(
        {
            "local_view_radius": obs.get("local_view_radius"),
            "local_view_origin": obs.get("local_view_origin"),
            "visible_tile_counts": _count_visible_tiles(visible_tiles),
            "visible_ore_positions": [
                item["pos"] for item in visible_tiles if int(item.get("tile", -1)) == int(Tile.ORE)
            ],
            "local_view": render_local_view(obs, radius=int(obs.get("local_view_radius", 4))),
            "visibility_note": "Unseen cells are intentionally omitted; do not infer hidden ore positions.",
        }
    )
    return summary


def build_action_messages(obs: dict, info: dict) -> list[dict]:
    payload = {
        "observation": summarize_observation(obs),
        "metrics": _compact_metrics(info),
        "required_json": {
            "mode": "action",
            "action": "MOVE_UP | MOVE_DOWN | MOVE_LEFT | MOVE_RIGHT | MINE | DIG | BUILD_ROAD | DROPOFF | NOOP",
            "action_id": "integer 0-8",
            "reason": "short reason",
            "confidence": "0.0-1.0",
        },
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def build_planner_messages(obs: dict, info: dict) -> list[dict]:
    payload = {
        "observation": summarize_observation(obs),
        "metrics": _compact_metrics(info),
        "instruction": (
            "Choose either one immediate action or a short environment-shaping plan. "
            "Prefer plans that build roads or dig useful passages only when they help repeated transport."
        ),
        "required_json": {
            "mode": "action | plan",
            "action": "optional primitive action name",
            "action_id": "optional integer 0-8",
            "subgoal": "optional subgoal name",
            "target_cells": "optional list of [row, col]",
            "preferred_actions": "optional list of primitive action names",
            "stop_condition": "optional stop condition",
            "reason": "short reason",
            "confidence": "0.0-1.0",
        },
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def build_self_evolution_messages(
    obs: dict,
    info: dict,
    memory_summary: dict,
    reflection: dict | None = None,
    shaping_opportunity: dict | None = None,
) -> list[dict]:
    payload = {
        "observation": summarize_observation(obs),
        "memory_summary": memory_summary,
        "previous_reflection": reflection or {},
        "metrics": _compact_metrics(info),
        "action_diagnostics": build_action_diagnostics(obs, memory_summary),
        "shaping_opportunity": shaping_opportunity or {"available": False, "reason": "not evaluated"},
        "instruction": (
            "Choose one legal primitive action. Exploration, mining, dropoff, digging, and road building "
            "are allowed, but the decision must be justified from visible evidence or remembered observations. "
            "shaping_opportunity is non-binding evidence; candidate_action is not an instruction. "
            "Use action_diagnostics before choosing. Do not claim a move heads toward base when delta_to_base "
            "is positive. If evidence is insufficient, prefer gathering information over pretending to know "
            "hidden targets."
        ),
        "required_json": {
            "mode": "action",
            "action": "MOVE_UP | MOVE_DOWN | MOVE_LEFT | MOVE_RIGHT | MINE | DIG | BUILD_ROAD | DROPOFF | NOOP",
            "action_id": "integer 0-8",
            "reason": "short reason grounded in observation/memory",
            "confidence": "0.0-1.0",
        },
    }
    return [
        {"role": "system", "content": SELF_EVOLUTION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def build_llm_road_learning_messages(
    obs: dict,
    info: dict,
    memory_summary: dict,
    shaping_opportunity: dict,
    route_action_id: int,
    exploration_state: dict,
) -> list[dict]:
    route_action_name = ACTION_IDS.get(int(route_action_id), str(route_action_id))
    payload = {
        "observation": summarize_observation(obs),
        "memory_summary": memory_summary,
        "metrics": _compact_metrics(info),
        "action_diagnostics": build_action_diagnostics(obs, memory_summary),
        "route_action": {
            "action": route_action_name,
            "action_id": int(route_action_id),
            "role": "low-level navigator recommendation",
        },
        "shaping_opportunity": shaping_opportunity,
        "exploration_state": exploration_state,
        "instruction": (
            "Choose exactly one legal primitive action. Normally follow route_action. "
            "Choose BUILD_ROAD only if shaping_opportunity.available is true and either: "
            "(1) exploration_state.learned_evidence_strong is true, or "
            "(2) exploration_budget_remaining is positive and the road is a reasonable exploration sample. "
            "If learned_builds_remaining is 0, do not choose BUILD_ROAD only from learned evidence. "
            "candidate_action is not an instruction. Do not say rough terrain is valuable unless the value "
            "comes from learned_estimate/history or you explicitly mark it as exploration."
        ),
        "required_json": {
            "mode": "action",
            "action": "MOVE_UP | MOVE_DOWN | MOVE_LEFT | MOVE_RIGHT | MINE | DIG | BUILD_ROAD | DROPOFF | NOOP",
            "action_id": "integer 0-8",
            "decision_source": "route | explore_road | learned_road",
            "reason": "short reason grounded in prompt evidence",
            "confidence": "0.0-1.0",
        },
    }
    return [
        {"role": "system", "content": LLM_ROAD_LEARNING_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def build_action_diagnostics(obs: dict, memory_summary: dict | None = None) -> list[dict]:
    agent_pos = _position(obs["agent_pos"])
    base_pos = _position(obs["base_pos"])
    known_ore_positions = [
        _position(pos) for pos in (memory_summary or {}).get("known_ore_locations", []) if len(pos) == 2
    ]
    diagnostics: list[dict] = []

    for action in Action:
        item = {
            "action": action.name,
            "action_id": int(action),
            "legal": False,
            "target_pos": None,
            "target_tile": None,
            "delta_to_base": None,
            "delta_to_nearest_known_ore": None,
            "reason": "",
        }
        if action in MOVE_DELTAS:
            target = _add(agent_pos, MOVE_DELTAS[action])
            tile = _visible_tile(obs, target)
            item.update(
                {
                    "target_pos": list(target),
                    "target_tile": _tile_name(tile),
                    "legal": tile is not None and tile != int(Tile.OBSTACLE),
                    "delta_to_base": _manhattan(target, base_pos) - _manhattan(agent_pos, base_pos),
                    "delta_to_nearest_known_ore": _distance_delta_to_nearest(agent_pos, target, known_ore_positions),
                    "reason": "visible passable target" if tile is not None and tile != int(Tile.OBSTACLE) else "blocked or unseen target",
                }
            )
        elif action == Action.MINE:
            can_mine = not bool(obs.get("has_ore")) and _has_current_or_adjacent_tile(obs, Tile.ORE)
            item.update(
                {
                    "legal": can_mine,
                    "target_pos": _first_current_or_adjacent_tile(obs, Tile.ORE),
                    "target_tile": "ORE" if can_mine else None,
                    "reason": "current or adjacent visible ore" if can_mine else "requires visible ore on current or adjacent cell and empty cargo",
                }
            )
        elif action == Action.DIG:
            target = _first_adjacent_tile(obs, Tile.OBSTACLE)
            item.update(
                {
                    "legal": target is not None,
                    "target_pos": target,
                    "target_tile": "OBSTACLE" if target is not None else None,
                    "reason": "adjacent visible obstacle" if target is not None else "requires adjacent visible obstacle",
                }
            )
        elif action == Action.BUILD_ROAD:
            current_tile = _visible_tile(obs, agent_pos)
            legal = current_tile in {int(Tile.GROUND), int(Tile.ROUGH)}
            item.update(
                {
                    "legal": legal,
                    "target_pos": list(agent_pos),
                    "target_tile": _tile_name(current_tile),
                    "reason": "current tile can be roaded" if legal else "requires current ground or rough tile",
                }
            )
        elif action == Action.DROPOFF:
            legal = bool(obs.get("has_ore")) and agent_pos == base_pos
            item.update(
                {
                    "legal": legal,
                    "target_pos": list(agent_pos),
                    "target_tile": _tile_name(_visible_tile(obs, agent_pos)),
                    "reason": "at base while carrying ore" if legal else "requires carrying ore at base",
                }
            )
        elif action == Action.NOOP:
            item.update({"legal": True, "target_pos": list(agent_pos), "reason": "always legal"})
        diagnostics.append(item)
    return diagnostics


def render_local_view(obs: dict, radius: int = 4) -> str:
    if "local_view" in obs:
        local_view = obs.get("local_view", [])
        origin_row, origin_col = obs.get("local_view_origin", [0, 0])
        agent_row, agent_col = obs["agent_pos"]
        rows = []
        for row_idx, row in enumerate(local_view):
            chars = []
            for col_idx, value in enumerate(row):
                abs_pos = (origin_row + row_idx, origin_col + col_idx)
                if abs_pos == (agent_row, agent_col):
                    chars.append("A")
                elif value is None:
                    chars.append(" ")
                else:
                    chars.append(TILE_CHARS[Tile(int(value))])
            rows.append("".join(chars))
        return "\n".join(rows)

    if obs.get("visible_tiles"):
        parts = []
        for item in obs.get("visible_tiles", []):
            parts.append(f"{item['pos']}:{_tile_name(int(item['tile']))}")
        return "visible_tiles " + ", ".join(parts)

    grid = obs["grid"]
    agent_row, agent_col = obs["agent_pos"]
    rows = []
    for row in range(agent_row - radius, agent_row + radius + 1):
        chars = []
        for col in range(agent_col - radius, agent_col + radius + 1):
            if row == agent_row and col == agent_col:
                chars.append("A")
            elif row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]):
                chars.append(" ")
            else:
                chars.append(TILE_CHARS[Tile(grid[row][col])])
        rows.append("".join(chars))
    return "\n".join(rows)


def _position(value) -> tuple[int, int]:
    return int(value[0]), int(value[1])


def _add(pos: tuple[int, int], delta: tuple[int, int]) -> tuple[int, int]:
    return pos[0] + delta[0], pos[1] + delta[1]


def _visible_tile(obs: dict, pos: tuple[int, int]) -> int | None:
    if obs.get("visible_tiles"):
        for item in obs["visible_tiles"]:
            if _position(item["pos"]) == pos:
                return int(item["tile"])
        return None
    if "local_view" in obs:
        origin_row, origin_col = obs.get("local_view_origin", [0, 0])
        row_idx = pos[0] - int(origin_row)
        col_idx = pos[1] - int(origin_col)
        local_view = obs.get("local_view", [])
        if 0 <= row_idx < len(local_view) and 0 <= col_idx < len(local_view[row_idx]):
            value = local_view[row_idx][col_idx]
            return None if value is None else int(value)
        return None
    if "grid" in obs:
        grid = obs["grid"]
        row, col = pos
        if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
            return int(grid[row][col])
    return None


def _tile_name(tile: int | None) -> str:
    if tile is None:
        return "UNKNOWN"
    return Tile(int(tile)).name


def _has_current_or_adjacent_tile(obs: dict, tile: Tile) -> bool:
    return _first_current_or_adjacent_tile(obs, tile) is not None


def _first_current_or_adjacent_tile(obs: dict, tile: Tile) -> list[int] | None:
    agent_pos = _position(obs["agent_pos"])
    if _visible_tile(obs, agent_pos) == int(tile):
        return list(agent_pos)
    return _first_adjacent_tile(obs, tile)


def _first_adjacent_tile(obs: dict, tile: Tile) -> list[int] | None:
    agent_pos = _position(obs["agent_pos"])
    for delta in MOVE_DELTAS.values():
        pos = _add(agent_pos, delta)
        if _visible_tile(obs, pos) == int(tile):
            return list(pos)
    return None


def _distance_delta_to_nearest(
    current: tuple[int, int],
    target: tuple[int, int],
    candidates: list[tuple[int, int]],
) -> int | None:
    if not candidates:
        return None
    current_distance = min(_manhattan(current, pos) for pos in candidates)
    target_distance = min(_manhattan(target, pos) for pos in candidates)
    return target_distance - current_distance


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _count_grid_tiles(grid: list[list[int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in grid:
        for value in row:
            tile_name = Tile(int(value)).name
            counts[tile_name] = counts.get(tile_name, 0) + 1
    return counts


def _count_visible_tiles(visible_tiles: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in visible_tiles:
        tile_name = Tile(int(item["tile"])).name
        counts[tile_name] = counts.get(tile_name, 0) + 1
    return counts


def _compact_metrics(info: dict) -> dict:
    keys = [
        "episode_reward",
        "ore_delivered",
        "num_mine",
        "num_dig",
        "num_build_road",
        "invalid_actions",
        "final_has_ore",
        "final_agent_pos",
        "carrying_steps",
        "road_usage_rate",
        "road_total_usage_count",
        "road_saved_cost",
        "road_build_cost",
        "road_net_payoff",
        "positive_road_payoff_count",
        "negative_road_payoff_count",
        "transport_steps_per_ore",
    ]
    return {key: info.get(key) for key in keys if key in info}
