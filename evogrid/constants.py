"""Shared constants for EvoGrid-Mine."""

from __future__ import annotations

from enum import IntEnum


class Tile(IntEnum):
    GROUND = 0
    BASE = 1
    ORE = 2
    OBSTACLE = 3
    ROUGH = 4
    ROAD = 5


class Action(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    MINE = 4
    DIG = 5
    BUILD_ROAD = 6
    DROPOFF = 7
    NOOP = 8


MOVE_DELTAS = {
    Action.MOVE_UP: (-1, 0),
    Action.MOVE_DOWN: (1, 0),
    Action.MOVE_LEFT: (0, -1),
    Action.MOVE_RIGHT: (0, 1),
}

ACTION_NAMES = {action.name: int(action) for action in Action}
ACTION_IDS = {int(action): action.name for action in Action}

TILE_CHARS = {
    Tile.GROUND: ".",
    Tile.BASE: "B",
    Tile.ORE: "O",
    Tile.OBSTACLE: "#",
    Tile.ROUGH: "~",
    Tile.ROAD: "=",
}


def action_from_name(name: str) -> Action:
    normalized = name.strip().upper()
    if normalized in ACTION_NAMES:
        return Action(ACTION_NAMES[normalized])
    raise ValueError(f"Unknown action name: {name}")

