"""Persistent dynamic edits for chunked worlds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


EVENT_TYPES = {"build_road", "dig", "deplete_ore"}


@dataclass(frozen=True)
class ChunkEvent:
    event_type: str
    x: int
    y: int
    value: Any = True
    actor_id: str = ""
    created_at: str = ""
    schema_version: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkEvent":
        event_type = str(data["event_type"])
        if event_type not in EVENT_TYPES:
            raise ValueError(f"unknown chunk event_type: {event_type}")
        return cls(
            event_type=event_type,
            x=int(data["x"]),
            y=int(data["y"]),
            value=data.get("value", True),
            actor_id=str(data.get("actor_id", "")),
            created_at=str(data.get("created_at", "")),
            schema_version=int(data.get("schema_version", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "x": self.x,
            "y": self.y,
            "value": self.value,
            "actor_id": self.actor_id,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }


class ChunkEventStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else None
        self._events: list[ChunkEvent] = []
        if self.path is not None and self.path.exists():
            self._load()

    def append(self, event: ChunkEvent | dict[str, Any]) -> ChunkEvent:
        event = event if isinstance(event, ChunkEvent) else ChunkEvent.from_dict(event)
        if not event.created_at:
            event = ChunkEvent(
                event_type=event.event_type,
                x=event.x,
                y=event.y,
                value=event.value,
                actor_id=event.actor_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                schema_version=event.schema_version,
            )
        self._events.append(event)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return event

    def events_for_bounds(self, x_min: int, x_max: int, y_min: int, y_max: int) -> list[ChunkEvent]:
        return [
            event
            for event in self._events
            if x_min <= event.x <= x_max and y_min <= event.y <= y_max
        ]

    def all_events(self) -> list[ChunkEvent]:
        return list(self._events)

    def _load(self) -> None:
        assert self.path is not None
        self._events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                self._events.append(ChunkEvent.from_dict(json.loads(line)))
