"""Credit accounting for roads built by the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from evogrid.constants import Tile

Position = Tuple[int, int]


@dataclass
class RoadCreditRecord:
    position: Position
    original_tile: int
    original_move_cost: float
    road_move_cost: float
    build_cost: float
    build_step: int
    usage_count: int = 0
    saved_cost: float = 0.0
    first_used_step: int | None = None
    last_used_step: int | None = None

    @property
    def saving_per_use(self) -> float:
        return self.original_move_cost - self.road_move_cost

    @property
    def net_payoff(self) -> float:
        return self.saved_cost - self.build_cost

    def record_use(self, step: int) -> None:
        self.usage_count += 1
        self.saved_cost += self.saving_per_use
        if self.first_used_step is None:
            self.first_used_step = step
        self.last_used_step = step

    def to_dict(self) -> dict:
        return {
            "position": list(self.position),
            "original_tile": self.original_tile,
            "original_tile_name": Tile(self.original_tile).name,
            "original_move_cost": self.original_move_cost,
            "road_move_cost": self.road_move_cost,
            "build_cost": self.build_cost,
            "build_step": self.build_step,
            "usage_count": self.usage_count,
            "saving_per_use": self.saving_per_use,
            "saved_cost": self.saved_cost,
            "net_payoff": self.net_payoff,
            "first_used_step": self.first_used_step,
            "last_used_step": self.last_used_step,
        }


@dataclass
class RoadCreditTracker:
    records: Dict[Position, RoadCreditRecord] = field(default_factory=dict)

    def record_build(
        self,
        position: Position,
        original_tile: Tile,
        original_move_cost: float,
        road_move_cost: float,
        build_cost: float,
        build_step: int,
    ) -> None:
        self.records[position] = RoadCreditRecord(
            position=position,
            original_tile=int(original_tile),
            original_move_cost=float(original_move_cost),
            road_move_cost=float(road_move_cost),
            build_cost=float(build_cost),
            build_step=int(build_step),
        )

    def record_use(self, position: Position, step: int) -> None:
        record = self.records.get(position)
        if record is not None:
            record.record_use(int(step))

    def clear(self) -> None:
        self.records.clear()

    def to_records(self) -> list[dict]:
        return [record.to_dict() for _, record in sorted(self.records.items())]

    def summary(self) -> dict:
        records = list(self.records.values())
        total_usage = sum(record.usage_count for record in records)
        total_saved = sum(record.saved_cost for record in records)
        total_build_cost = sum(record.build_cost for record in records)
        net_payoff = total_saved - total_build_cost
        positive = sum(1 for record in records if record.net_payoff > 0)
        negative = sum(1 for record in records if record.net_payoff < 0)
        return {
            "road_total_usage_count": total_usage,
            "road_saved_cost": total_saved,
            "road_build_cost": total_build_cost,
            "road_net_payoff": net_payoff,
            "positive_road_payoff_count": positive,
            "negative_road_payoff_count": negative,
        }
