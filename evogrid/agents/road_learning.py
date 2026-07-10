"""Learned road-building value estimates from past road payoff records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from evogrid.constants import Tile


@dataclass
class RoadLearningStats:
    sample_count: int = 0
    positive_count: int = 0
    total_payoff: float = 0.0
    total_usage: int = 0

    @property
    def mean_payoff(self) -> float:
        return self.total_payoff / self.sample_count if self.sample_count else 0.0

    @property
    def positive_rate(self) -> float:
        return self.positive_count / self.sample_count if self.sample_count else 0.0

    @property
    def mean_usage(self) -> float:
        return self.total_usage / self.sample_count if self.sample_count else 0.0

    def add(self, payoff: float, usage_count: int) -> None:
        self.sample_count += 1
        self.total_payoff += float(payoff)
        self.total_usage += int(usage_count)
        if payoff > 0.0:
            self.positive_count += 1

    def to_dict(self) -> dict:
        return {
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "mean_payoff": self.mean_payoff,
            "positive_rate": self.positive_rate,
            "mean_usage": self.mean_usage,
        }


@dataclass
class RoadLearningModule:
    """Estimate road-building value without choosing an action."""

    by_original_tile: dict[int, RoadLearningStats] = field(default_factory=dict)
    by_context: dict[tuple[int, str, str], RoadLearningStats] = field(default_factory=dict)
    by_route_context: dict[tuple[int, str], RoadLearningStats] = field(default_factory=dict)
    context_sample_count_by_tile: dict[int, int] = field(default_factory=dict)
    global_stats: RoadLearningStats = field(default_factory=RoadLearningStats)

    def update_from_records(self, records: Iterable[dict]) -> None:
        for record in records:
            original_tile = _optional_int(record.get("original_tile"))
            if original_tile is None:
                continue
            payoff = float(record.get("net_payoff", 0.0) or 0.0)
            usage_count = int(record.get("usage_count", 0) or 0)
            self.by_original_tile.setdefault(original_tile, RoadLearningStats()).add(payoff, usage_count)
            context_key = _record_context_key(original_tile, record)
            if context_key is not None:
                route_key = (original_tile, context_key[1])
                self.by_context.setdefault(context_key, RoadLearningStats()).add(payoff, usage_count)
                self.by_route_context.setdefault(route_key, RoadLearningStats()).add(payoff, usage_count)
                self.context_sample_count_by_tile[original_tile] = (
                    self.context_sample_count_by_tile.get(original_tile, 0) + 1
                )
            self.global_stats.add(payoff, usage_count)

    def estimate(self, current_tile: int | None, context: dict | None = None) -> dict:
        context_descriptor = _estimate_context(current_tile, context)
        if current_tile is not None and context_descriptor is not None:
            exact_stats = self.by_context.get(context_descriptor["context_key"])
            if exact_stats is not None and exact_stats.sample_count:
                return self._estimate_from_stats(
                    exact_stats,
                    "contextual",
                    context_descriptor,
                )
            route_stats = self.by_route_context.get(context_descriptor["route_key"])
            if route_stats is not None and route_stats.sample_count:
                return self._estimate_from_stats(
                    route_stats,
                    "contextual_route",
                    context_descriptor,
                )
            if self.context_sample_count_by_tile.get(int(current_tile), 0) > 0:
                return self._estimate_from_stats(
                    RoadLearningStats(),
                    "none",
                    context_descriptor,
                )

        tile_stats = self.by_original_tile.get(int(current_tile)) if current_tile is not None else None
        stats = tile_stats or RoadLearningStats()
        return self._estimate_from_stats(
            stats,
            "tile_specific" if stats.sample_count else "none",
            context_descriptor,
        )

    def _estimate_from_stats(
        self,
        stats: RoadLearningStats,
        source: str,
        context_descriptor: dict | None,
    ) -> dict:
        confidence = _confidence(stats.sample_count)
        return {
            "learned_value": stats.mean_payoff,
            "positive_rate": stats.positive_rate,
            "mean_usage": stats.mean_usage,
            "evidence_count": stats.sample_count,
            "confidence": confidence,
            "source": source if stats.sample_count else "none",
            "context": context_descriptor or {},
            "global_prior": self.global_stats.to_dict(),
            "uses_hidden_map": False,
            "uses_future_truth": False,
            "auto_execute": False,
        }

    def summary(self) -> dict:
        return {
            "global": self.global_stats.to_dict(),
            "by_original_tile": {
                Tile(tile).name: stats.to_dict()
                for tile, stats in sorted(self.by_original_tile.items())
            },
            "by_context": {
                _format_context_key(key): stats.to_dict()
                for key, stats in sorted(self.by_context.items())
            },
        }

    @classmethod
    def from_records(cls, records: Iterable[dict]) -> "RoadLearningModule":
        module = cls()
        module.update_from_records(records)
        return module


def _confidence(sample_count: int) -> float:
    return min(1.0, sample_count / 5.0)


def _optional_int(value) -> int | None:
    if value is None:
        return None
    return int(value)


def _record_context_key(original_tile: int, record: dict) -> tuple[int, str, str] | None:
    if "route_on_build" not in record and "known_as_transport_corridor" not in record:
        return None
    route_status = _route_status(
        bool(record.get("route_on_build")) or bool(record.get("known_as_transport_corridor"))
    )
    return original_tile, route_status, _length_bucket(record.get("route_remaining_length"))


def _estimate_context(current_tile: int | None, context: dict | None) -> dict | None:
    if current_tile is None or not context:
        return None
    route_context = context.get("route_context", {})
    memory_evidence = context.get("memory_evidence", {})
    on_transport = bool(route_context.get("on_current_route")) or bool(
        memory_evidence.get("known_as_transport_corridor")
    )
    route_status = _route_status(on_transport)
    length_bucket = _length_bucket(route_context.get("route_remaining_length"))
    tile = int(current_tile)
    return {
        "route_status": route_status,
        "route_length_bucket": length_bucket,
        "context_key": (tile, route_status, length_bucket),
        "route_key": (tile, route_status),
    }


def _route_status(on_transport: bool) -> str:
    return "transport_route" if on_transport else "off_route"


def _length_bucket(value) -> str:
    if value is None:
        return "unknown"
    length = int(value)
    if length <= 4:
        return "short"
    if length <= 10:
        return "medium"
    return "long"


def _format_context_key(key: tuple[int, str, str]) -> str:
    tile, route_status, length_bucket = key
    return f"{Tile(tile).name}:{route_status}:{length_bucket}"
