"""Deterministic seed partitions for leakage-resistant experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evogrid.envs.map_generation.seeding import derive_seed


PARTITION_NAMES = ("train", "gate", "verify", "test", "bootstrap")
SEED_KINDS = ("map", "agent", "bootstrap")


@dataclass(frozen=True)
class SeedPartition:
    name: str
    map_seeds: list[int]
    agent_seeds: list[int]
    bootstrap_seeds: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "map_seeds": list(self.map_seeds),
            "agent_seeds": list(self.agent_seeds),
            "bootstrap_seeds": list(self.bootstrap_seeds),
        }

    def all_namespaced(self) -> list[tuple[str, int]]:
        return (
            [("map", seed) for seed in self.map_seeds]
            + [("agent", seed) for seed in self.agent_seeds]
            + [("bootstrap", seed) for seed in self.bootstrap_seeds]
        )


@dataclass(frozen=True)
class SeedPartitions:
    root_seed: int
    partitions: dict[str, SeedPartition]

    def partition(self, name: str) -> SeedPartition:
        if name not in self.partitions:
            raise KeyError(f"unknown seed partition: {name}")
        return self.partitions[name]

    def assert_disjoint(self) -> None:
        seen: dict[tuple[str, int], str] = {}
        for partition_name, partition in self.partitions.items():
            for item in partition.all_namespaced():
                if item in seen:
                    kind, seed = item
                    raise ValueError(
                        f"seed collision for {kind} seed {seed}: {seen[item]} and {partition_name}"
                    )
                seen[item] = partition_name

    def to_manifest(self) -> dict[str, Any]:
        self.assert_disjoint()
        return {
            "root_seed": self.root_seed,
            "derivation": "blake2b64(root_seed, partition, kind, index)",
            "partitions": {
                name: self.partitions[name].to_dict()
                for name in PARTITION_NAMES
                if name in self.partitions
            },
        }


def make_seed_partitions(root_seed: int, sizes: dict[str, int] | None = None) -> SeedPartitions:
    """Create reproducible train/gate/verify/test/bootstrap seed partitions.

    ``sizes`` may provide per-kind defaults (``map``, ``agent``, ``bootstrap``)
    and/or per-partition overrides such as ``train_map`` or
    ``verify_bootstrap``. Expanded seed lists are saved in manifests so runs do
    not depend on implicit contiguous ranges.
    """

    sizes = sizes or {}
    partitions: dict[str, SeedPartition] = {}
    for name in PARTITION_NAMES:
        partitions[name] = SeedPartition(
            name=name,
            map_seeds=_derive_many(root_seed, name, "map", _size_for(sizes, name, "map")),
            agent_seeds=_derive_many(root_seed, name, "agent", _size_for(sizes, name, "agent")),
            bootstrap_seeds=_derive_many(
                root_seed,
                name,
                "bootstrap",
                _size_for(sizes, name, "bootstrap"),
            ),
        )
    result = SeedPartitions(root_seed=int(root_seed), partitions=partitions)
    result.assert_disjoint()
    return result


def _size_for(sizes: dict[str, int], partition_name: str, kind: str) -> int:
    value = sizes.get(f"{partition_name}_{kind}", sizes.get(kind, sizes.get("default", 4)))
    if int(value) < 0:
        raise ValueError(f"seed partition size must be non-negative: {partition_name}_{kind}")
    return int(value)


def _derive_many(root_seed: int, partition_name: str, kind: str, count: int) -> list[int]:
    return [derive_seed(root_seed, partition_name, kind, index) for index in range(count)]
