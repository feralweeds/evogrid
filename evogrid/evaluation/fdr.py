"""Multiple-candidate FDR helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FDRDecision:
    candidate_id: str
    p_value: float
    rank: int
    threshold: float
    rejected_null: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "p_value": self.p_value,
            "rank": self.rank,
            "threshold": self.threshold,
            "rejected_null": self.rejected_null,
        }


def benjamini_hochberg(candidate_p_values: dict[str, float], alpha: float = 0.05) -> list[FDRDecision]:
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1")
    ordered = sorted((candidate_id, float(p_value)) for candidate_id, p_value in candidate_p_values.items())
    ordered.sort(key=lambda item: (item[1], item[0]))
    m = len(ordered)
    if m == 0:
        return []
    passing_ranks = [
        rank
        for rank, (_, p_value) in enumerate(ordered, start=1)
        if p_value <= (rank / m) * float(alpha)
    ]
    cutoff_rank = max(passing_ranks) if passing_ranks else 0
    decisions = []
    for rank, (candidate_id, p_value) in enumerate(ordered, start=1):
        decisions.append(
            FDRDecision(
                candidate_id=candidate_id,
                p_value=p_value,
                rank=rank,
                threshold=(rank / m) * float(alpha),
                rejected_null=rank <= cutoff_rank,
            )
        )
    return decisions
