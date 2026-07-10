"""Evidence gates for learned road-building decisions."""

from __future__ import annotations

ACTIONABLE_LEARNED_SOURCES = {"contextual", "contextual_route", "tile_specific"}
CONTEXTUAL_LEARNED_SOURCES = {"contextual", "contextual_route"}


def learned_road_evidence_gate(
    opportunity: dict,
    learned_value_threshold: float = 0.0,
    confidence_threshold: float = 0.0,
    min_contextual_evidence_count: int = 1,
    positive_rate_threshold: float = 0.0,
    require_contextual_evidence: bool = False,
    require_on_route: bool = False,
) -> dict:
    estimate = opportunity.get("learned_estimate", {}) or {}
    source = estimate.get("source")
    evidence_count = int(estimate.get("evidence_count", 0) or 0)
    learned_value = float(estimate.get("learned_value", 0.0) or 0.0)
    positive_rate = float(estimate.get("positive_rate", 0.0) or 0.0)
    confidence = float(estimate.get("confidence", 0.0) or 0.0)
    route_context = opportunity.get("route_context", {}) or {}

    reasons = []
    allowed_sources = CONTEXTUAL_LEARNED_SOURCES if require_contextual_evidence else ACTIONABLE_LEARNED_SOURCES
    if source not in allowed_sources:
        reasons.append("source_not_allowed")
    if evidence_count < int(min_contextual_evidence_count):
        reasons.append("insufficient_evidence_count")
    if positive_rate < float(positive_rate_threshold):
        reasons.append("positive_rate_below_threshold")
    if learned_value <= float(learned_value_threshold):
        reasons.append("mean_payoff_below_threshold")
    if confidence < float(confidence_threshold):
        reasons.append("confidence_below_threshold")
    if require_on_route and not bool(route_context.get("on_current_route")):
        reasons.append("not_on_current_route")

    return {
        "passes": not reasons,
        "failed_reasons": reasons,
        "source": source or "none",
        "evidence_count": evidence_count,
        "positive_rate": positive_rate,
        "learned_value": learned_value,
        "confidence": confidence,
        "thresholds": {
            "min_contextual_evidence_count": int(min_contextual_evidence_count),
            "positive_rate_threshold": float(positive_rate_threshold),
            "learned_value_threshold": float(learned_value_threshold),
            "confidence_threshold": float(confidence_threshold),
            "require_contextual_evidence": bool(require_contextual_evidence),
            "require_on_route": bool(require_on_route),
        },
    }
