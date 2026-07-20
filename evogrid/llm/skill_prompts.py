"""Prompt metadata for Skill proposal calls."""

from __future__ import annotations

import hashlib


SKILL_PROPOSER_PROMPT_ID = "skill_proposer_v1"
SKILL_PROPOSER_PROMPT_VERSION = "1.1.0"
SKILL_PROPOSER_PROMPT = (
    "Review only training-partition observable trajectories and propose reusable Skills "
    "as JSON objects using the allowed declarative DSL. Do not mark Skills verified. "
    "Do not generate Python, shell, SQL, dynamic imports, or hidden-map features.\n"
    "Return exactly one JSON object with a top-level skills array. Each item in skills "
    "must be a complete SkillSpec object with these exact keys: schema_version, skill_id, "
    "version, status, name, description, problem_addressed, source, applicability, "
    "procedure, budget, objective, dependencies, rationale. Use status proposed only. "
    "Use procedure, not procedures. The procedure field must be a non-empty array of "
    "declarative nodes whose op is one of the allowed_procedure_ops supplied in the user "
    "message. ACT nodes must use primitive action names, for example BUILD_ROAD. "
    "IF conditions must be structured JSON expressions, not strings. Do not include "
    "spec_hash, verification_reports, storage_status, verified fields, or markdown.\n"
    "Example shape: {\"skills\":[{\"schema_version\":1,\"skill_id\":\"road_on_reused_rough_route\","
    "\"version\":\"1.0.0\",\"status\":\"proposed\",\"name\":\"Road on reused rough route\","
    "\"description\":\"Build a road only when repeated rough-terrain travel can repay construction.\","
    "\"problem_addressed\":\"Repeated high-cost transport\",\"source\":{\"proposer\":\"llm\"},"
    "\"applicability\":{\"all\":[{\"feature\":\"current.terrain_band\",\"op\":\"in\","
    "\"value\":[\"ROUGH\",\"VERY_ROUGH\"]}]},\"procedure\":[{\"op\":\"ESTIMATE\","
    "\"estimator\":\"future_route_uses\",\"store_as\":\"n_use\"},{\"op\":\"ESTIMATE\","
    "\"estimator\":\"road_break_even_uses\",\"store_as\":\"n_break_even\"},{\"op\":\"IF\","
    "\"condition\":{\"left\":{\"var\":\"n_use\"},\"op\":\"gte\",\"right\":{\"var\":\"n_break_even\"}},"
    "\"then\":[{\"op\":\"ACT\",\"action\":\"BUILD_ROAD\"}],\"else\":[{\"op\":\"RETURN\","
    "\"result\":\"not_applicable\"}]}],\"budget\":{\"max_runtime_steps\":4,"
    "\"max_environment_actions\":1,\"max_nested_skill_depth\":0},\"objective\":"
    "{\"primary_metric\":\"road_net_payoff\",\"direction\":\"maximize\","
    "\"negative_context_metric\":\"false_trigger_rate\"},\"dependencies\":[],"
    "\"rationale\":\"Short train-only rationale.\"}]}"
)

SKILL_REVISION_PROMPT_ID = "skill_revision_v1"
SKILL_REVISION_PROMPT_VERSION = "1.7.0"
SKILL_REVISION_PROMPT = (
    "Revise one failed Candidate Skill into a new Candidate SkillSpec version. "
    "Use only the previous SkillSpec, its train source metadata, and aggregated "
    "verification gate feedback. Do not use per-seed records, hidden map truth, or "
    "test-partition information. Do not mark the Skill verified. Return exactly one "
    "JSON object with a top-level skills array containing one complete SkillSpec. "
    "The revised Skill must keep the same skill_id, use the requested new version, "
    "and use status proposed. The revision should narrow applicability and procedure "
    "to address failed gates instead of relaxing verification thresholds. If aggregate "
    "feedback reports zero activation, remove or relax unreachable applicability gates "
    "rather than adding stricter memory requirements. The revised Skill must make an "
    "executable change to applicability, procedure, budget, objective, or dependencies; "
    "metadata-only changes to description, rationale, source, or version are rejected. Use procedure, "
    "not procedures. Applicability leaves may use only feature names supplied in "
    "allowed_applicability_features and op names supplied in allowed_applicability_ops. "
    "Use canonical op names only: eq, ne, lt, lte, gt, gte, in, not_in. Do not use "
    "symbolic or spaced aliases such as ==, !=, >=, <=, not in, or not-in. "
    "Do not invent features such as current.road_exists, "
    "road.already_built, map.route_reused, or hidden evaluator fields. To avoid building "
    "on an existing road, use current.tile_type with the observable tile ids supplied in "
    "the user message; for example current.tile_type == 0 means current tile is GROUND. "
    "When ACT BUILD_ROAD is used, guard it with current.tile_type in the buildable tile "
    "ids GROUND and ROUGH, not only with current.terrain_band. If aggregate feedback "
    "shows negative effect with many road builds or runtime failures, reduce overbuilding "
    "with current.tile_type, memory.visit_count_bucket, or route.remaining_length_bucket. "
    "Bucket features are string enums: route.remaining_length_bucket is one of short, "
    "medium, long, unknown and memory.visit_count_bucket is one of low, medium, high; "
    "use eq/ne/in/not_in with those strings, never numeric comparisons such as gte 2. "
    "A valid route-length applicability leaf is exactly like "
    "{\"feature\":\"route.remaining_length_bucket\",\"op\":\"in\",\"value\":[\"medium\",\"long\"]}. "
    "If current.tile_type is already guarded and activated road payoff is still negative, "
    "add a further executable constraint such as a conservative route.remaining_length_bucket "
    "or memory.visit_count_bucket condition; do not repeat the same applicability. "
    "Prefer route.is_known_transport_route, route.exists, memory.similar_outcome_count, "
    "or memory.similar_mean_payoff when requiring reuse evidence, but avoid positive "
    "memory thresholds when feedback indicates they prevented activation. IF conditions must be "
    "structured JSON expressions, not strings. "
    "Do not include spec_hash, verification_reports, storage_status, verified fields, "
    "markdown, Python, shell, SQL, or dynamic imports."
)


def skill_proposer_prompt_hash() -> str:
    return hashlib.sha256(SKILL_PROPOSER_PROMPT.encode("utf-8")).hexdigest()


def skill_revision_prompt_hash() -> str:
    return hashlib.sha256(SKILL_REVISION_PROMPT.encode("utf-8")).hexdigest()
