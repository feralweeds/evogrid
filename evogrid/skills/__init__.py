"""Skill data contracts for EvoGrid."""

from evogrid.skills.context import SkillContext
from evogrid.skills.predicates import PredicateResult, evaluate_predicate
from evogrid.skills.proposer import ProposalResult, SkillProposer
from evogrid.skills.registry import SkillRegistry, VerificationLease
from evogrid.skills.runtime import SkillRuntime, SkillRuntimeResult
from evogrid.skills.selector import SkillSelection, SkillSelector
from evogrid.skills.schemas import (
    SkillRecord,
    SkillSpec,
    VerificationReport,
    canonical_json,
    compute_spec_hash,
    validate_status_transition,
)

__all__ = [
    "SkillRecord",
    "SkillContext",
    "SkillSpec",
    "SkillRuntime",
    "SkillRuntimeResult",
    "SkillRegistry",
    "VerificationLease",
    "SkillSelection",
    "SkillSelector",
    "PredicateResult",
    "ProposalResult",
    "SkillProposer",
    "VerificationReport",
    "canonical_json",
    "compute_spec_hash",
    "evaluate_predicate",
    "validate_status_transition",
]
