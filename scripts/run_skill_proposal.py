from __future__ import annotations

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import yaml

from evogrid.llm.deepseek_client import DeepSeekClient
from evogrid.llm.skill_prompts import SKILL_PROPOSER_PROMPT, skill_proposer_prompt_hash
from evogrid.skills.proposer import SkillProposer
from evogrid.skills.registry import SkillRegistry
from scripts.calibrate_fractal_maps import _runtime_metadata, _stable_hash


def run_skill_proposal(config_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    out_dir = Path(out_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    proposal_config = config.get("proposal", {})
    started_at = datetime.now(timezone.utc).isoformat()
    registry = SkillRegistry(out_dir / "skills")
    trajectories = list(proposal_config.get("trajectories", []))
    backend_name = str(proposal_config.get("backend", "fixture_empty"))
    backend_metadata: dict[str, Any] = {"schema_version": 1, "backend": backend_name}
    backend = _build_backend(backend_name, proposal_config, backend_metadata)
    result = SkillProposer(registry, backend=backend).propose_from_trajectories(trajectories)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
    proposal_manifest = {
        "schema_version": 1,
        "accepted": [
            {
                "skill_id": record.spec.skill_id,
                "version": record.spec.version,
                "status": record.spec.status,
                "spec_hash": record.spec.spec_hash,
                "path": f"skills/candidates/{record.spec.skill_id}/{record.spec.version}.json",
            }
            for record in result.accepted
        ],
        "rejected": result.rejected,
        "backend": str(proposal_config.get("backend", "fixture_empty")),
        "backend_metadata": backend_metadata,
        "source_partition": "train",
        "verification_started": False,
        "verified_written": False,
    }
    (out_dir / "proposal_manifest.json").write_text(
        json.dumps(proposal_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "run_id": out_dir.name,
        "experiment_type": "skill_proposal",
        "mode": str(proposal_config.get("mode", "pilot")),
        "mock_smoke": True,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "resolved_config_hash": _stable_hash(config),
        "accepted_candidate_count": len(result.accepted),
        "rejected_proposal_count": len(result.rejected),
        "runtime": _runtime_metadata(),
        "outputs": {
            "config_resolved": "config_resolved.yaml",
            "proposal_manifest": "proposal_manifest.json",
            "registry_events": "skills/registry_events.jsonl",
        },
        "formal_acceptance": {
            "passed": False,
            "conclusion_level": "E0",
            "gate_report": "proposal_manifest.json",
        },
        "completion_status": "completed",
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a schema-bounded Skill proposal pilot.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    manifest = run_skill_proposal(args.config, args.out)
    print(
        json.dumps(
            {
                "run_manifest": str(Path(args.out) / "run_manifest.json"),
                "accepted_candidate_count": manifest["accepted_candidate_count"],
                "rejected_proposal_count": manifest["rejected_proposal_count"],
            }
        )
    )


def _fixture_backend(name: str):
    def backend(payload: dict[str, Any]) -> dict[str, Any]:
        if name == "fixture_invalid_exec":
            return {"skills": [_fixture_skill(procedure=[{"op": "EXEC", "code": "print(1)"}])]}
        if name == "fixture_verified_attempt":
            skill = _fixture_skill()
            skill["status"] = "verified"
            return {"skills": [skill]}
        if name == "fixture_valid":
            return {"skills": [_fixture_skill()]}
        return {"skills": []}

    return backend


def _build_backend(name: str, config: dict[str, Any], metadata: dict[str, Any]):
    if name == "deepseek":
        return _deepseek_backend(config.get("deepseek", {}), metadata)
    return _fixture_backend(name)


def _deepseek_backend(config: dict[str, Any], metadata: dict[str, Any], client: Any | None = None):
    client = client or DeepSeekClient(
        base_url=config.get("base_url"),
        model=config.get("model"),
        timeout=config.get("timeout"),
        max_tokens=config.get("max_tokens"),
        json_mode=bool(config.get("json_mode", True)),
    )
    temperature = float(config.get("temperature", 0.2))
    metadata.update(
        {
            "provider": "deepseek",
            "model": client.model,
            "base_url": client.base_url,
            "temperature": temperature,
            "prompt_hash": skill_proposer_prompt_hash(),
            "response_received": False,
            "api_key_env": "DEEPSEEK_API_KEY",
        }
    )

    def backend(payload: dict[str, Any]) -> str:
        messages = [
            {
                "role": "system",
                "content": SKILL_PROPOSER_PROMPT + " Return JSON only with a top-level skills array.",
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "propose_candidate_skills",
                        "constraints": {
                            "allowed_status": payload.get("allowed_status"),
                            "source_partition": payload.get("source_partition"),
                            "allowed_procedure_ops": payload.get("allowed_procedure_ops"),
                        },
                        "prompt_id": payload.get("prompt_id"),
                        "prompt_version": payload.get("prompt_version"),
                        "prompt_hash": payload.get("prompt_hash"),
                        "trajectories": payload.get("trajectories", []),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            },
        ]
        completion = client.chat_completion(messages, temperature=temperature, json_mode=True)
        content = completion["content"]
        metadata.update(
            {
                "response_received": True,
                "finish_reason": completion.get("finish_reason"),
                "response_model": completion.get("model"),
                "usage": completion.get("usage", {}),
                "content_chars": len(content),
                "content_preview": content[:500],
            }
        )
        return content

    return backend


def _fixture_skill(procedure: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "skill_id": "proposal_fixture_road_skill",
        "version": "1.0.0",
        "status": "proposed",
        "name": "Proposal fixture road skill",
        "description": "Fixture proposal for candidate registration plumbing.",
        "problem_addressed": "Repeated high-cost transport",
        "source": {"proposer": "llm"},
        "applicability": {
            "all": [{"feature": "current.terrain_band", "op": "in", "value": ["ROUGH", "VERY_ROUGH"]}]
        },
        "procedure": procedure or [{"op": "ACT", "action": "BUILD_ROAD"}],
        "budget": {"max_runtime_steps": 1, "max_environment_actions": 1, "max_nested_skill_depth": 0},
        "objective": {"primary_metric": "road_net_payoff", "direction": "maximize"},
        "dependencies": [],
    }


if __name__ == "__main__":
    main()
