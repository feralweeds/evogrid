from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from evogrid.evaluation.acceptance import audit_run_directory
from evogrid.llm.skill_prompts import SKILL_PROPOSER_PROMPT, SKILL_PROPOSER_PROMPT_VERSION
from scripts.run_skill_proposal import _deepseek_backend, run_skill_proposal


class SkillProposalScriptTest(unittest.TestCase):
    def test_valid_fixture_writes_candidate_only_e0(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            config_path = temp_path / "proposal.yaml"
            out_dir = temp_path / "out"
            config_path.write_text(yaml.safe_dump(_config("fixture_valid")), encoding="utf-8")

            manifest = run_skill_proposal(config_path, out_dir)
            proposal_manifest = json.loads((out_dir / "proposal_manifest.json").read_text(encoding="utf-8"))
            audit = audit_run_directory(out_dir)

            self.assertEqual(manifest["accepted_candidate_count"], 1)
            self.assertEqual(manifest["formal_acceptance"]["conclusion_level"], "E0")
            self.assertTrue((out_dir / "skills" / "candidates" / "proposal_fixture_road_skill" / "1.0.0.json").exists())
            self.assertFalse(any((out_dir / "skills" / "verified").glob("*/*.json")))
            self.assertFalse(proposal_manifest["verification_started"])
            self.assertFalse(proposal_manifest["verified_written"])
            self.assertEqual(proposal_manifest["accepted"][0]["status"], "candidate")
            self.assertTrue(audit.passed)
            self.assertEqual(audit.conclusion_level, "E0")

    def test_invalid_fixture_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            config_path = temp_path / "proposal.yaml"
            out_dir = temp_path / "out"
            config_path.write_text(yaml.safe_dump(_config("fixture_invalid_exec")), encoding="utf-8")

            manifest = run_skill_proposal(config_path, out_dir)
            proposal_manifest = json.loads((out_dir / "proposal_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["accepted_candidate_count"], 0)
            self.assertEqual(manifest["rejected_proposal_count"], 1)
            self.assertIn("procedure.op", proposal_manifest["rejected"][0]["reason"])

    def test_deepseek_backend_records_response_metadata_with_fake_client(self):
        metadata = {}
        backend = _deepseek_backend(
            {"temperature": 0.1},
            metadata,
            client=_FakeDeepSeekClient(),
        )

        raw = backend(
            {
                "allowed_status": "proposed",
                "source_partition": "train",
                "allowed_procedure_ops": ["ACT"],
                "prompt_id": "skill_proposer_v1",
                "prompt_version": "1.0.0",
                "prompt_hash": "hash",
                "trajectories": [],
            }
        )

        self.assertIn("skills", raw)
        self.assertTrue(metadata["response_received"])
        self.assertEqual(metadata["finish_reason"], "stop")
        self.assertEqual(metadata["usage"]["total_tokens"], 12)

    def test_skill_proposer_prompt_names_required_schema_fields(self):
        self.assertEqual(SKILL_PROPOSER_PROMPT_VERSION, "1.1.0")
        for text in (
            "top-level skills array",
            "skill_id",
            "applicability",
            "procedure, not procedures",
            "IF conditions must be structured JSON expressions",
            "Do not include spec_hash",
        ):
            self.assertIn(text, SKILL_PROPOSER_PROMPT)


def _config(backend: str) -> dict:
    return {
        "proposal": {
            "schema_version": 1,
            "mode": "pilot",
            "backend": backend,
            "trajectories": [
                {
                    "episode_id": "train/episode/1",
                    "partition": "train",
                    "observations": [{"agent_pos": [2, 2]}],
                }
            ],
        }
    }


class _FakeDeepSeekClient:
    model = "fake-deepseek"
    base_url = "https://example.test"

    def chat_completion(self, messages, temperature=0.2, json_mode=True):
        self.messages = messages
        return {
            "content": '{"skills": []}',
            "finish_reason": "stop",
            "model": self.model,
            "usage": {"total_tokens": 12},
        }


if __name__ == "__main__":
    unittest.main()
