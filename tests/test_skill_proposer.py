from __future__ import annotations

import tempfile
import unittest

from evogrid.skills.proposer import SkillProposer
from evogrid.skills.registry import SkillRegistry
from tests.test_skill_schema import _skill_dict


class SkillProposerTest(unittest.TestCase):
    def test_valid_proposal_registers_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            registry = SkillRegistry(temp)
            proposal = _skill_dict(status="proposed")
            proposer = SkillProposer(registry, backend=lambda payload: {"skills": [proposal]})

            result = proposer.propose_from_trajectories([{"episode_id": "train/ep/1", "obs": []}])

            self.assertEqual(len(result.accepted), 1)
            self.assertEqual(result.accepted[0].spec.status, "candidate")
            self.assertEqual(result.accepted[0].spec.source["source_episode_ids"], ["train/ep/1"])
            self.assertEqual(result.rejected, [])

    def test_invalid_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            proposal = _skill_dict(status="proposed")
            proposal["procedure"] = [{"op": "EXEC", "code": "print(1)"}]
            proposer = SkillProposer(SkillRegistry(temp), backend=lambda payload: [proposal])

            result = proposer.propose_from_trajectories([{"episode_id": "train/ep/1"}])

            self.assertEqual(result.accepted, [])
            self.assertEqual(len(result.rejected), 1)
            self.assertIn("procedure.op", result.rejected[0]["reason"])

    def test_verified_attempt_is_forced_to_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            proposal = _skill_dict(status="verified")
            proposal["source"]["proposer"] = "llm"
            proposer = SkillProposer(SkillRegistry(temp), backend=lambda payload: proposal)

            result = proposer.propose_from_trajectories([{"episode_id": "train/ep/1"}])

            self.assertEqual(len(result.accepted), 1)
            self.assertEqual(result.accepted[0].spec.status, "candidate")

    def test_duplicate_structure_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            proposal = _skill_dict(status="proposed")
            proposer = SkillProposer(SkillRegistry(temp), backend=lambda payload: {"skills": [proposal, proposal]})

            result = proposer.propose_from_trajectories([{"episode_id": "train/ep/1"}])

            self.assertEqual(len(result.accepted), 1)
            self.assertEqual(len(result.rejected), 1)
            self.assertIn("duplicate", result.rejected[0]["reason"])

    def test_hidden_fields_are_not_passed_to_backend(self):
        seen_payload = {}

        def backend(payload):
            seen_payload.update(payload)
            return []

        with tempfile.TemporaryDirectory() as temp:
            proposer = SkillProposer(SkillRegistry(temp), backend=backend)
            proposer.propose_from_trajectories(
                [
                    {
                        "episode_id": "train/ep/1",
                        "ore_positions": [[9, 9]],
                        "audit": {"x": 1},
                        "obs": [{"nested": {"shortest_path_length": 4}}],
                    }
                ]
            )

        trajectory = seen_payload["trajectories"][0]
        self.assertNotIn("ore_positions", trajectory)
        self.assertNotIn("audit", trajectory)
        self.assertNotIn("shortest_path_length", trajectory["obs"][0]["nested"])
        self.assertIn("allowed_procedure_ops", seen_payload)

    def test_non_train_trajectory_is_rejected_before_backend_call(self):
        calls = 0

        def backend(payload):
            nonlocal calls
            calls += 1
            return []

        with tempfile.TemporaryDirectory() as temp:
            proposer = SkillProposer(SkillRegistry(temp), backend=backend)
            result = proposer.propose_from_trajectories(
                [{"episode_id": "verify/ep/1", "partition": "verify", "obs": []}]
            )

        self.assertEqual(calls, 0)
        self.assertEqual(result.accepted, [])
        self.assertEqual(result.rejected[0]["reason"], "non_train_source_trajectory")


if __name__ == "__main__":
    unittest.main()
