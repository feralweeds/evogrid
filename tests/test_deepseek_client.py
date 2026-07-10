from __future__ import annotations

import unittest

from evogrid.llm.deepseek_client import DeepSeekClient
from evogrid.llm.parser import extract_json_object
from evogrid.llm.schemas import LLMDecision


class DeepSeekClientTest(unittest.TestCase):
    def test_extract_json_from_markdown(self):
        parsed = extract_json_object('```json\n{"action": "NOOP", "action_id": 8}\n```')
        self.assertEqual(parsed["action_id"], 8)

    def test_decision_action_name(self):
        decision = LLMDecision.from_dict({"action": "MOVE_RIGHT"})
        self.assertEqual(decision.action_id, 3)

    def test_client_ignores_unexpanded_env_placeholders(self):
        client = DeepSeekClient(base_url="${DEEPSEEK_BASE_URL}", model="${DEEPSEEK_MODEL}")
        self.assertEqual(client.base_url, "https://api.deepseek.com")
        self.assertEqual(client.model, "deepseek-chat")

    def test_decision_accepts_string_preferred_action(self):
        decision = LLMDecision.from_dict({"mode": "plan", "preferred_actions": "MOVE_LEFT"})
        self.assertEqual(decision.first_preferred_action_id(), 2)


if __name__ == "__main__":
    unittest.main()
