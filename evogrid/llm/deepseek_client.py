"""Small DeepSeek chat client using the stdlib HTTP stack."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from evogrid.llm.parser import extract_json_object


class DeepSeekClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        max_tokens: int | None = None,
        json_mode: bool = True,
    ):
        self.api_key = _clean_value(api_key or os.getenv("DEEPSEEK_API_KEY"))
        self.base_url = _clean_value(
            base_url or os.getenv("DEEPSEEK_BASE_URL"),
            "https://api.deepseek.com",
        ).rstrip("/")
        self.model = _clean_value(model or os.getenv("DEEPSEEK_MODEL"), "deepseek-chat")
        self.timeout = int(_clean_value(timeout or os.getenv("DEEPSEEK_TIMEOUT"), 30))
        self.max_tokens = int(_clean_value(max_tokens or os.getenv("DEEPSEEK_MAX_TOKENS"), 512))
        self.json_mode = json_mode

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        json_mode: bool | None = None,
    ) -> str:
        return self.chat_completion(
            messages=messages,
            temperature=temperature,
            json_mode=json_mode,
        )["content"]

    def chat_completion(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        json_mode: bool | None = None,
    ) -> dict:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        use_json_mode = self.json_mode if json_mode is None else json_mode
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DeepSeek request failed: {exc}") from exc

        parsed = json.loads(body)
        choice = parsed["choices"][0]
        message = choice["message"]
        return {
            "content": message.get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "model": parsed.get("model"),
            "usage": parsed.get("usage", {}),
        }

    def chat_json(self, messages: list[dict], temperature: float = 0.0) -> dict:
        return extract_json_object(self.chat(messages, temperature=temperature))


def _clean_value(value, default=None):
    if value is None:
        return default
    text = str(value).strip()
    if not text or (text.startswith("${") and text.endswith("}")):
        return default
    return text
