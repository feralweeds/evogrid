"""Configuration loading helpers."""

from __future__ import annotations

import os
from pathlib import Path


def load_yaml(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load YAML configs.") from exc
    raw = Path(path).read_text(encoding="utf-8")
    raw = _expand_env(raw)
    data = yaml.safe_load(raw)
    return data or {}


def _expand_env(text: str) -> str:
    for key, value in os.environ.items():
        text = text.replace("${" + key + "}", value)
    return text

