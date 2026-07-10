"""Placeholder for GIF/MP4 generation."""

from __future__ import annotations


def frames_to_text(frames: list[str]) -> str:
    return "\n\n---\n\n".join(frames)


def save_frames_text(path, frames: list[str]):
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frames_to_text(frames), encoding="utf-8")
    return path
