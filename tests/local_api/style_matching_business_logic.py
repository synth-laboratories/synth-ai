"""Style matching business logic: dataset and helpers (no synth-ai dependencies)."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any


DATASET_NAME = "style_matching"
DEFAULT_SPLIT = "default"
AVAILABLE_SPLITS: tuple[str, ...] = (DEFAULT_SPLIT,)
TOOL_NAME = "submit_essay"


def _compute_repo_root() -> Path:
    p = Path(__file__).resolve()
    parents = list(p.parents)
    if len(parents) >= 4:
        return parents[3]
    if "/opt/synth_ai_repo" in os.getenv("PYTHONPATH", "") or Path("/opt/synth_ai_repo/synth_ai").exists():
        return Path("/opt/synth_ai_repo")
    return Path.cwd()


REPO_ROOT = _compute_repo_root()
DATASET_PATH = REPO_ROOT / "cookbooks" / "products" / "graphgen" / "style_matching_dataset.json"


class StyleMatchingDataset:
    """In-memory dataset loader for the style matching cookbook."""

    def __init__(self) -> None:
        self._loaded = False
        self._tasks: list[dict[str, Any]] = []
        self._gold_outputs: list[dict[str, Any]] = []
        self._initial_prompt: str = ""
        self._input_schema: dict[str, Any] = {}
        self._output_schema: dict[str, Any] = {}

    def _load(self) -> None:
        if self._loaded:
            return
        if not DATASET_PATH.exists():
            raise RuntimeError(f"Style matching dataset not found: {DATASET_PATH}")
        raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        self._tasks = list(raw.get("tasks") or [])
        self._gold_outputs = list(raw.get("gold_outputs") or [])
        self._initial_prompt = str(raw.get("initial_prompt") or "").strip()
        self._input_schema = raw.get("input_schema") or raw.get("metadata", {}).get("input_schema") or {}
        self._output_schema = raw.get("output_schema") or raw.get("metadata", {}).get("output_schema") or {}
        self._loaded = True

    def ensure_ready(self, splits: Sequence[str]) -> None:
        self._load()

    def size(self, split: str) -> int:
        self._load()
        return len(self._tasks)

    def sample(self, *, split: str, index: int) -> dict[str, Any]:
        self._load()
        size = len(self._tasks)
        if size == 0:
            raise RuntimeError("Style matching dataset is empty")
        idx = int(index) % size
        task = self._tasks[idx]
        payload = task.get("input") or {}
        return {
            "index": idx,
            "split": split,
            "task_id": str(task.get("id", f"task_{idx}")),
            "outline": str(payload.get("outline", "")),
            "topic": str(payload.get("topic", "")),
            "notes": list(payload.get("notes") or []),
        }

    def describe(self) -> dict[str, Any]:
        self._load()
        return {
            "dataset_name": DATASET_NAME,
            "size": len(self._tasks),
            "input_schema": self._input_schema,
            "output_schema": self._output_schema,
            "initial_prompt": self._initial_prompt,
        }

    @property
    def initial_prompt(self) -> str:
        self._load()
        return self._initial_prompt

    @property
    def input_schema(self) -> dict[str, Any]:
        self._load()
        return self._input_schema

    @property
    def output_schema(self) -> dict[str, Any]:
        self._load()
        return self._output_schema

    @property
    def gold_outputs(self) -> list[dict[str, Any]]:
        self._load()
        return list(self._gold_outputs)

    @property
    def tasks(self) -> list[dict[str, Any]]:
        self._load()
        return list(self._tasks)

    def task_by_id(self, task_id: str) -> dict[str, Any] | None:
        self._load()
        for task in self._tasks:
            if str(task.get("id", "")) == task_id:
                return task
        return None


def format_notes(notes: Sequence[str]) -> str:
    cleaned = [str(note).strip() for note in notes if str(note).strip()]
    if not cleaned:
        return "- (none)"
    return "\n".join(f"- {note}" for note in cleaned)


def get_default_messages_templates() -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "pattern": "{system_prompt}",
        },
        {
            "role": "user",
            "pattern": (
                "Outline: {outline}\n"
                "Topic: {topic}\n"
                "Notes:\n{notes}\n\n"
                "Write a short, punchy essay that matches the requested style. "
                "Return the result via the submit_essay tool."
            ),
        },
    ]


def get_submit_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Submit the essay title and content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["title", "content"],
            },
        },
    }


def parse_essay_from_tool_call(args_str: str) -> dict[str, str]:
    try:
        payload = json.loads(args_str)
    except Exception:
        payload = {}
    title = payload.get("title") if isinstance(payload, dict) else ""
    content = payload.get("content") if isinstance(payload, dict) else ""
    return {
        "title": str(title or "").strip(),
        "content": str(content or "").strip(),
    }
