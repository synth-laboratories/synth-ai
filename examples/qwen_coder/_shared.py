#!/usr/bin/env python3
"""Shared helpers for Qwen coder SFT examples."""

from __future__ import annotations

import json
import os
from pathlib import Path

TRAIN_DATA_PATH = Path("examples/qwen_coder/ft_data/coder_sft.small.jsonl")
VAL_DATA_PATH = Path("examples/qwen_coder/ft_data/coder_sft.small.val.jsonl")
DATA_DIR = TRAIN_DATA_PATH.parent

_FALLBACK_RECORDS: list[dict[str, object]] = [
    {
        "messages": [
            {"role": "user", "content": "Write a Python function `add(a, b)` that returns the sum of two numbers."}
        ],
        "response": "def add(a, b):\n    return a + b\n",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Implement a Python function `reverse_string(s)` that returns the reversed string.",
            }
        ],
        "response": "def reverse_string(s: str) -> str:\n    return s[::-1]\n",
    },
    {
        "messages": [
            {
                "role": "user",
                "content": "Write a Python function `count_words(text)` returning a dict mapping words to counts.",
            }
        ],
        "response": "from collections import Counter\n\ndef count_words(text: str) -> dict[str, int]:\n    words = [w for w in text.split() if w]\n    return dict(Counter(words))\n",
    },
]


def ensure_tiny_dataset() -> Path:
    """Ensure the tiny coder dataset exists, generating or writing a fallback if needed."""
    if TRAIN_DATA_PATH.exists():
        return TRAIN_DATA_PATH

    try:
        from examples.qwen_coder.generate_dataset import main as gen_main  # type: ignore

        gen_main()
        if TRAIN_DATA_PATH.exists():
            return TRAIN_DATA_PATH
    except Exception:
        # Fall back to inline dataset below.
        pass

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with TRAIN_DATA_PATH.open("w", encoding="utf-8") as fh:
        for record in _FALLBACK_RECORDS:
            fh.write(json.dumps(record, separators=(",", ":")))
            fh.write("\n")
    return TRAIN_DATA_PATH


def optional_validation_dataset() -> Path | None:
    """Return validation dataset path if present."""
    if VAL_DATA_PATH.exists():
        return VAL_DATA_PATH
    return None


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_path(default_filename: str) -> Path:
    """Resolve output path for storing fine-tuned model ids."""
    override = os.getenv("QWEN_CODER_FT_OUTPUT")
    if override:
        return _ensure_parent(Path(override).expanduser())
    return _ensure_parent(DATA_DIR / default_filename)


def resolve_model_id_path(default_filename: str) -> Path:
    """Resolve path to read a stored fine-tuned model id."""
    override = os.getenv("QWEN_CODER_FT_MODEL_PATH")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_dir():
            return candidate / default_filename
        return candidate
    return DATA_DIR / default_filename


def resolve_infer_output_path(default_filename: str) -> Path:
    """Resolve path for writing inference outputs."""
    override = os.getenv("QWEN_CODER_FT_INFER_OUTPUT")
    if override:
        return _ensure_parent(Path(override).expanduser())
    return _ensure_parent(DATA_DIR / default_filename)


__all__ = [
    "DATA_DIR",
    "TRAIN_DATA_PATH",
    "VAL_DATA_PATH",
    "ensure_tiny_dataset",
    "optional_validation_dataset",
    "resolve_output_path",
    "resolve_model_id_path",
    "resolve_infer_output_path",
]
