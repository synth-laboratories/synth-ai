"""Banking77 business logic: dataset API and scoring (no synth-ai dependencies)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# Dataset configuration
DATASET_NAME = os.getenv("BANKING77_DATASET_NAME", "banking77")
DEFAULT_SPLIT = "train"
AVAILABLE_SPLITS: tuple[str, ...] = ("train", "test")
TOOL_NAME = "banking77_classify"


def _compute_repo_root() -> Path:
    """Compute the repository root path."""
    p = Path(__file__).resolve()
    parents = list(p.parents)
    if len(parents) >= 4:
        return parents[3]
    # Modal inline deploy: code may be at /root/*.py, but we mount synth_ai at /opt/synth_ai_repo/synth_ai
    if "/opt/synth_ai_repo" in os.getenv("PYTHONPATH", "") or Path("/opt/synth_ai_repo/synth_ai").exists():
        return Path("/opt/synth_ai_repo")
    return Path.cwd()


REPO_ROOT = _compute_repo_root()


class ClassifyRequest(BaseModel):
    """Request model for classification endpoint."""
    query: str


class ClassifyResponse(BaseModel):
    """Response model for classification endpoint."""
    intent: str
    confidence: float | None = None


class Banking77Dataset:
    """Lazy Hugging Face dataset loader for Banking77."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._label_names: list[str] | None = None

    def _load_split(self, split: str):
        if split not in AVAILABLE_SPLITS:
            raise ValueError(f"Unknown split: {split}. Available: {AVAILABLE_SPLITS}")
        if split not in self._cache:
            try:
                from datasets import load_dataset as _load_dataset  # lazy import
                
                # Normalize dataset name: use "banking77" (canonical name) instead of "PolyAI/banking77"
                dataset_name = DATASET_NAME
                if dataset_name == "PolyAI/banking77":
                    dataset_name = "banking77"
                
                hf_home = os.getenv("HF_HOME")
                hf_datasets_cache = os.getenv("HF_DATASETS_CACHE")
                hf_hub_cache = os.getenv("HF_HUB_CACHE")
                print(
                    f"[Banking77Dataset] Loading dataset '{dataset_name}' split '{split}' "
                    f"(HF_HOME={hf_home}, HF_DATASETS_CACHE={hf_datasets_cache}, HF_HUB_CACHE={hf_hub_cache})",
                    flush=True,
                )
                
                try:
                    ds = _load_dataset(
                        dataset_name,
                        split=split,
                        trust_remote_code=False,
                        download_mode="reuse_cache_if_exists",
                        num_proc=0,
                    )
                except Exception as cache_exc:
                    print(
                        f"[Banking77Dataset] Cache load failed, trying download: {cache_exc}",
                        flush=True,
                    )
                    ds = _load_dataset(
                        dataset_name,
                        split=split,
                        trust_remote_code=False,
                        num_proc=0,
                    )
                
                self._cache[split] = ds
                if self._label_names is None and hasattr(ds.features.get("label"), "names"):  # type: ignore[union-attr]
                    self._label_names = ds.features["label"].names  # type: ignore[union-attr]
                print(
                    f"[Banking77Dataset] Successfully loaded {len(ds)} examples from '{dataset_name}' split '{split}'",
                    flush=True,
                )
            except Exception as exc:
                import traceback
                error_details = traceback.format_exc()
                print(
                    f"[Banking77Dataset] Dataset load failed: {exc}\n{error_details}",
                    flush=True,
                )
                raise RuntimeError(
                    f"Dataset preparation failed: {split}: Failed to load Banking77 dataset. "
                    f"Dataset: {DATASET_NAME} | Split: {split} | Error: {exc}"
                ) from exc
        return self._cache[split]

    def ensure_ready(self, splits: Sequence[str]) -> None:
        """Preload dataset splits."""
        for split in splits:
            self._load_split(split)

    def size(self, split: str) -> int:
        """Get the number of examples in a split."""
        dataset = self._load_split(split)
        return len(dataset)

    def sample(self, *, split: str, index: int) -> dict[str, Any]:
        """Get a sample from the dataset by index."""
        dataset = self._load_split(split)
        size = len(dataset)
        if size == 0:
            raise RuntimeError(f"Banking77 split '{split}' is empty")
        idx = int(index) % size
        row = dataset[int(idx)]

        label_idx = int(row.get("label", 0))
        label_text = self.get_label_name(label_idx)

        return {
            "index": idx,
            "split": split,
            "text": str(row.get("text", "")),
            "label": label_text,
            "label_idx": label_idx,
        }

    def get_label_name(self, label_idx: int) -> str:
        """Convert label index to label name."""
        if self._label_names is None:
            self._load_split(DEFAULT_SPLIT)
        if self._label_names and 0 <= label_idx < len(self._label_names):
            return self._label_names[label_idx]
        return f"label_{label_idx}"

    @property
    def label_names(self) -> list[str]:
        """Get the list of all label names."""
        if self._label_names is None:
            self._load_split(DEFAULT_SPLIT)
        return self._label_names or []


class Banking77Scorer:
    """Scorer for Banking77 intent classification."""

    @staticmethod
    def normalize_intent(intent: str) -> str:
        """Normalize intent string for comparison."""
        return intent.lower().replace("_", " ").strip()

    @classmethod
    def score(cls, predicted: str, expected: str) -> tuple[bool, float]:
        """
        Score a prediction against expected intent.
        
        Returns:
            Tuple of (is_correct, reward)
        """
        predicted_normalized = cls.normalize_intent(predicted)
        expected_normalized = cls.normalize_intent(expected)
        is_correct = predicted_normalized == expected_normalized
        reward = 1.0 if is_correct else 0.0
        return is_correct, reward


def get_classify_tool_schema() -> dict[str, Any]:
    """Get the tool schema for banking77 classification."""
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Return the predicted banking77 intent label in the `intent` field.",
            "parameters": {
                "type": "object",
                "properties": {"intent": {"type": "string"}},
                "required": ["intent"],
            },
        },
    }


def get_default_messages_templates() -> list[dict[str, str]]:
    """Get the default message templates for the classification task."""
    return [
        {
            "role": "system",
            "pattern": (
                "You are an expert banking assistant that classifies customer queries into banking intents. "
                "Given a customer message, respond with exactly one intent label from the provided list using the `banking77_classify` tool."
            ),
        },
        {
            "role": "user",
            "pattern": "Customer Query: {query}\n\nAvailable Intents:\n{available_intents}\n\nClassify this query into one of the above banking intents using the tool call.",
        },
    ]


def format_available_intents(label_names: list[str]) -> str:
    """Format label names as a numbered list for the prompt."""
    return "\n".join(f"{i+1}. {label}" for i, label in enumerate(label_names))

