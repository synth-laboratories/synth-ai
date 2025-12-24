"""Financial NER business logic: dataset and scoring (no synth-ai dependencies)."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel


# Dataset configuration
DATASET_NAME = "financial_ner"
DEFAULT_SPLIT = "train"
AVAILABLE_SPLITS: tuple[str, ...] = ("train", "validation")
TOOL_NAME = "extract_entities"

# Entity types for Financial NER
ENTITY_TYPES = ["Company", "Date", "Location", "Money", "Person", "Product", "Quantity"]


def _compute_repo_root() -> Path:
    """Compute the repository root path."""
    p = Path(__file__).resolve()
    parents = list(p.parents)
    if len(parents) >= 4:
        return parents[3]
    if "/opt/synth_ai_repo" in os.getenv("PYTHONPATH", "") or Path("/opt/synth_ai_repo/synth_ai").exists():
        return Path("/opt/synth_ai_repo")
    return Path.cwd()


REPO_ROOT = _compute_repo_root()


class ExtractRequest(BaseModel):
    """Request model for extraction endpoint."""
    text: str


class ExtractResponse(BaseModel):
    """Response model for extraction endpoint."""
    entities: dict[str, list[str]]


class FinancialNERDataset:
    """In-memory dataset for Financial NER task loading from HuggingFace FIRE dataset."""

    CACHE_FILE = Path(__file__).parent / ".fire_dataset_cache.json"
    HF_DATASET_NAME = "Cleanlab/fire-financial-ner-extraction"

    def __init__(self) -> None:
        self._train_examples: list[dict[str, Any]] = []
        self._validation_examples: list[dict[str, Any]] = []
        self._loaded = False

    def _load_dataset(self) -> None:
        """Load FIRE dataset from cache or HuggingFace."""
        if self._loaded:
            return

        # Try loading from cache first
        if self.CACHE_FILE.exists():
            print(f"[financial_ner_task_app] Loading dataset from cache: {self.CACHE_FILE}", flush=True)
            with open(self.CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
                self._train_examples = cached.get("train", [])
                self._validation_examples = cached.get("validation", [])
                self._loaded = True
                print(f"[financial_ner_task_app] Loaded from cache: {len(self._train_examples)} train, {len(self._validation_examples)} validation examples", flush=True)
                return

        # Load from HuggingFace
        print(f"[financial_ner_task_app] Downloading FIRE dataset from HuggingFace: {self.HF_DATASET_NAME}", flush=True)
        try:
            from datasets import load_dataset
            import ast

            hf_dataset = load_dataset(self.HF_DATASET_NAME, split="train")
            print(f"[financial_ner_task_app] Downloaded {len(hf_dataset)} examples from HuggingFace", flush=True)

            all_examples = []
            for idx, example in enumerate(hf_dataset):
                text = example.get("text", "")
                ground_truth_str = example.get("ground_truth", "")

                if not text or not ground_truth_str:
                    continue

                # Parse ground_truth string
                try:
                    ground_truth = ast.literal_eval(ground_truth_str)
                except Exception:
                    continue

                if not isinstance(ground_truth, dict):
                    continue

                # Normalize entities
                entities = {}
                for key in ENTITY_TYPES:
                    value = ground_truth.get(key)
                    if value is None:
                        entities[key] = []
                    elif isinstance(value, list):
                        entities[key] = [str(v) for v in value if v]
                    else:
                        entities[key] = [str(value)] if value else []

                all_examples.append({"text": text, "entities": entities})

            # Split 80/20 for train/validation
            split_idx = int(len(all_examples) * 0.8)
            self._train_examples = all_examples[:split_idx]
            self._validation_examples = all_examples[split_idx:]

            # Cache to file
            print(f"[financial_ner_task_app] Caching dataset to: {self.CACHE_FILE}", flush=True)
            with open(self.CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "train": self._train_examples,
                    "validation": self._validation_examples,
                }, f, ensure_ascii=False)

            self._loaded = True
            print(f"[financial_ner_task_app] Dataset ready: {len(self._train_examples)} train, {len(self._validation_examples)} validation examples", flush=True)

        except ImportError:
            raise RuntimeError("datasets library required. Install with: pip install datasets")

    def _get_examples(self, split: str) -> list[dict[str, Any]]:
        """Get examples for the given split."""
        self._load_dataset()
        if split == "validation" or split == "test":
            return self._validation_examples
        return self._train_examples

    def size(self, split: str) -> int:
        """Get the number of examples in a split."""
        return len(self._get_examples(split))

    def sample(self, *, split: str, index: int) -> dict[str, Any]:
        """Get a sample from the dataset by index."""
        examples = self._get_examples(split)
        size = len(examples)
        if size == 0:
            raise RuntimeError(f"Financial NER split '{split}' is empty")
        idx = int(index) % size
        example = examples[idx]

        return {
            "index": idx,
            "split": split,
            "text": example["text"],
            "entities": example["entities"],
        }

    def ensure_ready(self, splits: Sequence[str]) -> None:
        """Preload dataset."""
        self._load_dataset()


class FinancialNERScorer:
    """Scorer for Financial NER entity extraction."""

    @staticmethod
    def score_entities(predicted: dict[str, list[str]], expected: dict[str, list[str]]) -> tuple[int, int, float]:
        """
        Score predicted entities against expected.
        
        Returns:
            Tuple of (correct_types, total_types, reward)
        """
        correct_types = 0
        for etype in ENTITY_TYPES:
            expected_set = set(expected.get(etype, []))
            predicted_set = set(predicted.get(etype, []))
            if expected_set == predicted_set:
                correct_types += 1

        reward = correct_types / len(ENTITY_TYPES)
        return correct_types, len(ENTITY_TYPES), reward


def get_extract_tool_schema() -> dict[str, Any]:
    """Get the tool schema for entity extraction."""
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Extract named entities from the financial text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "object",
                        "properties": {
                            etype: {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": f"{etype} entities found in the text"
                            }
                            for etype in ENTITY_TYPES
                        },
                        "required": ENTITY_TYPES,
                    }
                },
                "required": ["entities"],
            },
        },
    }


def get_default_messages_templates() -> list[dict[str, str]]:
    """Get the default message templates for the NER task."""
    return [
        {
            "role": "system",
            "pattern": (
                "You are an expert at extracting named entities from financial news. "
                "Extract all entities of the specified types from the text. Return the results as JSON using the `extract_entities` tool."
            ),
        },
        {
            "role": "user",
            "pattern": (
                "Extract entities from the following financial text. Extract these entity types: {entity_types}\n\n"
                "Text: {text}\n\n"
                "Use the extract_entities tool to return a JSON object with entity type keys and lists of extracted entity values."
            ),
        },
    ]


def parse_entities_from_tool_call(args_str: str) -> dict[str, list[str]]:
    """Parse entities from a tool call arguments string."""
    predicted_entities: dict[str, list[str]] = {etype: [] for etype in ENTITY_TYPES}
    
    try:
        args = json.loads(args_str)
        entities_data = args.get("entities", {})
        if isinstance(entities_data, dict):
            for etype in ENTITY_TYPES:
                value = entities_data.get(etype, [])
                if isinstance(value, str):
                    predicted_entities[etype] = [value]
                elif isinstance(value, list):
                    predicted_entities[etype] = value
                else:
                    predicted_entities[etype] = []
    except Exception:
        pass
    
    return predicted_entities

