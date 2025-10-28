from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from synth_ai.environments.tasks.core import (
    Impetus,
    Intent,
    SplitInfo,
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceSet,
)

from .engine import DEFAULT_SOLUTIONS


@dataclass
class WordleTaskInstanceMetadata(TaskInstanceMetadata):
    word_length: int
    max_guesses: int
    target_word: str
    enforce_wordlist: bool
    seed: int | None = None
    consume_invalid_attempts: bool = True


@dataclass
class WordleTaskInstance(TaskInstance):
    async def serialize(self) -> dict:
        return {
            "id": str(self.id),
            "impetus": {"instructions": self.impetus.instructions},
            "intent": {
                "rubric": self.intent.rubric,
                "gold_trajectories": self.intent.gold_trajectories,
                "gold_state_diff": self.intent.gold_state_diff,
            },
            "metadata": {
                "word_length": self.metadata.word_length,
                "max_guesses": self.metadata.max_guesses,
                "target_word": self.metadata.target_word,
                "enforce_wordlist": self.metadata.enforce_wordlist,
                "seed": self.metadata.seed,
                "consume_invalid_attempts": self.metadata.consume_invalid_attempts,
            },
            "is_reproducible": self.is_reproducible,
            "initial_engine_snapshot": self.initial_engine_snapshot,
        }

    @classmethod
    async def deserialize(cls, data: dict) -> WordleTaskInstance:
        from uuid import UUID

        metadata = WordleTaskInstanceMetadata(
            word_length=data["metadata"]["word_length"],
            max_guesses=data["metadata"]["max_guesses"],
            target_word=data["metadata"]["target_word"],
            enforce_wordlist=data["metadata"]["enforce_wordlist"],
            seed=data["metadata"].get("seed"),
            consume_invalid_attempts=data["metadata"].get("consume_invalid_attempts", True),
        )

        return cls(
            id=UUID(data["id"]),
            impetus=Impetus(instructions=data["impetus"]["instructions"]),
            intent=Intent(
                rubric=data["intent"]["rubric"],
                gold_trajectories=data["intent"]["gold_trajectories"],
                gold_state_diff=data["intent"]["gold_state_diff"],
            ),
            metadata=metadata,
            is_reproducible=data["is_reproducible"],
            initial_engine_snapshot=data["initial_engine_snapshot"],
        )


def _stable_uuid_for_instance(idx: int, target: str) -> UUID:
    import uuid

    return uuid.uuid5(uuid.NAMESPACE_URL, f"wordle-fixed-v1:{idx}:{target}")


def _load_fixed_instances_json() -> tuple[list[dict], dict]:
    """Load fixed instances definition from instances.json (if present).

    Returns a tuple (instances, defaults) where instances is a list of dicts with at least
    target_word fields, and defaults contains default params.
    """
    import os

    # Allow override via env var
    override = os.getenv("WORDLE_INSTANCES_JSON")
    p = Path(override) if override else Path(__file__).with_name("instances.json")
    if not p.exists():
        return [], {}
    try:
        data = json.loads(p.read_text())
        defaults = data.get("defaults", {}) or {}
        insts = data.get("instances", []) or []
        return insts, defaults
    except Exception:
        return [], {}


# Note: generation helpers removed from runtime. Use the provided script in tools/
_ = None


async def create_wordle_taskset(
    *,
    word_length: int = 5,
    max_guesses: int = 6,
    enforce_wordlist: bool = False,
    sample_size: int = 30,
    consume_invalid_attempts: bool = True,
) -> TaskInstanceSet:
    """Create a Wordle taskset.

    Priority:
    1) If instances.json exists, use it to produce a fixed, stable taskset with deterministic IDs.
    2) Otherwise, fall back to a procedural slice of DEFAULT_SOLUTIONS (stable ordering).
    """

    json_insts, json_defaults = _load_fixed_instances_json()

    instances: list[WordleTaskInstance] = []
    # Assemble fixed targets from JSON only (no runtime generation)
    fixed_targets: list[str] = []
    if json_insts:
        fixed_targets.extend(
            [
                str(r.get("target_word", "")).strip().lower()
                for r in json_insts
                if r.get("target_word")
            ]
        )

    if fixed_targets:
        # Use fixed_targets, honoring defaults and slicing by sample_size
        chosen = fixed_targets[:sample_size]
        for i, tgt in enumerate(chosen):
            md = WordleTaskInstanceMetadata(
                word_length=int(word_length),
                max_guesses=int(max_guesses),
                target_word=tgt,
                enforce_wordlist=bool(enforce_wordlist),
                seed=i,
                consume_invalid_attempts=bool(consume_invalid_attempts),
            )
            impetus = Impetus(
                instructions=(
                    "Play Wordle. Submit one word per turn consisting only of letters. "
                    f"You have up to {md.max_guesses} guesses to find the {md.word_length}-letter target word. "
                    "Feedback per letter: G=correct position, Y=present elsewhere, B=absent."
                )
            )
            intent = Intent(
                rubric={"goal": "Guess the target word in as few moves as possible"},
                gold_trajectories=None,
                gold_state_diff={"target_known": False},
            )
            inst = WordleTaskInstance(
                id=_stable_uuid_for_instance(i, md.target_word),
                impetus=impetus,
                intent=intent,
                metadata=md,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )
            instances.append(inst)
    else:
        # Procedural fallback: stable ordering from DEFAULT_SOLUTIONS
        pool = [w for w in DEFAULT_SOLUTIONS if len(w) == word_length] or [
            w for w in DEFAULT_SOLUTIONS if len(w) == 5
        ]
        sample = pool[:sample_size]
        for i, target in enumerate(sample):
            seed = i
            md = WordleTaskInstanceMetadata(
                word_length=word_length,
                max_guesses=max_guesses,
                target_word=target,
                enforce_wordlist=enforce_wordlist,
                seed=seed,
                consume_invalid_attempts=consume_invalid_attempts,
            )
            impetus = Impetus(
                instructions=(
                    "Play Wordle. Submit one word per turn consisting only of letters. "
                    f"You have up to {max_guesses} guesses to find the {word_length}-letter target word. "
                    "Feedback per letter: G=correct position, Y=present elsewhere, B=absent."
                )
            )
            intent = Intent(
                rubric={"goal": "Guess the target word in as few moves as possible"},
                gold_trajectories=None,
                gold_state_diff={"target_known": False},
            )
            inst = WordleTaskInstance(
                id=_stable_uuid_for_instance(i, target),
                impetus=impetus,
                intent=intent,
                metadata=md,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )
            instances.append(inst)

    # Deterministic split based on index positions
    val_ids = {instances[i].id for i in range(0, len(instances), 5)}
    test_ids = {instances[i].id for i in range(0, len(instances), 7)}
    split = SplitInfo(val_instance_ids=val_ids, test_instance_ids=test_ids, _is_split_defined=True)

    return TaskInstanceSet(
        name="Wordle Fixed TaskSet" if json_insts else "Wordle Example TaskSet",
        description=(
            "Fixed set from instances.json (stable ordering)."
            if json_insts
            else "Lightweight Wordle tasks with fixed targets and seeds."
        ),
        instances=instances,
        split_info=split,
    )


# Alias
taskset = create_wordle_taskset
