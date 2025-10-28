from __future__ import annotations

import random
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any

from synth_ai.environments.environment.rewards.core import RewardComponent, RewardStack
from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.reproducibility.core import IReproducibleEngine
from synth_ai.environments.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from synth_ai.environments.tasks.core import TaskInstance

DEFAULT_SOLUTIONS = [
    "cigar",
    "rebut",
    "sissy",
    "humph",
    "awake",
    "blush",
    "focal",
    "evade",
    "naval",
    "serve",
    "heath",
    "dwarf",
    "model",
    "karma",
    "stink",
    "grade",
    "quiet",
    "bench",
    "abate",
    "feign",
    "major",
    "death",
    "fresh",
    "crust",
    "stool",
    "colon",
    "abase",
    "marry",
    "react",
    "batty",
    "pride",
    "floss",
    "helix",
    "croak",
    "staff",
    "paper",
    "unfed",
    "whelp",
    "trawl",
    "outdo",
    "adobe",
    "crazy",
    "sower",
    "repay",
    "digit",
    "crate",
    "cluck",
    "spike",
    "mimic",
    "pound",
]


def _sanitize(word: str) -> str:
    w = word.strip().lower()
    if not w or not all(c in string.ascii_lowercase for c in w):
        raise ValueError("word must contain only a–z letters")
    return w


def _score_guess(guess: str, target: str) -> str:
    res = ["B"] * len(target)
    counts = Counter(target)
    for i, ch in enumerate(guess):
        if ch == target[i]:
            res[i] = "G"
            counts[ch] -= 1
    for i, ch in enumerate(guess):
        if res[i] == "G":
            continue
        if counts.get(ch, 0) > 0:
            res[i] = "Y"
            counts[ch] -= 1
    return "".join(res)


@dataclass
class WordlePublicState:
    word_length: int
    remaining_guesses: int
    max_guesses: int
    guesses: list[str]
    feedback: list[str]  # Parallel to guesses; strings of 'G/Y/B'
    last_feedback: str | None
    last_guess: str | None
    terminated: bool
    status: str  # "in_progress" | "won" | "lost"

    @property
    def board_text(self) -> str:
        if not self.guesses:
            return "(no guesses yet)"
        lines = []
        for g, fb in zip(self.guesses, self.feedback, strict=False):
            spaced = " ".join(list(fb))
            lines.append(f"{g.upper()} | {spaced}")
        return "\n".join(lines)


@dataclass
class WordlePrivateState:
    reward_last: float
    total_reward: float
    terminated: bool
    truncated: bool


@dataclass
class WordleEngineSnapshot(StatefulEngineSnapshot):
    task_instance_dict: dict
    engine_snapshot: dict


class WordleWinComponent(RewardComponent):
    async def score(self, state: WordlePublicState, action: Any) -> float:
        return 1.0 if state.status == "won" else 0.0


class WordleInvalidGuessComponent(RewardComponent):
    def __init__(self) -> None:
        self.invalid_attempted = False

    async def score(self, state: WordlePublicState, action: Any) -> float:
        if self.invalid_attempted:
            self.invalid_attempted = False
            return -1.0
        return 0.0


class WordleEngine(StatefulEngine, IReproducibleEngine):
    def __init__(self, task_instance: TaskInstance):
        self.task_instance = task_instance

        # Read config from metadata
        md = getattr(task_instance, "metadata", None)
        self.word_length: int = getattr(md, "word_length", 5) if md else 5
        self.max_guesses: int = getattr(md, "max_guesses", 6) if md else 6
        self.enforce_wordlist: bool = getattr(md, "enforce_wordlist", False) if md else False
        # Toggle: whether invalid actions consume a turn (default True)
        self.consume_invalid_attempts: bool = (
            getattr(md, "consume_invalid_attempts", True) if md else True
        )

        self.base_word_list: list[str] = [
            w for w in DEFAULT_SOLUTIONS if len(w) == self.word_length
        ] or [w for w in DEFAULT_SOLUTIONS if len(w) == 5]

        # Target selection: prefer explicit target_word in metadata; else pick deterministically by seed
        self.fixed_target: str | None = (
            _sanitize(getattr(md, "target_word", ""))
            if md and getattr(md, "target_word", None)
            else None
        )
        self.seed: int | None = getattr(md, "seed", None) if md else None

        # Runtime state
        self.target: str | None = None
        self.guesses: list[str] = []
        self.feedback: list[str] = []
        self.remaining_guesses: int = self.max_guesses
        self.status: str = "in_progress"
        self.terminated: bool = False
        self.total_reward: float = 0.0

        # Rewards
        self.invalid_component = WordleInvalidGuessComponent()
        self.reward_stack = RewardStack([WordleWinComponent(), self.invalid_component])

    async def _reset_engine(
        self, *, seed: int | None = None
    ) -> tuple[WordlePrivateState, WordlePublicState]:
        if seed is None:
            seed = self.seed
        if seed is not None and self.fixed_target is None:
            random.seed(seed)
        self.target = self.fixed_target or random.choice(self.base_word_list)
        self.guesses = []
        self.feedback = []
        self.remaining_guesses = self.max_guesses
        self.status = "in_progress"
        self.terminated = False
        self.total_reward = 0.0

        pub = WordlePublicState(
            word_length=self.word_length,
            remaining_guesses=self.remaining_guesses,
            max_guesses=self.max_guesses,
            guesses=[],
            feedback=[],
            last_feedback=None,
            last_guess=None,
            terminated=False,
            status=self.status,
        )
        priv = WordlePrivateState(
            reward_last=0.0,
            total_reward=0.0,
            terminated=False,
            truncated=False,
        )
        return priv, pub

    async def _step_engine(self, action: str) -> tuple[WordlePrivateState, WordlePublicState]:
        assert self.target is not None
        guess = _sanitize(action)

        # Validate
        if len(guess) != self.word_length or (
            self.enforce_wordlist and guess not in self.base_word_list
        ):
            # Penalize invalid action; do not consume a guess
            self.invalid_component.invalid_attempted = True
            if self.consume_invalid_attempts:
                # consume a turn on invalid guesses
                if self.remaining_guesses > 0:
                    self.remaining_guesses -= 1
                if self.remaining_guesses == 0:
                    self.status = "lost"
                    self.terminated = True
            pub = WordlePublicState(
                word_length=self.word_length,
                remaining_guesses=self.remaining_guesses,
                max_guesses=self.max_guesses,
                guesses=self.guesses.copy(),
                feedback=self.feedback.copy(),
                last_feedback=self.feedback[-1] if self.feedback else None,
                last_guess=self.guesses[-1] if self.guesses else None,
                terminated=self.terminated,
                status=self.status,
            )
            reward = await self.reward_stack.step_reward(pub, action)
            self.total_reward += reward
            priv = WordlePrivateState(
                reward_last=reward,
                total_reward=self.total_reward,
                terminated=self.terminated,
                truncated=False,
            )
            return priv, pub

        fb = _score_guess(guess, self.target)
        self.guesses.append(guess)
        self.feedback.append(fb)
        self.remaining_guesses -= 1

        if guess == self.target:
            self.status = "won"
            self.terminated = True
        elif self.remaining_guesses == 0:
            self.status = "lost"
            self.terminated = True
        else:
            self.status = "in_progress"

        pub = WordlePublicState(
            word_length=self.word_length,
            remaining_guesses=self.remaining_guesses,
            max_guesses=self.max_guesses,
            guesses=self.guesses.copy(),
            feedback=self.feedback.copy(),
            last_feedback=fb,
            last_guess=guess,
            terminated=self.terminated,
            status=self.status,
        )

        reward = await self.reward_stack.step_reward(pub, action)
        self.total_reward += reward
        priv = WordlePrivateState(
            reward_last=reward,
            total_reward=self.total_reward,
            terminated=self.terminated,
            truncated=False,
        )
        return priv, pub

    async def _serialize_engine(self) -> WordleEngineSnapshot:
        return WordleEngineSnapshot(
            task_instance_dict=await self.task_instance.serialize(),
            engine_snapshot={
                "word_length": self.word_length,
                "max_guesses": self.max_guesses,
                "enforce_wordlist": self.enforce_wordlist,
                "consume_invalid_attempts": self.consume_invalid_attempts,
                "base_word_list": self.base_word_list,
                "fixed_target": self.fixed_target,
                "seed": self.seed,
                "target": self.target,
                "guesses": self.guesses,
                "feedback": self.feedback,
                "remaining_guesses": self.remaining_guesses,
                "status": self.status,
                "terminated": self.terminated,
                "total_reward": self.total_reward,
            },
        )

    @classmethod
    async def _deserialize_engine(cls, snapshot: WordleEngineSnapshot) -> WordleEngine:
        task_instance = await TaskInstance.deserialize(snapshot.task_instance_dict)
        engine = cls(task_instance)
        s = snapshot.engine_snapshot
        engine.word_length = s["word_length"]
        engine.max_guesses = s["max_guesses"]
        engine.enforce_wordlist = s["enforce_wordlist"]
        engine.consume_invalid_attempts = s.get("consume_invalid_attempts", True)
        engine.base_word_list = s.get("base_word_list", engine.base_word_list)
        engine.fixed_target = s.get("fixed_target")
        engine.seed = s.get("seed")
        engine.target = s.get("target")
        engine.guesses = s.get("guesses", [])
        engine.feedback = s.get("feedback", [])
        engine.remaining_guesses = s.get("remaining_guesses", engine.max_guesses)
        engine.status = s.get("status", "in_progress")
        engine.terminated = s.get("terminated", False)
        engine.total_reward = s.get("total_reward", 0.0)
        return engine

    def get_current_states_for_observation(self) -> tuple[WordlePrivateState, WordlePublicState]:
        pub = WordlePublicState(
            word_length=self.word_length,
            remaining_guesses=self.remaining_guesses,
            max_guesses=self.max_guesses,
            guesses=self.guesses.copy(),
            feedback=self.feedback.copy(),
            last_feedback=self.feedback[-1] if self.feedback else None,
            last_guess=self.guesses[-1] if self.guesses else None,
            terminated=self.terminated,
            status=self.status,
        )
        priv = WordlePrivateState(
            reward_last=0.0,
            total_reward=self.total_reward,
            terminated=self.terminated,
            truncated=False,
        )
        return priv, pub


class SynthWordleObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: WordlePublicState, priv: WordlePrivateState
    ) -> InternalObservation:
        header = f"WORDLE ({pub.word_length} letters, {pub.max_guesses} max guesses)"
        lines = [
            header,
            "Submit a single English word (letters only).",
            "",
            pub.board_text,
            "",
        ]
        if pub.status == "in_progress":
            lines.append(f"You have {pub.remaining_guesses} guesses left.")
        elif pub.status == "won":
            lines.append("You guessed the word! ✅")
        else:
            lines.append("Out of guesses. ❌")

        return {
            "text": "\n".join(lines),
            "status": pub.status,
            "remaining_guesses": pub.remaining_guesses,
            "guesses": pub.guesses,
            "feedback": pub.feedback,
            "reward_last": priv.reward_last,
            "total_reward": priv.total_reward,
            "terminated": pub.terminated,
        }


class SynthWordleCheckpointObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: WordlePublicState, priv: WordlePrivateState
    ) -> InternalObservation:
        return {
            "board_text_final": pub.board_text,
            "status_final": pub.status,
            "total_reward": priv.total_reward,
            "terminated": pub.terminated,
        }
