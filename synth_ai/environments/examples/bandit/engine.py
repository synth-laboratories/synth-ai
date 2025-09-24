from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.reproducibility.core import IReproducibleEngine
from synth_ai.environments.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from synth_ai.environments.tasks.core import TaskInstance


@dataclass
class BanditPublicState:
    arm_count: int
    step_count: int
    max_steps: int
    last_arm: int | None
    last_reward: float | None
    cumulative_reward: float
    reward_history: list[float]
    arm_pull_counts: list[int]
    terminated: bool
    status: str

    @property
    def steps_remaining(self) -> int:
        return max(0, self.max_steps - self.step_count)

    @property
    def average_reward(self) -> float:
        return self.cumulative_reward / self.step_count if self.step_count else 0.0


@dataclass
class BanditPrivateState:
    reward_last: float
    total_reward: float
    terminated: bool
    truncated: bool
    true_expected_rewards: list[float]
    step_count: int


@dataclass
class BanditEngineSnapshot(StatefulEngineSnapshot):
    task_instance_dict: dict[str, Any]
    engine_snapshot: dict[str, Any]


class BanditEngine(StatefulEngine, IReproducibleEngine):
    def __init__(self, task_instance: TaskInstance):
        self.task_instance = task_instance

        metadata = getattr(task_instance, "metadata", None)
        self.bandit_type: str = getattr(metadata, "bandit_type", "bernoulli")
        self.max_steps: int = int(getattr(metadata, "max_steps", 100))
        self.seed: int | None = getattr(metadata, "seed", None)

        self.arm_probabilities: list[float] | None = None
        self.arm_means: list[float] | None = None
        self.arm_stds: list[float] | None = None

        if self.bandit_type.lower() == "bernoulli":
            probs = list(getattr(metadata, "arm_probabilities", []) or [])
            if not probs:
                probs = [0.1, 0.5, 0.9]
            self.arm_probabilities = [float(p) for p in probs]
            self.true_expected_rewards = self.arm_probabilities.copy()
        elif self.bandit_type.lower() == "gaussian":
            means = list(getattr(metadata, "arm_means", []) or [])
            if not means:
                means = [0.0, 0.5, 1.0]
            stds = getattr(metadata, "arm_stds", None)
            if stds is None:
                stds_list = [0.1] * len(means)
            elif isinstance(stds, int | float):
                stds_list = [float(stds)] * len(means)
            else:
                stds_list = [float(s) for s in stds]
            if len(stds_list) != len(means):
                raise ValueError("arm_stds must match arm_means length")
            self.arm_means = [float(m) for m in means]
            self.arm_stds = stds_list
            self.true_expected_rewards = self.arm_means.copy()
        else:
            raise ValueError(f"Unsupported bandit_type: {self.bandit_type}")

        self.arm_count = len(self.true_expected_rewards)
        if self.arm_count == 0:
            raise ValueError("Bandit must have at least one arm")

        self._rng = np.random.default_rng(self.seed)

        # Runtime state
        self.step_count: int = 0
        self.total_reward: float = 0.0
        self.last_reward: float = 0.0
        self.last_arm: int | None = None
        self.reward_history: list[float] = []
        self.arm_history: list[int] = []
        self.arm_pull_counts: list[int] = [0 for _ in range(self.arm_count)]
        self.terminated: bool = False
        self.status: str = "in_progress"

    async def _reset_engine(self) -> tuple[BanditPrivateState, BanditPublicState]:
        if self.seed is not None:
            self._rng = np.random.default_rng(self.seed)
        else:
            self._rng = np.random.default_rng()

        self.step_count = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_arm = None
        self.reward_history = []
        self.arm_history = []
        self.arm_pull_counts = [0 for _ in range(self.arm_count)]
        self.terminated = False
        self.status = "in_progress"

        private_state = self._build_private_state(reward=0.0)
        public_state = self._build_public_state(reward=None)
        return private_state, public_state

    async def _step_engine(self, arm_index: int) -> tuple[BanditPrivateState, BanditPublicState]:
        if self.terminated:
            raise RuntimeError("Bandit episode already terminated")

        if not isinstance(arm_index, int):
            raise TypeError("arm_index must be an integer")
        if arm_index < 0 or arm_index >= self.arm_count:
            raise ValueError(f"arm_index {arm_index} out of range 0..{self.arm_count - 1}")

        reward = float(self._sample_reward(arm_index))

        self.step_count += 1
        self.total_reward += reward
        self.last_reward = reward
        self.last_arm = arm_index
        self.reward_history.append(reward)
        self.arm_history.append(arm_index)
        self.arm_pull_counts[arm_index] += 1

        if self.step_count >= self.max_steps:
            self.terminated = True
            self.status = "completed"

        private_state = self._build_private_state(reward=reward)
        public_state = self._build_public_state(reward=reward)
        return private_state, public_state

    def _sample_reward(self, arm_index: int) -> float:
        if self.bandit_type.lower() == "bernoulli":
            assert self.arm_probabilities is not None
            success_prob = self.arm_probabilities[arm_index]
            return 1.0 if self._rng.random() < success_prob else 0.0

        if self.bandit_type.lower() == "gaussian":
            assert self.arm_means is not None and self.arm_stds is not None
            mean = self.arm_means[arm_index]
            std = self.arm_stds[arm_index]
            return float(self._rng.normal(loc=mean, scale=std))

        raise RuntimeError(f"Unknown bandit_type during sampling: {self.bandit_type}")

    def get_current_states_for_observation(
        self,
    ) -> tuple[BanditPrivateState, BanditPublicState]:
        private_state = self._build_private_state(reward=self.last_reward)
        public_state = self._build_public_state(reward=self.last_reward)
        return private_state, public_state

    def _build_private_state(self, reward: float) -> BanditPrivateState:
        return BanditPrivateState(
            reward_last=float(reward),
            total_reward=float(self.total_reward),
            terminated=self.terminated,
            truncated=False,
            true_expected_rewards=self.true_expected_rewards.copy(),
            step_count=self.step_count,
        )

    def _build_public_state(self, reward: float | None) -> BanditPublicState:
        return BanditPublicState(
            arm_count=self.arm_count,
            step_count=self.step_count,
            max_steps=self.max_steps,
            last_arm=self.last_arm,
            last_reward=float(reward) if reward is not None else (self.last_reward if self.step_count else None),
            cumulative_reward=float(self.total_reward),
            reward_history=self.reward_history.copy(),
            arm_pull_counts=self.arm_pull_counts.copy(),
            terminated=self.terminated,
            status=self.status,
        )

    async def _serialize_engine(self) -> BanditEngineSnapshot:
        snapshot = {
            "bandit_type": self.bandit_type,
            "arm_probabilities": self.arm_probabilities,
            "arm_means": self.arm_means,
            "arm_stds": self.arm_stds,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "last_reward": self.last_reward,
            "last_arm": self.last_arm,
            "reward_history": self.reward_history,
            "arm_history": self.arm_history,
            "arm_pull_counts": self.arm_pull_counts,
            "terminated": self.terminated,
            "status": self.status,
            "true_expected_rewards": self.true_expected_rewards,
            "rng_state": self._rng.bit_generator.state,
        }
        return BanditEngineSnapshot(
            task_instance_dict=await self.task_instance.serialize(),
            engine_snapshot=snapshot,
        )

    @classmethod
    async def _deserialize_engine(cls, snapshot: BanditEngineSnapshot) -> BanditEngine:
        from .taskset import BanditTaskInstance

        task_instance = await BanditTaskInstance.deserialize(snapshot.task_instance_dict)
        engine = cls(task_instance)

        data = snapshot.engine_snapshot
        engine.bandit_type = data.get("bandit_type", engine.bandit_type)
        engine.max_steps = int(data.get("max_steps", engine.max_steps))
        engine.seed = data.get("seed", engine.seed)
        engine.arm_probabilities = data.get("arm_probabilities", engine.arm_probabilities)
        engine.arm_means = data.get("arm_means", engine.arm_means)
        engine.arm_stds = data.get("arm_stds", engine.arm_stds)
        engine.true_expected_rewards = list(data.get("true_expected_rewards", engine.true_expected_rewards))
        engine.arm_count = len(engine.true_expected_rewards)

        engine.step_count = int(data.get("step_count", 0))
        engine.total_reward = float(data.get("total_reward", 0.0))
        engine.last_reward = float(data.get("last_reward", 0.0))
        engine.last_arm = data.get("last_arm")
        engine.reward_history = list(data.get("reward_history", []))
        engine.arm_history = list(data.get("arm_history", []))
        engine.arm_pull_counts = list(data.get("arm_pull_counts", [0 for _ in range(engine.arm_count)]))
        engine.terminated = bool(data.get("terminated", False))
        engine.status = data.get("status", "in_progress")

        engine._rng = np.random.default_rng()
        rng_state = data.get("rng_state")
        if rng_state is not None:
            engine._rng.bit_generator.state = rng_state

        return engine


class SynthBanditObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: BanditPublicState, priv: BanditPrivateState
    ) -> InternalObservation:
        observation: InternalObservation = {
            "arm_count": pub.arm_count,
            "steps_taken": pub.step_count,
            "steps_remaining": pub.steps_remaining,
            "last_arm": pub.last_arm,
            "last_reward": pub.last_reward,
            "cumulative_reward": priv.total_reward,
            "average_reward": pub.average_reward,
            "arm_pull_counts": pub.arm_pull_counts,
            "reward_history": pub.reward_history,
            "terminated": pub.terminated,
            "status": pub.status,
        }
        return observation


class SynthBanditCheckpointObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: BanditPublicState, priv: BanditPrivateState
    ) -> InternalObservation:
        observation: InternalObservation = {
            "arm_count": pub.arm_count,
            "total_reward": priv.total_reward,
            "steps_taken": pub.step_count,
            "best_expected_reward": max(priv.true_expected_rewards) if priv.true_expected_rewards else None,
            "terminated": pub.terminated,
            "status": pub.status,
        }
        return observation
