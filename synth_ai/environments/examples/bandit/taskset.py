from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from synth_ai.environments.tasks.core import (
    Impetus,
    Intent,
    SplitInfo,
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceSet,
)


@dataclass
class BanditTaskInstanceMetadata(TaskInstanceMetadata):
    name: str
    bandit_type: str
    arm_probabilities: list[float] | None
    arm_means: list[float] | None
    arm_stds: list[float] | None
    max_steps: int
    seed: int | None = None


@dataclass
class BanditTaskInstance(TaskInstance):
    async def serialize(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "impetus": {"instructions": self.impetus.instructions},
            "intent": {
                "rubric": self.intent.rubric,
                "gold_trajectories": self.intent.gold_trajectories,
                "gold_state_diff": self.intent.gold_state_diff,
            },
            "metadata": {
                "name": self.metadata.name,
                "bandit_type": self.metadata.bandit_type,
                "arm_probabilities": self.metadata.arm_probabilities,
                "arm_means": self.metadata.arm_means,
                "arm_stds": self.metadata.arm_stds,
                "max_steps": self.metadata.max_steps,
                "seed": self.metadata.seed,
            },
            "is_reproducible": self.is_reproducible,
            "initial_engine_snapshot": self.initial_engine_snapshot,
        }

    @classmethod
    async def deserialize(cls, data: dict[str, Any]) -> BanditTaskInstance:
        metadata = BanditTaskInstanceMetadata(
            name=data["metadata"].get("name", "bandit"),
            bandit_type=data["metadata"].get("bandit_type", "bernoulli"),
            arm_probabilities=data["metadata"].get("arm_probabilities"),
            arm_means=data["metadata"].get("arm_means"),
            arm_stds=data["metadata"].get("arm_stds"),
            max_steps=int(data["metadata"].get("max_steps", 100)),
            seed=data["metadata"].get("seed"),
        )
        return cls(
            id=UUID(data["id"]),
            impetus=Impetus(instructions=data["impetus"]["instructions"]),
            intent=Intent(
                rubric=data["intent"].get("rubric", {}),
                gold_trajectories=data["intent"].get("gold_trajectories"),
                gold_state_diff=data["intent"].get("gold_state_diff", {}),
            ),
            metadata=metadata,
            is_reproducible=bool(data.get("is_reproducible", True)),
            initial_engine_snapshot=data.get("initial_engine_snapshot"),
        )


def _expected_rewards(metadata: BanditTaskInstanceMetadata) -> list[float]:
    if metadata.bandit_type.lower() == "bernoulli" and metadata.arm_probabilities:
        return [float(p) for p in metadata.arm_probabilities]
    if metadata.bandit_type.lower() == "gaussian" and metadata.arm_means:
        return [float(m) for m in metadata.arm_means]
    return []


def _default_bandit_configs() -> list[dict[str, Any]]:
    return [
        {
            "name": "bernoulli-easy",
            "bandit_type": "bernoulli",
            "arm_probabilities": [0.1, 0.4, 0.8],
            "max_steps": 50,
            "seed": 7,
        },
        {
            "name": "bernoulli-close",
            "bandit_type": "bernoulli",
            "arm_probabilities": [0.45, 0.5, 0.55],
            "max_steps": 75,
            "seed": 21,
        },
        {
            "name": "gaussian-wide",
            "bandit_type": "gaussian",
            "arm_means": [0.0, 0.3, 1.0],
            "arm_stds": [0.2, 0.2, 0.4],
            "max_steps": 60,
            "seed": 14,
        },
    ]


def _build_impetus(metadata: BanditTaskInstanceMetadata, arm_count: int) -> Impetus:
    bandit_desc = metadata.bandit_type.capitalize()
    return Impetus(
        instructions=(
            f"You are interacting with a {bandit_desc} multi-armed bandit task. "
            f"There are {arm_count} arms, indexed from 0 to {arm_count - 1}. "
            f"You may pull up to {metadata.max_steps} arms. "
            "Your objective is to maximize the cumulative reward by learning which arm has the highest expected payoff."
        )
    )


def _build_intent(expected_rewards: Iterable[float]) -> Intent:
    rewards = list(expected_rewards)
    best_arm = max(range(len(rewards)), key=rewards.__getitem__) if rewards else 0
    return Intent(
        rubric={
            "goal": "Maximize cumulative reward over the episode",
            "measurement": "Higher total reward indicates better performance",
        },
        gold_trajectories=None,
        gold_state_diff={
            "best_arm_index": best_arm,
            "expected_rewards": rewards,
        },
    )


async def create_bandit_taskset(
    configs: list[dict[str, Any]] | None = None,
) -> TaskInstanceSet:
    configs = configs or _default_bandit_configs()

    instances: list[BanditTaskInstance] = []
    for config in configs:
        metadata = BanditTaskInstanceMetadata(
            name=config.get("name", "bandit"),
            bandit_type=config.get("bandit_type", "bernoulli"),
            arm_probabilities=config.get("arm_probabilities"),
            arm_means=config.get("arm_means"),
            arm_stds=config.get("arm_stds"),
            max_steps=int(config.get("max_steps", 100)),
            seed=config.get("seed"),
        )

        expected = _expected_rewards(metadata)
        arm_count = len(expected) if expected else (
            len(metadata.arm_probabilities or [])
            or len(metadata.arm_means or [])
            or 0
        )
        if arm_count == 0:
            arm_count = 1

        impetus = _build_impetus(metadata, arm_count)
        intent = _build_intent(expected)

        instance = BanditTaskInstance(
            id=uuid4(),
            impetus=impetus,
            intent=intent,
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )
        instances.append(instance)

    # Simple deterministic splits by index
    val_ids = {instances[i].id for i in range(0, len(instances), 3)}
    test_ids = {instances[i].id for i in range(1, len(instances), 3)}

    split_info = SplitInfo(
        val_instance_ids=val_ids,
        test_instance_ids=test_ids,
        _is_split_defined=bool(instances),
    )

    return TaskInstanceSet(
        name="Bandit Example TaskSet",
        description="Lightweight Bernoulli and Gaussian bandit tasks inspired by OpenAI Gym bandits.",
        instances=instances,
        split_info=split_info,
    )


# Alias for convenience

taskset = create_bandit_taskset
