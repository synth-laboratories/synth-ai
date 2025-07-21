# taskset.py
from __future__ import annotations
import asyncio
from uuid import uuid4
import os

from datasets import load_dataset
from dataclasses import dataclass, asdict

from synth_ai.environments.tasks.core import (
    Task,
    TaskInstance,
    TaskInstanceSet,
    TaskInstanceMetadata,
    SplitInfo,
    Impetus,
    Intent,
)

enron_task = Task(
    global_premises="Answer factual questions by reading Enron e-mails",
    global_constraints="",
    global_objectives="Provide the correct answer; minimise queries",
    shared_env_params={},
)


# --------------------------------------------------------------------------- metadata
@dataclass
class EnronTaskInstanceMetadata(TaskInstanceMetadata):
    split: str
    email_count: int
    message_ids: list[str]


@dataclass
class EnronTaskInstance(TaskInstance):
    async def serialize(self):
        data = asdict(self)
        if isinstance(data.get("id"), uuid4().__class__):
            data["id"] = str(data["id"])
        return data

    @classmethod
    async def deserialize(cls, data: dict) -> "EnronTaskInstance":
        return cls(**data)


# --------------------------------------------------------------------------- task-set builder
# Use a local dataset cache under examples/enron/dataset
CACHE_DIR = os.path.join(os.path.dirname(__file__), "dataset")
os.makedirs(CACHE_DIR, exist_ok=True)


async def create_enron_taskset() -> TaskInstanceSet:
    ds_train = load_dataset(
        "corbt/enron_emails_sample_questions",
        split="train",
        cache_dir=CACHE_DIR,
    )
    ds_test = load_dataset(
        "corbt/enron_emails_sample_questions",
        split="test",
        cache_dir=CACHE_DIR,
    )

    def to_instance(row: dict, split: str) -> EnronTaskInstance:
        impetus = Impetus(instructions=row["question"])
        intent = Intent(
            rubric={"goal": "Answer the question using the Enron emails."},
            gold_trajectories=None,
            gold_state_diff={"answer": row["answer"]},
        )
        metadata = EnronTaskInstanceMetadata(
            split=split,
            email_count=len(row["message_ids"]),
            message_ids=row["message_ids"],
        )
        return EnronTaskInstance(
            id=uuid4(),
            impetus=impetus,
            intent=intent,
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=row,
        )

    train_instances = [to_instance(r, "train") for r in ds_train]
    test_instances = [to_instance(r, "test") for r in ds_test]

    split_info = SplitInfo(
        val_instance_ids=set(),
        test_instance_ids={inst.id for inst in test_instances},
        _is_split_defined=True,
    )

    return TaskInstanceSet(
        name="Enron-QA",
        description="QA over Enron email dataset sample.",
        instances=train_instances + test_instances,
        split_info=split_info,
    )


# quick sanity check ----------------------------------------------------------
if __name__ == "__main__":

    async def _main():
        ts = await create_enron_taskset()
        print(f"{len(ts.instances)} instances built.")

    asyncio.run(_main())
