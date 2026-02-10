"""AIME dataset helper for GEPA compatibility."""
# See: specifications/tanha/master_specification.md

from __future__ import annotations


def init_dataset():
    """Load the AIME datasets used in GEPA examples."""
    import random

    from datasets import load_dataset

    train_split = [
        {
            "input": item["problem"],
            "additional_context": {"solution": item["solution"]},
            "answer": "### " + str(item["answer"]),
        }
        for item in load_dataset("AI-MO/aimo-validation-aime")["train"]
    ]
    random.Random(0).shuffle(train_split)
    test_split = [
        {"input": item["problem"], "answer": "### " + str(item["answer"])}
        for item in load_dataset("MathArena/aime_2025")["train"]
    ]

    trainset = train_split[: len(train_split) // 2]
    valset = train_split[len(train_split) // 2 :]
    testset = test_split * 5

    return trainset, valset, testset
