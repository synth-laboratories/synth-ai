"""Banking77 dataset helper for GEPA compatibility."""
# See: specifications/tanha/master_specification.md

from __future__ import annotations

from typing import Any


def _extract_label_names(dataset: Any) -> list[str]:
    train_split = dataset.get("train")
    if train_split is None:
        raise ValueError("Banking77 dataset missing train split.")
    features = getattr(train_split, "features", None)
    if not isinstance(features, dict):
        raise ValueError("Banking77 dataset missing features metadata.")
    label_feature = features.get("label")
    names = getattr(label_feature, "names", None)
    if not names:
        raise ValueError("Banking77 dataset label names are unavailable.")
    return list(names)


def get_labels() -> list[str]:
    """Return Banking77 intent labels."""
    # See: specifications/tanha/master_specification.md
    from datasets import load_dataset

    dataset = load_dataset("banking77")
    return _extract_label_names(dataset)


def init_dataset(
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load Banking77 into (train, val, test) splits."""
    # See: specifications/tanha/master_specification.md
    import random

    from datasets import load_dataset

    dataset = load_dataset("banking77")
    label_names = _extract_label_names(dataset)

    def _format_item(item: dict[str, Any]) -> dict[str, Any]:
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Banking77 item missing text.")
        label_id = item.get("label")
        if not isinstance(label_id, int) or label_id < 0 or label_id >= len(label_names):
            raise ValueError("Banking77 item missing valid label id.")
        label_name = label_names[label_id]
        return {
            "input": text,
            "answer": label_name,
            "additional_context": {
                "label_id": str(label_id),
                "label": label_name,
                "labels": ", ".join(label_names),
            },
        }

    train_items = [_format_item(item) for item in dataset["train"]]
    random.Random(seed).shuffle(train_items)
    midpoint = len(train_items) // 2
    trainset = train_items[:midpoint]
    valset = train_items[midpoint:]
    testset = [_format_item(item) for item in dataset["test"]]
    return trainset, valset, testset
