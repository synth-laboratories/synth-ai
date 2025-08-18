"""
Utility functions and generic filters for taskset creation.
"""

from collections.abc import Collection
from typing import Any
from uuid import UUID, uuid4

from synth_ai.environments.tasks.core import (
    SplitInfo,
    TaskInstance,
    TaskInstanceMetadataFilter,
    TaskInstanceSet,
)


def parse_or_new_uuid(raw_id: str | None) -> UUID:
    """
    Parse a raw ID string into a UUID, or generate a new one if invalid or missing.
    """
    try:
        return UUID(raw_id)  # type: ignore[arg-type]
    except Exception:
        return uuid4()


class ValueFilter(TaskInstanceMetadataFilter):
    """
    Filter TaskInstances by exact match of a metadata attribute.
    """

    def __init__(self, key: str, values: Collection[Any]):
        self.key = key
        self.values = set(values)

    def __call__(self, instance: TaskInstance) -> bool:
        return getattr(instance.metadata, self.key, None) in self.values


class RangeFilter(TaskInstanceMetadataFilter):
    """
    Filter TaskInstances where a numeric metadata attribute falls within [min_value, max_value].
    """

    def __init__(
        self,
        key: str,
        min_value: float | None = None,
        max_value: float | None = None,
    ):
        self.key = key
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, instance: TaskInstance) -> bool:
        value = getattr(instance.metadata, self.key, None)
        if self.min_value is not None and (value is None or value < self.min_value):
            return False
        if self.max_value is not None and (value is None or value > self.max_value):
            return False
        return True


def make_taskset(
    name: str,
    description: str,
    instances: list[TaskInstance],
    val_filter: TaskInstanceMetadataFilter | None = None,
    test_filter: TaskInstanceMetadataFilter | None = None,
) -> TaskInstanceSet:
    """
    Assemble a TaskInstanceSet by applying optional validation and test filters.
    """
    val_ids: set[Any] = set()
    test_ids: set[Any] = set()
    if val_filter:
        val_ids = {inst.id for inst in instances if val_filter(inst)}
    if test_filter:
        test_ids = {inst.id for inst in instances if test_filter(inst)}
    is_defined = val_filter is not None or test_filter is not None
    split_info = SplitInfo(
        val_instance_ids=val_ids,
        test_instance_ids=test_ids,
        _is_split_defined=is_defined,
    )
    return TaskInstanceSet(
        name=name,
        description=description,
        instances=instances,
        split_info=split_info,
    )
