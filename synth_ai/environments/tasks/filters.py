from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

from synth_ai.environments.tasks.core import TaskInstance, TaskInstanceMetadataFilter


@dataclass
class ValueFilter(TaskInstanceMetadataFilter):
    key: str
    values: Collection[Any]

    def __call__(self, instance: TaskInstance) -> bool:
        instance_value = getattr(instance.metadata, self.key, None)
        if instance_value is None:
            return False
        return instance_value in self.values


@dataclass
class RangeFilter(TaskInstanceMetadataFilter):
    key: str
    min_val: float | None = None
    max_val: float | None = None

    def __call__(self, instance: TaskInstance) -> bool:
        instance_value = getattr(instance.metadata, self.key, None)
        if instance_value is None:
            # If the attribute doesn't exist on the metadata, it can't be in range.
            return False

        if not isinstance(instance_value, (int, float)):
            # If the attribute is not a number, it can't be in a numerical range.
            # Or, we could raise an error, depending on desired strictness.
            return False

        if self.min_val is not None and instance_value < self.min_val:
            return False
        if self.max_val is not None and instance_value > self.max_val:
            return False
        return True
