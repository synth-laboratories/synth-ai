"""Typed checkpoint-branch request contracts for Research swarms.

# See: backend SMR branch/checkpoint authority (SmrBranchMode / branch request schemas).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class SmrBranchMode(StrEnum):
    EXACT = "exact"
    WITH_MESSAGE = "with_message"


@dataclass(frozen=True, slots=True)
class SmrRunBranchRequest:
    checkpoint_id: str | None = None
    checkpoint_record_id: str | None = None
    checkpoint_uri: str | None = None
    mode: SmrBranchMode = SmrBranchMode.EXACT
    message: str | None = None
    reason: str | None = None
    title: str | None = None
    source_node_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_id", _optional_text(self.checkpoint_id))
        object.__setattr__(self, "checkpoint_record_id", _optional_text(self.checkpoint_record_id))
        object.__setattr__(self, "checkpoint_uri", _optional_text(self.checkpoint_uri))
        object.__setattr__(self, "message", _optional_text(self.message))
        object.__setattr__(self, "reason", _optional_text(self.reason))
        object.__setattr__(self, "title", _optional_text(self.title))
        object.__setattr__(self, "source_node_id", _optional_text(self.source_node_id))
        reference_count = sum(
            1
            for value in (
                self.checkpoint_id,
                self.checkpoint_record_id,
                self.checkpoint_uri,
            )
            if value is not None
        )
        if reference_count != 1:
            raise ValueError(
                "exactly one of checkpoint_id, checkpoint_record_id, or checkpoint_uri is required"
            )
        if self.mode == SmrBranchMode.WITH_MESSAGE and self.message is None:
            raise ValueError("message is required when mode is with_message")
        if self.mode == SmrBranchMode.EXACT and self.message is not None:
            raise ValueError("message must be omitted when mode is exact")

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"mode": self.mode.value}
        for key in (
            "checkpoint_id",
            "checkpoint_record_id",
            "checkpoint_uri",
            "message",
            "reason",
            "title",
            "source_node_id",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


__all__ = [
    "SmrBranchMode",
    "SmrRunBranchRequest",
]
