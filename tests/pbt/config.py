"""Shared configuration dataclass for SDK Hypothesis stateful PBT suites (SYN-6050)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["SdkPbtConfig"]


@dataclass(frozen=True, slots=True)
class SdkPbtConfig:
    base_url: str | None
    api_key: str | None
    trace_out: Path | None
    max_examples: int
    stateful_step_count: int
    spec_dir: Path | None
