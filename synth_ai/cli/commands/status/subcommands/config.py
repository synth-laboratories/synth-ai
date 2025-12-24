"""Status configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass





@dataclass
class StatusConfig:
    base_url: str = "https://api.usesynth.ai"
