"""DSPy drop-in compatibility surface backed by Synth AI."""
# See: specifications/tanha/master_specification.md

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LM:
    """Minimal LM container compatible with common DSPy optimizer usage."""

    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = str(model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = dict(kwargs)

    def __str__(self) -> str:
        return self.model


@dataclass
class _Settings:
    lm: LM | None = None
    max_errors: int | None = None


settings = _Settings()


def configure(*, lm: LM | None = None, max_errors: int | None = None, **_: Any) -> None:
    """Set global DSPy-compatible settings used by compatibility optimizers."""
    if lm is not None:
        settings.lm = lm
    if max_errors is not None:
        settings.max_errors = max_errors


from .gepa import GEPA  # noqa: E402
from .miprov2 import MIPROv2, MIPROv2DetailedResult  # noqa: E402

__all__ = [
    "GEPA",
    "LM",
    "MIPROv2",
    "MIPROv2DetailedResult",
    "configure",
    "settings",
]
