from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Judgement:
    def __init__(
        self,
        criteria: str,
        score: float,
        reasoning: str = "",
        evidence: list[str] | None = None,
    ) -> None:
        self.criteria = criteria
        self.score = score
        self.reasoning = reasoning
        self.evidence = evidence or []


class BaseEval(ABC):
    @abstractmethod
    async def run(self, data: Any) -> list[Judgement]:
        """Execute the evaluation and return a list of judgements."""
