from typing import List


class Judgement:
    def __init__(
        self, criteria: str, score: float, reasoning: str = "", evidence: List[str] = None
    ):
        self.criteria = criteria
        self.score = score
        self.reasoning = reasoning
        self.evidence = evidence or []


class BaseEval:
    async def run(self, data: any) -> List[Judgement]:
        pass
