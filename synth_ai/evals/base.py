from typing import List


class Judgement:
    criteria: str
    score: float


class BaseEval:

    async def run(self, Any) -> List[Judgement]:
        pass