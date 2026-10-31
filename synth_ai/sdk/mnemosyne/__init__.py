"""Project evaluation evidence adapter; no Swarm actor grant is implied."""

from .evidence import AsyncEvaluationEvidence, EvaluationEvidence


class MnemosyneClient:
    def __init__(self, transport):
        self.evaluation_evidence = EvaluationEvidence(transport)


class AsyncMnemosyneClient:
    def __init__(self, transport):
        self.evaluation_evidence = AsyncEvaluationEvidence(transport)
