"""Project evaluation evidence adapter; no Swarm actor grant is implied."""

from .evidence import AsyncEvaluationEvidence, EvaluationEvidence
from .native_traces import AsyncNativeTraces, NativeTraces


class MnemosyneClient:
    def __init__(self, transport):
        self.evaluation_evidence = EvaluationEvidence(transport)
        self.native_traces = NativeTraces(transport)


class AsyncMnemosyneClient:
    def __init__(self, transport):
        self.evaluation_evidence = AsyncEvaluationEvidence(transport)
        self.native_traces = AsyncNativeTraces(transport)
