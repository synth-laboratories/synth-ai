from .config import StreamConfig
from .handlers import (
    BufferedHandler,
    CallbackHandler,
    CLIHandler,
    EvalHandler,
    GraphEvolveHandler,
    GraphGenHandler,
    IntegrationTestHandler,
    JSONHandler,
    LossCurveHandler,
    OptimizationStreamHandler,
    PromptLearningHandler,
    RichHandler,
    StreamHandler,
)

try:
    from .rust_streamer import JobStreamer
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for streaming.") from exc
from .streamer import StreamEndpoints
from .types import StreamMessage, StreamType

__all__ = [
    "BufferedHandler",
    "CallbackHandler",
    "CLIHandler",
    "EvalHandler",
    "GraphEvolveHandler",
    "GraphGenHandler",
    "IntegrationTestHandler",
    "JSONHandler",
    "JobStreamer",
    "LossCurveHandler",
    "OptimizationStreamHandler",
    "PromptLearningHandler",
    "RichHandler",
    "StreamConfig",
    "StreamEndpoints",
    "StreamHandler",
    "StreamMessage",
    "StreamType",
]
