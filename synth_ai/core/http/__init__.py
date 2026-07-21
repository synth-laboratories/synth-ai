"""Shared sync and async HTTP operation substrate."""

from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpMethod, HttpRequest, OperationId, OperationMetadata
from synth_ai.core.http.retry import RetryPolicy
from synth_ai.core.http.streaming import SseEvent, iter_sse_events
from synth_ai.core.http.transport import HttpTransport

__all__ = [
    "AsyncHttpTransport",
    "HttpMethod",
    "HttpRequest",
    "HttpTransport",
    "OperationId",
    "OperationMetadata",
    "RetryPolicy",
    "SseEvent",
    "iter_sse_events",
]
