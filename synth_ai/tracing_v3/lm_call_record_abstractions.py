"""Unified abstractions for recording LLM API calls (inputs and results).

These records normalize different provider API shapes (Chat Completions,
Completions, Responses) into a single schema suitable for storage and
analysis, and are intended to be attached to LMCAISEvent(s) as a list of
call records.

Integration proposal:
- Update LMCAISEvent to store `call_records: list[LLMCallRecord]` and remove
  per-call fields like `model_name`, `provider`, and token counts from the
  event itself. Those belong on each LLMCallRecord. Aggregates (e.g.,
  total_tokens across records, cost_usd) can remain on LMCAISEvent and be
  derived from the records.

Design goals:
- Capture both input and output payloads in a provider-agnostic way.
- Preserve provider-specific request params for auditability.
- Represent tool calls (requested by the model) and tool results distinctly.
- Support streaming (optionally via `chunks`), but emphasize a final collapsed
  `LLMCallRecord` for most analytics and fine-tuning data extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class LLMUsage:
    """Token usage reported by the provider.

    All fields are optional because some providers or stages may omit them.
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    # Reasoning/chain-of-thought style token accounting (if provider exposes it)
    reasoning_tokens: int | None = None
    reasoning_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    # Caching/billing/cost
    cache_write_tokens: int | None = None
    cache_read_tokens: int | None = None
    billable_input_tokens: int | None = None
    billable_output_tokens: int | None = None
    cost_usd: float | None = None


@dataclass
class LLMRequestParams:
    """Provider request parameters.

    Store provider-agnostic params explicitly and keep a `raw_params` map for
    anything provider-specific (top_k, frequency_penalty, etc.).
    """

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    # Common non-agnostic knobs
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    n: int | None = None
    best_of: int | None = None
    response_format: dict[str, Any] | None = None
    json_mode: bool | None = None
    tool_config: dict[str, Any] | None = None
    raw_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMContentPart:
    """A content item within a message (text, tool-structured JSON, image, etc.)."""

    type: str
    text: str | None = None
    # For Responses API or multimodal payloads, keep a generic value
    data: dict[str, Any] | None = None
    # Blob reference fields (for image/audio/video)
    mime_type: str | None = None
    uri: str | None = None
    base64_data: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    duration_ms: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    language: str | None = None


@dataclass
class LLMMessage:
    """A message in a chat-style exchange.

    For Completions-style calls, `role="user"` with one text part is typical for input,
    and `role="assistant"` for output. Responses API can emit multiple parts;
    use `parts` for generality.
    """

    role: str  # e.g., system, user, assistant, tool, function, developer
    parts: list[LLMContentPart] = field(default_factory=list)
    name: str | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallSpec:
    """A tool/function call requested by the model (not yet executed)."""

    name: str
    arguments_json: str  # serialized JSON payload provided by the model
    arguments: dict[str, Any] | None = None  # parsed convenience
    call_id: str | None = None  # provider-assigned or synthesized
    index: int | None = None  # ordinal within a batch
    parent_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallResult:
    """The result of executing a tool/function call outside the model.

    This is distinct from the model's own output. Attach execution details for
    auditability.
    """

    call_id: str | None = None  # correlate to ToolCallSpec
    output_text: str | None = None
    exit_code: int | None = None
    status: Literal["ok", "error"] | None = None
    error_message: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMChunk:
    """Optional streaming chunk representation (for Responses/Chat streaming)."""

    sequence_index: int
    received_at: datetime
    event_type: str | None = None  # e.g., content.delta, tool.delta, message.stop
    choice_index: int | None = None
    raw_json: str | None = None
    delta_text: str | None = None
    delta: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMCallRecord:
    """Normalized record of a single LLM API call.

    Fields capture both the request (input) and the response (output), with
    optional tool calls and results as emitted by/through the agent runtime.
    """

    # Identity and classification
    call_id: str
    api_type: str  # e.g., "chat_completions", "completions", "responses"
    provider: str | None = None  # e.g., "openai", "anthropic"
    model_name: str = ""
    schema_version: str = "1.0"

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None
    latency_ms: int | None = None  # convenience cache (completed - started)

    # Request
    request_params: LLMRequestParams = field(default_factory=LLMRequestParams)
    input_messages: list[LLMMessage] = field(default_factory=list)
    input_text: str | None = None  # for completions-style prompts
    tool_choice: str | None = None  # e.g., "auto", "none", or a specific tool

    # Response
    output_messages: list[LLMMessage] = field(default_factory=list)
    outputs: list[LLMMessage] = field(default_factory=list)  # for n>1 choices
    output_text: str | None = None  # for completions-style outputs
    output_tool_calls: list[ToolCallSpec] = field(default_factory=list)
    usage: LLMUsage | None = None
    finish_reason: str | None = None
    choice_index: int | None = None

    # Tool execution results (post-model, optional)
    tool_results: list[ToolCallResult] = field(default_factory=list)

    # Streaming (optional)
    chunks: list[LLMChunk] | None = None

    # Raw payloads for audit/debugging
    request_raw_json: str | None = None
    response_raw_json: str | None = None

    # Provider- or call-specific extra data (tracing ids, etc.)
    span_id: str | None = None
    trace_id: str | None = None
    provider_request_id: str | None = None
    request_server_timing: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Error/outcome
    outcome: Literal["success", "error", "timeout", "cancelled"] | None = None
    error: dict[str, Any] | None = None  # {code, message, type, raw}
    # Logprob traces (optional)
    token_traces: list[dict[str, Any]] | None = None
    # Safety/refusal (optional)
    safety: dict[str, Any] | None = None
    refusal: dict[str, Any] | None = None
    # Privacy/redactions
    redactions: list[dict[str, Any]] | None = None


def compute_latency_ms(record: LLMCallRecord) -> int | None:
    """Compute and update latency_ms from timestamps if available."""
    if record.started_at and record.completed_at:
        delta = int((record.completed_at - record.started_at).total_seconds() * 1000)
        record.latency_ms = delta
        return delta
    return record.latency_ms


# Provider mapping guidance (summary)
# -----------------------------------
# - OpenAI Chat Completions:
#   - api_type = "chat_completions"
#   - input_messages from `messages`, output_messages from `choices[].message`
#   - usage from response.usage
#   - tool_calls map to ToolCallSpec (choices[].message.tool_calls)
#
# - OpenAI Completions:
#   - api_type = "completions"
#   - input_text from `prompt`, output_text from `choices[].text`
#   - usage from response.usage
#
# - OpenAI Responses API (streamed):
#   - api_type = "responses"
#   - input_messages from `input[]` or `messages[]` per content type
#   - output_messages from streamed `message` nodes; usage from terminal chunk
#   - chunks hold raw SSE segments if desired
#   - tool_calls from streamed `function_call`/`tool_call` nodes
#
# Tool execution results should be attached as ToolCallResult entries when the
# agent runtime executes the requested tool(s) and has ground-truth outputs.
