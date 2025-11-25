"""Unit tests for LMCAISEvent with LLMCallRecord integration.

This module tests the new call_records field in LMCAISEvent and demonstrates
proper usage patterns for migrating from legacy fields to the new structure.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TypedDict, cast

import pytest

from ..abstractions import (
    LMCAISEvent,
    SessionTimeStep,
    SessionTrace,
    TimeRecord,
)
from ..lm_call_record_abstractions import (
    LLMCallRecord,
    LLMContentPart,
    LLMMessage,
    LLMUsage,
    ToolCallResult,
    ToolCallSpec,
    compute_latency_ms,
)


class OpenAIChoiceMessage(TypedDict):
    """Typed representation of an OpenAI chat completion message."""

    role: str
    content: str


class OpenAIChoice(TypedDict):
    """Typed representation of an OpenAI chat completion choice."""

    index: int
    message: OpenAIChoiceMessage
    finish_reason: str


class OpenAIUsage(TypedDict):
    """Token usage block for OpenAI responses."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatResponse(TypedDict):
    """Minimal schema for OpenAI chat completion responses used in testing."""

    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage


class AnthropicContentBlock(TypedDict):
    """Content block from Anthropic Messages API."""

    type: str
    text: str


class AnthropicUsage(TypedDict):
    """Token usage block for Anthropic Messages API."""

    input_tokens: int
    output_tokens: int


class AnthropicResponse(TypedDict):
    """Minimal schema for Anthropic messages responses used in testing."""

    id: str
    type: str
    role: str
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: str
    usage: AnthropicUsage


class TestLLMCallRecord:
    """Test LLMCallRecord creation and manipulation."""

    def test_create_basic_call_record(self):
        """Test creating a basic LLMCallRecord."""
        call_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)

        record = LLMCallRecord(
            call_id=call_id,
            api_type="chat_completions",
            provider="openai",
            model_name="gpt-4",
            started_at=started_at,
            input_messages=[
                LLMMessage(role="user", parts=[LLMContentPart(type="text", text="What is 2+2?")])
            ],
            output_messages=[
                LLMMessage(role="assistant", parts=[LLMContentPart(type="text", text="4")])
            ],
            usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15, cost_usd=0.0003),
            finish_reason="stop",
        )

        assert record.call_id == call_id
        assert record.api_type == "chat_completions"
        assert record.model_name == "gpt-4"
        assert record.usage is not None
        assert record.usage.total_tokens == 15
        assert len(record.input_messages) == 1
        assert len(record.output_messages) == 1

    def test_compute_latency(self):
        """Test latency computation from timestamps."""
        started_at = datetime.now(timezone.utc)
        completed_at = datetime.now(timezone.utc)

        record = LLMCallRecord(
            call_id=str(uuid.uuid4()),
            api_type="chat_completions",
            started_at=started_at,
            completed_at=completed_at,
        )

        latency = compute_latency_ms(record)
        assert latency is not None
        assert latency >= 0
        assert record.latency_ms == latency

    def test_tool_call_record(self):
        """Test LLMCallRecord with tool calls."""
        record = LLMCallRecord(
            call_id=str(uuid.uuid4()),
            api_type="chat_completions",
            provider="openai",
            model_name="gpt-4",
            output_tool_calls=[
                ToolCallSpec(
                    name="get_weather",
                    arguments_json='{"location": "San Francisco"}',
                    arguments={"location": "San Francisco"},
                    call_id="tool_1",
                )
            ],
            tool_results=[
                ToolCallResult(
                    call_id="tool_1", output_text="72°F, sunny", status="ok", duration_ms=150
                )
            ],
        )

        assert len(record.output_tool_calls) == 1
        assert record.output_tool_calls[0].name == "get_weather"
        assert len(record.tool_results) == 1
        assert record.tool_results[0].status == "ok"


class TestLMCAISEventWithCallRecords:
    """Test LMCAISEvent with integrated LLMCallRecord."""

    def test_create_event_with_call_records(self):
        """Test creating an LMCAISEvent with call_records."""
        call_record = LLMCallRecord(
            call_id=str(uuid.uuid4()),
            api_type="chat_completions",
            provider="openai",
            model_name="gpt-4",
            usage=LLMUsage(input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.003),
            latency_ms=500,
        )

        event = LMCAISEvent(
            system_instance_id="llm_system",
            time_record=TimeRecord(event_time=time.time()),
            call_records=[call_record],
        )

        assert len(event.call_records) == 1
        event_call_record = event.call_records[0]
        assert event_call_record.usage is not None
        assert event_call_record.model_name == "gpt-4"
        assert event_call_record.usage.total_tokens == 150

    def test_aggregate_from_call_records(self):
        """Test computing aggregates from multiple call_records."""
        call_records = [
            LLMCallRecord(
                call_id=str(uuid.uuid4()),
                api_type="chat_completions",
                model_name="gpt-4",
                usage=LLMUsage(
                    input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.003
                ),
                latency_ms=500,
            ),
            LLMCallRecord(
                call_id=str(uuid.uuid4()),
                api_type="chat_completions",
                model_name="gpt-4",
                usage=LLMUsage(
                    input_tokens=200, output_tokens=100, total_tokens=300, cost_usd=0.006
                ),
                latency_ms=700,
            ),
        ]

        # Create event with call_records
        event = LMCAISEvent(
            system_instance_id="llm_system",
            time_record=TimeRecord(event_time=time.time()),
            call_records=call_records,
        )

        # Compute aggregates from call_records
        total_input_tokens = sum(
            r.usage.input_tokens for r in call_records if r.usage and r.usage.input_tokens
        )
        total_output_tokens = sum(
            r.usage.output_tokens for r in call_records if r.usage and r.usage.output_tokens
        )
        total_tokens = sum(
            r.usage.total_tokens for r in call_records if r.usage and r.usage.total_tokens
        )
        total_cost = sum(r.usage.cost_usd for r in call_records if r.usage and r.usage.cost_usd)
        total_latency = sum(r.latency_ms for r in call_records if r.latency_ms)

        # Set aggregates on event
        event.input_tokens = total_input_tokens
        event.output_tokens = total_output_tokens
        event.total_tokens = total_tokens
        event.cost_usd = total_cost
        event.latency_ms = total_latency

        assert event.input_tokens == 300
        assert event.output_tokens == 150
        assert event.total_tokens == 450
        assert abs(event.cost_usd - 0.009) < 0.0001  # Use floating point comparison
        assert event.latency_ms == 1200

    def test_migration_pattern(self):
        """Test migration from legacy fields to call_records."""
        # Legacy pattern (what we're migrating from)
        legacy_event = LMCAISEvent(
            system_instance_id="llm_system",
            time_record=TimeRecord(event_time=time.time()),
            model_name="gpt-4",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.003,
            latency_ms=500,
        )

        # New pattern (what we're migrating to)
        new_event = LMCAISEvent(
            system_instance_id="llm_system",
            time_record=TimeRecord(event_time=time.time()),
            # Aggregates can stay on the event
            total_tokens=150,
            cost_usd=0.003,
            latency_ms=500,
            # Details go in call_records
            call_records=[
                LLMCallRecord(
                    call_id=str(uuid.uuid4()),
                    api_type="chat_completions",
                    provider="openai",
                    model_name="gpt-4",
                    usage=LLMUsage(
                        input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.003
                    ),
                    latency_ms=500,
                )
            ],
        )

        # Both should represent the same information
        assert legacy_event.total_tokens == new_event.total_tokens
        assert legacy_event.cost_usd == new_event.cost_usd
        assert legacy_event.model_name == new_event.call_records[0].model_name
        assert legacy_event.provider == new_event.call_records[0].provider


class TestComplexScenarios:
    """Test complex scenarios with multiple calls and tool usage."""

    def test_multi_turn_conversation(self):
        """Test a multi-turn conversation with multiple LLM calls."""
        session = SessionTrace(session_id=str(uuid.uuid4()), created_at=datetime.now(timezone.utc))

        # Turn 1: Initial question
        turn1 = SessionTimeStep(step_id="turn_1", step_index=0, turn_number=1)

        turn1_event = LMCAISEvent(
            system_instance_id="llm_system",
            time_record=TimeRecord(event_time=time.time()),
            call_records=[
                LLMCallRecord(
                    call_id=str(uuid.uuid4()),
                    api_type="chat_completions",
                    model_name="gpt-4",
                    input_messages=[
                        LLMMessage(
                            role="user",
                            parts=[LLMContentPart(type="text", text="What's the weather?")],
                        )
                    ],
                    output_messages=[
                        LLMMessage(
                            role="assistant",
                            parts=[
                                LLMContentPart(type="text", text="I'll check the weather for you.")
                            ],
                        )
                    ],
                    output_tool_calls=[
                        ToolCallSpec(
                            name="get_weather",
                            arguments_json='{"location": "current"}',
                            call_id="weather_1",
                        )
                    ],
                    usage=LLMUsage(input_tokens=10, output_tokens=20, total_tokens=30),
                )
            ],
        )

        turn1.events.append(turn1_event)
        session.session_time_steps.append(turn1)

        # Turn 2: Tool result and response
        turn2 = SessionTimeStep(step_id="turn_2", step_index=1, turn_number=2)

        turn2_event = LMCAISEvent(
            system_instance_id="llm_system",
            time_record=TimeRecord(event_time=time.time()),
            call_records=[
                LLMCallRecord(
                    call_id=str(uuid.uuid4()),
                    api_type="chat_completions",
                    model_name="gpt-4",
                    input_messages=[
                        LLMMessage(
                            role="tool",
                            tool_call_id="weather_1",
                            parts=[LLMContentPart(type="text", text="San Francisco: 72°F, sunny")],
                        )
                    ],
                    output_messages=[
                        LLMMessage(
                            role="assistant",
                            parts=[
                                LLMContentPart(
                                    type="text",
                                    text="The weather in San Francisco is 72°F and sunny.",
                                )
                            ],
                        )
                    ],
                    usage=LLMUsage(input_tokens=15, output_tokens=25, total_tokens=40),
                    tool_results=[
                        ToolCallResult(
                            call_id="weather_1",
                            output_text="San Francisco: 72°F, sunny",
                            status="ok",
                        )
                    ],
                )
            ],
        )

        turn2.events.append(turn2_event)
        session.session_time_steps.append(turn2)

        # Verify session structure
        assert len(session.session_time_steps) == 2
        assert len(session.session_time_steps[0].events) == 1
        assert len(session.session_time_steps[1].events) == 1

        # Verify tool call flow
        first_event = session.session_time_steps[0].events[0]
        second_event = session.session_time_steps[1].events[0]
        assert isinstance(first_event, LMCAISEvent)
        assert isinstance(second_event, LMCAISEvent)
        turn1_call = first_event.call_records[0]
        turn2_call = second_event.call_records[0]

        assert len(turn1_call.output_tool_calls) == 1
        assert turn1_call.output_tool_calls[0].name == "get_weather"
        assert len(turn2_call.tool_results) == 1
        assert turn2_call.tool_results[0].call_id == "weather_1"

    def test_streaming_response(self):
        """Test LLMCallRecord with streaming chunks."""
        from ..lm_call_record_abstractions import LLMChunk

        chunks = [
            LLMChunk(
                sequence_index=0,
                received_at=datetime.now(timezone.utc),
                event_type="content.delta",
                delta_text="The",
                choice_index=0,
            ),
            LLMChunk(
                sequence_index=1,
                received_at=datetime.now(timezone.utc),
                event_type="content.delta",
                delta_text=" answer",
                choice_index=0,
            ),
            LLMChunk(
                sequence_index=2,
                received_at=datetime.now(timezone.utc),
                event_type="content.delta",
                delta_text=" is",
                choice_index=0,
            ),
            LLMChunk(
                sequence_index=3,
                received_at=datetime.now(timezone.utc),
                event_type="content.delta",
                delta_text=" 42",
                choice_index=0,
            ),
            LLMChunk(
                sequence_index=4,
                received_at=datetime.now(timezone.utc),
                event_type="message.stop",
                choice_index=0,
            ),
        ]

        record = LLMCallRecord(
            call_id=str(uuid.uuid4()),
            api_type="responses",  # OpenAI Responses API style
            model_name="gpt-4",
            chunks=chunks,
            output_text="The answer is 42",  # Final collapsed output
            usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

        assert record.chunks is not None
        assert len(record.chunks) == 5
        assert record.output_text == "The answer is 42"

        # Reconstruct from chunks
        reconstructed = "".join(c.delta_text for c in chunks if c.delta_text)
        assert reconstructed == "The answer is 42"


class TestProviderMappings:
    """Test mapping different provider formats to LLMCallRecord."""

    def test_openai_chat_completions_mapping(self):
        """Test mapping OpenAI Chat Completions to LLMCallRecord."""
        # Simulate OpenAI response structure
        openai_response: OpenAIChatResponse = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 9, "total_tokens": 19},
        }

        # Map to LLMCallRecord
        record = LLMCallRecord(
            call_id=openai_response["id"],
            api_type="chat_completions",
            provider="openai",
            model_name=openai_response["model"],
            output_messages=[
                LLMMessage(
                    role=openai_response["choices"][0]["message"]["role"],
                    parts=[
                        LLMContentPart(
                            type="text", text=openai_response["choices"][0]["message"]["content"]
                        )
                    ],
                )
            ],
            usage=LLMUsage(
                input_tokens=openai_response["usage"]["prompt_tokens"],
                output_tokens=openai_response["usage"]["completion_tokens"],
                total_tokens=openai_response["usage"]["total_tokens"],
            ),
            finish_reason=openai_response["choices"][0]["finish_reason"],
            provider_request_id=openai_response["id"],
        )

        assert record.call_id == "chatcmpl-123"
        assert record.model_name == "gpt-4"
        assert record.usage is not None
        assert record.usage.total_tokens == 19
        assert record.finish_reason == "stop"

    def test_anthropic_messages_mapping(self):
        """Test mapping Anthropic Messages API to LLMCallRecord."""
        # Simulate Anthropic response structure
        anthropic_response: AnthropicResponse = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "I'll help you with that."}],
            "model": "claude-3-opus-20240229",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 15, "output_tokens": 12},
        }

        # Map to LLMCallRecord
        record = LLMCallRecord(
            call_id=anthropic_response["id"],
            api_type="messages",  # Anthropic Messages API
            provider="anthropic",
            model_name=anthropic_response["model"],
            output_messages=[
                LLMMessage(
                    role=anthropic_response["role"],
                    parts=[
                        LLMContentPart(type=content["type"], text=content["text"])
                        for content in anthropic_response["content"]
                    ],
                )
            ],
            usage=LLMUsage(
                input_tokens=anthropic_response["usage"]["input_tokens"],
                output_tokens=anthropic_response["usage"]["output_tokens"],
                total_tokens=(
                    anthropic_response["usage"]["input_tokens"]
                    + anthropic_response["usage"]["output_tokens"]
                ),
            ),
            finish_reason=anthropic_response["stop_reason"],
            provider_request_id=anthropic_response["id"],
        )

        assert record.call_id == "msg_123"
        assert record.model_name == "claude-3-opus-20240229"
        assert record.usage is not None
        assert record.usage.total_tokens == 27
        assert record.finish_reason == "end_turn"


@dataclass
class _AggregateAccumulator:
    """Mutable accumulator for aggregate statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    models_used: set[str] = field(default_factory=set)
    providers_used: set[str] = field(default_factory=set)
    tool_calls_count: int = 0
    error_count: int = 0


class AggregatesSummary(TypedDict):
    """Return type for aggregate helper."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: int
    models_used: list[str]
    providers_used: list[str]
    tool_calls_count: int
    error_count: int


def helper_compute_aggregates_from_records(call_records: list[LLMCallRecord]) -> AggregatesSummary:
    """Helper function to compute aggregates from call_records.

    This demonstrates the pattern for computing event-level aggregates
    from a list of LLMCallRecord instances.
    """
    aggregates = _AggregateAccumulator()

    for record in call_records:
        if record.usage:
            if record.usage.input_tokens:
                aggregates.input_tokens += record.usage.input_tokens
            if record.usage.output_tokens:
                aggregates.output_tokens += record.usage.output_tokens
            if record.usage.total_tokens:
                aggregates.total_tokens += record.usage.total_tokens
            if record.usage.cost_usd:
                aggregates.cost_usd += record.usage.cost_usd

        if record.latency_ms is not None:
            aggregates.latency_ms += record.latency_ms

        if record.model_name:
            aggregates.models_used.add(record.model_name)

        if record.provider:
            aggregates.providers_used.add(record.provider)

        aggregates.tool_calls_count += len(record.output_tool_calls)

        if record.outcome == "error":
            aggregates.error_count += 1

    return cast(
        AggregatesSummary,
        {
            "input_tokens": aggregates.input_tokens,
            "output_tokens": aggregates.output_tokens,
            "total_tokens": aggregates.total_tokens,
            "cost_usd": aggregates.cost_usd,
            "latency_ms": aggregates.latency_ms,
            "models_used": list(aggregates.models_used),
            "providers_used": list(aggregates.providers_used),
            "tool_calls_count": aggregates.tool_calls_count,
            "error_count": aggregates.error_count,
        },
    )


class TestAggregateHelper:
    """Test the aggregate computation helper."""

    def test_compute_aggregates(self):
        """Test computing aggregates from multiple call records."""
        records = [
            LLMCallRecord(
                call_id="1",
                api_type="chat_completions",
                model_name="gpt-4",
                provider="openai",
                usage=LLMUsage(
                    input_tokens=100, output_tokens=50, total_tokens=150, cost_usd=0.003
                ),
                latency_ms=500,
                output_tool_calls=[ToolCallSpec(name="tool1", arguments_json="{}")],
            ),
            LLMCallRecord(
                call_id="2",
                api_type="messages",
                model_name="claude-3-opus",
                provider="anthropic",
                usage=LLMUsage(
                    input_tokens=200, output_tokens=100, total_tokens=300, cost_usd=0.006
                ),
                latency_ms=700,
                outcome="success",
            ),
            LLMCallRecord(
                call_id="3",
                api_type="chat_completions",
                model_name="gpt-4",
                provider="openai",
                outcome="error",
                error={"code": "rate_limit", "message": "Rate limit exceeded"},
            ),
        ]

        aggregates = helper_compute_aggregates_from_records(records)

        assert aggregates["input_tokens"] == 300
        assert aggregates["output_tokens"] == 150
        assert aggregates["total_tokens"] == 450
        assert abs(aggregates["cost_usd"] - 0.009) < 0.0001  # Floating point comparison
        assert aggregates["latency_ms"] == 1200
        assert set(aggregates["models_used"]) == {"gpt-4", "claude-3-opus"}
        assert set(aggregates["providers_used"]) == {"openai", "anthropic"}
        assert aggregates["tool_calls_count"] == 1
        assert aggregates["error_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
