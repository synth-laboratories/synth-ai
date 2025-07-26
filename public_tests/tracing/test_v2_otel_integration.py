"""
Tests for OpenTelemetry integration in v2 tracing.

This module tests:
- OTel span creation and attributes
- Semantic conventions compliance
- BatchSpanProcessor integration
- Cost metrics and PII masking
- Dual-mode tracing (v2 + OTel)
"""

import pytest
import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from synth_ai.tracing_v2.decorators import (
    trace_span, ai_call, environment_step, runtime_operation,
    set_session_tracer, set_system_id, set_turn_number,
    setup_otel_tracer, mask_pii, calculate_cost,
    _otel_tracer
)
from synth_ai.tracing_v2.session_tracer import SessionTracer, CAISEvent
from synth_ai.tracing_v2.config import TracingConfig
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, Span
from opentelemetry.sdk.trace.export import BatchSpanProcessor, InMemorySpanExporter
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.semconv.trace import SpanAttributes


class TestOTelSpanCreation:
    """Test OpenTelemetry span creation and attributes."""
    
    @pytest.fixture
    def setup_otel_with_exporter(self):
        """Set up OTel with in-memory exporter."""
        exporter = InMemorySpanExporter()
        setup_otel_tracer()
        
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
        
        yield exporter
        
        # Clean up
        trace._TRACER_PROVIDER = None
    
    @pytest.fixture
    def mock_config(self, monkeypatch):
        """Mock tracing configuration with OTel enabled."""
        config = TracingConfig(
            enabled=True,
            otel_enabled=True,
            otel_service_name="test-service",
            otel_service_version="1.0.0",
            otel_deployment_environment="test",
            emit_events=True
        )
        monkeypatch.setattr("synth_ai.tracing_v2.decorators.get_config", lambda: config)
        return config
    
    def test_trace_span_decorator_basic(self, mock_config, setup_otel_with_exporter):
        """Test basic span creation with @trace_span."""
        
        @trace_span("test_operation", kind=SpanKind.INTERNAL)
        def test_function(x: int, y: int) -> int:
            return x + y
        
        result = test_function(5, 3)
        assert result == 8
        
        # Get spans
        setup_otel_with_exporter.shutdown()
        spans = setup_otel_with_exporter.get_finished_spans()
        
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "test_operation"
        assert span.kind == SpanKind.INTERNAL
        assert span.status.status_code == StatusCode.OK
    
    @pytest.mark.asyncio
    async def test_trace_span_async_with_attributes(self, mock_config, setup_otel_with_exporter):
        """Test async span with custom attributes."""
        
        def attrs_fn(args, kwargs, result, error):
            return {
                "input.x": args[0],
                "input.y": args[1],
                "output.result": result,
                "operation.type": "multiplication"
            }
        
        @trace_span("async_multiply", attrs_fn=attrs_fn)
        async def async_multiply(x: int, y: int) -> int:
            await asyncio.sleep(0.01)
            return x * y
        
        result = await async_multiply(4, 7)
        assert result == 28
        
        # Get spans
        setup_otel_with_exporter.shutdown()
        spans = setup_otel_with_exporter.get_finished_spans()
        
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "async_multiply"
        assert span.attributes["input.x"] == 4
        assert span.attributes["input.y"] == 7
        assert span.attributes["output.result"] == 28
        assert span.attributes["operation.type"] == "multiplication"
    
    def test_trace_span_with_error(self, mock_config, setup_otel_with_exporter):
        """Test span records errors correctly."""
        
        @trace_span("failing_operation")
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        # Get spans
        setup_otel_with_exporter.shutdown()
        spans = setup_otel_with_exporter.get_finished_spans()
        
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "failing_operation"
        assert span.status.status_code == StatusCode.ERROR
        assert "ValueError: Test error" in span.status.description
        
        # Check error event
        events = list(span.events)
        assert len(events) >= 1
        error_event = events[0]
        assert error_event.name == "exception"
        assert error_event.attributes["exception.type"] == "ValueError"
        assert error_event.attributes["exception.message"] == "Test error"
    
    def test_nested_spans(self, mock_config, setup_otel_with_exporter):
        """Test nested span relationships."""
        
        @trace_span("inner_operation")
        def inner_function(x: int) -> int:
            return x * 2
        
        @trace_span("outer_operation")
        def outer_function(x: int) -> int:
            result = inner_function(x)
            return result + 10
        
        result = outer_function(5)
        assert result == 20
        
        # Get spans
        setup_otel_with_exporter.shutdown()
        spans = setup_otel_with_exporter.get_finished_spans()
        
        assert len(spans) == 2
        
        # Find spans by name
        inner_span = next(s for s in spans if s.name == "inner_operation")
        outer_span = next(s for s in spans if s.name == "outer_operation")
        
        # Verify parent-child relationship
        assert inner_span.parent.span_id == outer_span.context.span_id


class TestOTelSemanticConventions:
    """Test compliance with OpenTelemetry semantic conventions."""
    
    @pytest.fixture
    def setup_otel_with_exporter(self):
        """Set up OTel with in-memory exporter."""
        exporter = InMemorySpanExporter()
        setup_otel_tracer()
        
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
        
        yield exporter
        
        # Clean up
        trace._TRACER_PROVIDER = None
    
    @pytest.fixture
    def mock_config(self, monkeypatch):
        """Mock tracing configuration."""
        config = TracingConfig(
            enabled=True,
            otel_enabled=True,
            otel_service_name="ai-test-service",
            emit_events=True
        )
        monkeypatch.setattr("synth_ai.tracing_v2.decorators.get_config", lambda: config)
        return config
    
    @pytest.mark.asyncio
    async def test_ai_call_semantic_conventions(self, mock_config, setup_otel_with_exporter):
        """Test @ai_call follows AI semantic conventions."""
        tracer = SessionTracer()
        
        @ai_call
        async def test_llm_call(prompt: str, model: str = "gpt-4") -> Dict[str, Any]:
            return {
                "response": "Test response",
                "model": model,
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 100,
                    "total_tokens": 150
                }
            }
        
        set_session_tracer(tracer)
        set_system_id("test_ai_system")
        set_turn_number(1)
        
        async with tracer.start_session("test_session"):
            await test_llm_call("Test prompt")
        
        # Get spans
        setup_otel_with_exporter.shutdown()
        spans = setup_otel_with_exporter.get_finished_spans()
        
        # Find AI span
        ai_spans = [s for s in spans if s.kind == SpanKind.CLIENT]
        assert len(ai_spans) >= 1
        
        span = ai_spans[0]
        
        # Verify semantic convention attributes
        assert span.attributes.get("gen_ai.system") == "test_ai_system"
        assert span.attributes.get("gen_ai.request.model") == "gpt-4"
        assert span.attributes.get("gen_ai.usage.prompt_tokens") == 50
        assert span.attributes.get("gen_ai.usage.completion_tokens") == 100
        assert span.attributes.get("gen_ai.usage.total_tokens") == 150
        
        # Check for cost metric
        assert "gen_ai.usage.cost_usd" in span.attributes
        assert span.attributes["gen_ai.usage.cost_usd"] > 0
    
    def test_environment_step_conventions(self, mock_config, setup_otel_with_exporter):
        """Test @environment_step semantic conventions."""
        
        @environment_step
        def env_action(action: str, state: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "state": {"position": [1, 2]},
                "reward": 10.0,
                "done": False
            }
        
        set_system_id("test_environment")
        
        result = env_action("move", {"position": [0, 0]})
        assert result["reward"] == 10.0
        
        # Get spans
        setup_otel_with_exporter.shutdown()
        spans = setup_otel_with_exporter.get_finished_spans()
        
        env_spans = [s for s in spans if "environment" in s.name.lower()]
        assert len(env_spans) >= 1
        
        span = env_spans[0]
        assert span.attributes.get("environment.name") == "test_environment"
        assert span.attributes.get("environment.reward") == 10.0
        assert span.attributes.get("environment.done") is False
    
    def test_runtime_operation_conventions(self, mock_config, setup_otel_with_exporter):
        """Test @runtime_operation semantic conventions."""
        
        @runtime_operation(operation_type="data_processing")
        def process_data(data: List[int]) -> Dict[str, Any]:
            return {
                "sum": sum(data),
                "count": len(data),
                "mean": sum(data) / len(data)
            }
        
        set_system_id("test_runtime")
        
        result = process_data([1, 2, 3, 4, 5])
        assert result["sum"] == 15
        
        # Get spans
        setup_otel_with_exporter.shutdown()
        spans = setup_otel_with_exporter.get_finished_spans()
        
        runtime_spans = [s for s in spans if s.attributes.get("runtime.operation_type")]
        assert len(runtime_spans) >= 1
        
        span = runtime_spans[0]
        assert span.attributes.get("runtime.operation_type") == "data_processing"
        assert span.attributes.get("runtime.success") is True


class TestOTelCostAndPIIFeatures:
    """Test cost calculation and PII masking features."""
    
    @pytest.fixture
    def mock_config(self, monkeypatch):
        """Mock tracing configuration."""
        config = TracingConfig(
            enabled=True,
            otel_enabled=True,
            emit_events=True
        )
        monkeypatch.setattr("synth_ai.tracing_v2.decorators.get_config", lambda: config)
        return config
    
    def test_cost_calculation_accuracy(self):
        """Test cost calculation for various models."""
        # GPT-4
        cost_gpt4 = calculate_cost("gpt-4", 1000, 2000)
        expected_gpt4 = (1000 * 30.0 + 2000 * 60.0) / 1_000_000
        assert cost_gpt4 == expected_gpt4
        
        # GPT-3.5
        cost_gpt35 = calculate_cost("gpt-3.5-turbo", 1000, 1000)
        expected_gpt35 = (1000 * 0.5 + 1000 * 1.5) / 1_000_000
        assert cost_gpt35 == expected_gpt35
        
        # Claude models
        cost_claude_opus = calculate_cost("claude-3-opus", 500, 1000)
        expected_opus = (500 * 15.0 + 1000 * 75.0) / 1_000_000
        assert cost_claude_opus == expected_opus
        
        cost_claude_sonnet = calculate_cost("claude-3-sonnet", 500, 1000)
        expected_sonnet = (500 * 3.0 + 1000 * 15.0) / 1_000_000
        assert cost_claude_sonnet == expected_sonnet
        
        # Unknown model
        cost_unknown = calculate_cost("unknown-model-xyz", 100, 100)
        assert cost_unknown == 0.0
    
    def test_pii_masking_comprehensive(self):
        """Test PII masking for various patterns."""
        # Email addresses
        text_email = "Contact me at john.doe@example.com or jane_smith123@company.co.uk"
        masked_email = mask_pii(text_email)
        assert "[EMAIL]" in masked_email
        assert "john.doe@example.com" not in masked_email
        assert "jane_smith123@company.co.uk" not in masked_email
        
        # Phone numbers
        text_phone = "Call me at 555-123-4567 or (555) 987-6543 or +1-555-321-9876"
        masked_phone = mask_pii(text_phone)
        assert masked_phone.count("[PHONE]") >= 3
        assert "555-123-4567" not in masked_phone
        
        # SSN patterns
        text_ssn = "My SSN is 123-45-6789"
        masked_ssn = mask_pii(text_ssn)
        assert "[SSN]" in masked_ssn
        assert "123-45-6789" not in masked_ssn
        
        # Credit card patterns
        text_cc = "Card number: 4111-1111-1111-1111"
        masked_cc = mask_pii(text_cc)
        assert "[CREDIT_CARD]" in masked_cc
        assert "4111-1111-1111-1111" not in masked_cc
        
        # Combined text
        combined = "Email john@example.com, phone 555-1234, SSN 123-45-6789"
        masked_combined = mask_pii(combined)
        assert "[EMAIL]" in masked_combined
        assert "[PHONE]" in masked_combined
        assert "[SSN]" in masked_combined
        assert "john@example.com" not in masked_combined
    
    @pytest.mark.asyncio
    async def test_pii_masking_in_spans(self, mock_config, setup_otel_with_exporter):
        """Test PII masking is applied in span attributes."""
        setup_otel_tracer()
        exporter = InMemorySpanExporter()
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
        
        @ai_call
        async def sensitive_llm_call(prompt: str) -> Dict[str, Any]:
            return {
                "response": f"Processing: {prompt}",
                "model": "gpt-4",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20}
            }
        
        tracer = SessionTracer()
        set_session_tracer(tracer)
        set_system_id("pii_test")
        
        sensitive_prompt = "Process user john.doe@example.com with SSN 123-45-6789"
        
        async with tracer.start_session("pii_session"):
            await sensitive_llm_call(sensitive_prompt)
        
        # Get spans
        exporter.shutdown()
        spans = exporter.get_finished_spans()
        
        # Check that PII was masked in span events
        for span in spans:
            for event in span.events:
                if "prompt" in event.name or "gen_ai" in event.name:
                    # Check attributes don't contain PII
                    for key, value in event.attributes.items():
                        if isinstance(value, str):
                            assert "john.doe@example.com" not in value
                            assert "123-45-6789" not in value
                            if "prompt" in key.lower():
                                assert "[EMAIL]" in value or "[SSN]" in value


class TestOTelBatchProcessing:
    """Test BatchSpanProcessor integration."""
    
    @pytest.fixture
    def setup_otel_with_batch_config(self):
        """Set up OTel with batch processor configuration."""
        exporter = InMemorySpanExporter()
        setup_otel_tracer()
        
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            # Configure batch processor with test-friendly settings
            processor = BatchSpanProcessor(
                exporter,
                max_queue_size=10,
                max_export_batch_size=5,
                schedule_delay_millis=100
            )
            provider.add_span_processor(processor)
        
        yield exporter, processor
        
        # Clean up
        trace._TRACER_PROVIDER = None
    
    @pytest.fixture
    def mock_config(self, monkeypatch):
        """Mock tracing configuration."""
        config = TracingConfig(
            enabled=True,
            otel_enabled=True,
            otel_batch_max_queue_size=10,
            otel_batch_max_export_size=5,
            otel_batch_schedule_delay_ms=100
        )
        monkeypatch.setattr("synth_ai.tracing_v2.decorators.get_config", lambda: config)
        return config
    
    @pytest.mark.asyncio
    async def test_batch_export_behavior(self, mock_config, setup_otel_with_batch_config):
        """Test that spans are batched before export."""
        exporter, processor = setup_otel_with_batch_config
        
        @trace_span("batch_test")
        async def create_span(i: int):
            await asyncio.sleep(0.001)
            return i
        
        # Create multiple spans quickly
        tasks = []
        for i in range(8):
            tasks.append(create_span(i))
        
        await asyncio.gather(*tasks)
        
        # Spans should not be exported immediately
        initial_spans = exporter.get_finished_spans()
        assert len(initial_spans) < 8  # Not all exported yet
        
        # Wait for batch export
        await asyncio.sleep(0.2)
        processor.force_flush()
        
        # Now all spans should be exported
        final_spans = exporter.get_finished_spans()
        assert len(final_spans) == 8
    
    def test_batch_processor_shutdown(self, mock_config, setup_otel_with_batch_config):
        """Test graceful shutdown exports remaining spans."""
        exporter, processor = setup_otel_with_batch_config
        
        @trace_span("shutdown_test")
        def create_span(i: int):
            return i * 2
        
        # Create spans
        for i in range(3):
            create_span(i)
        
        # Shutdown should export all pending spans
        processor.shutdown()
        
        spans = exporter.get_finished_spans()
        assert len(spans) == 3


class TestDualModeTracing:
    """Test dual-mode tracing (v2 + OTel) functionality."""
    
    @pytest.fixture
    def mock_config(self, monkeypatch):
        """Mock tracing configuration with both v2 and OTel enabled."""
        config = TracingConfig(
            enabled=True,
            otel_enabled=True,
            emit_events=True,
            otel_service_name="dual-mode-service"
        )
        monkeypatch.setattr("synth_ai.tracing_v2.decorators.get_config", lambda: config)
        return config
    
    @pytest.fixture
    def setup_dual_mode(self):
        """Set up both v2 tracer and OTel."""
        v2_tracer = SessionTracer()
        otel_exporter = InMemorySpanExporter()
        
        setup_otel_tracer()
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            processor = BatchSpanProcessor(otel_exporter)
            provider.add_span_processor(processor)
        
        yield v2_tracer, otel_exporter
        
        # Clean up
        trace._TRACER_PROVIDER = None
    
    @pytest.mark.asyncio
    async def test_dual_mode_ai_call(self, mock_config, setup_dual_mode):
        """Test that @ai_call creates both v2 events and OTel spans."""
        v2_tracer, otel_exporter = setup_dual_mode
        
        v2_events = []
        async def capture_v2_event(event):
            v2_events.append(event)
        
        v2_tracer.add_event = capture_v2_event
        
        @ai_call
        async def dual_mode_llm(prompt: str) -> Dict[str, Any]:
            return {
                "response": "Dual mode response",
                "model": "gpt-4",
                "usage": {"prompt_tokens": 25, "completion_tokens": 35}
            }
        
        set_session_tracer(v2_tracer)
        set_system_id("dual_system")
        set_turn_number(1)
        
        async with v2_tracer.start_session("dual_session"):
            await dual_mode_llm("Test prompt")
        
        # Check v2 events
        assert len(v2_events) == 1
        v2_event = v2_events[0]
        assert isinstance(v2_event, CAISEvent)
        assert v2_event.model_name == "gpt-4"
        assert v2_event.prompt_tokens == 25
        
        # Check OTel spans
        otel_exporter.shutdown()
        spans = otel_exporter.get_finished_spans()
        ai_spans = [s for s in spans if s.kind == SpanKind.CLIENT]
        
        assert len(ai_spans) >= 1
        otel_span = ai_spans[0]
        assert otel_span.attributes.get("gen_ai.request.model") == "gpt-4"
        assert otel_span.attributes.get("gen_ai.usage.prompt_tokens") == 25
    
    @pytest.mark.asyncio
    async def test_v2_only_mode(self, monkeypatch, setup_dual_mode):
        """Test v2-only mode when OTel is disabled."""
        config = TracingConfig(
            enabled=True,
            otel_enabled=False,  # OTel disabled
            emit_events=True
        )
        monkeypatch.setattr("synth_ai.tracing_v2.decorators.get_config", lambda: config)
        
        v2_tracer, otel_exporter = setup_dual_mode
        
        v2_events = []
        async def capture_v2_event(event):
            v2_events.append(event)
        
        v2_tracer.add_event = capture_v2_event
        
        @ai_call(v2_only=True)
        async def v2_only_llm(prompt: str) -> Dict[str, Any]:
            return {
                "response": "V2 only response",
                "model": "gpt-3.5-turbo",
                "usage": {"prompt_tokens": 15, "completion_tokens": 25}
            }
        
        set_session_tracer(v2_tracer)
        set_system_id("v2_only_system")
        
        async with v2_tracer.start_session("v2_only_session"):
            await v2_only_llm("Test")
        
        # Check v2 events created
        assert len(v2_events) == 1
        
        # Check no OTel spans created
        otel_exporter.shutdown()
        spans = otel_exporter.get_finished_spans()
        assert len(spans) == 0
    
    @pytest.mark.asyncio
    async def test_otel_only_mode(self, mock_config, setup_dual_mode):
        """Test OTel-only mode without v2 events."""
        v2_tracer, otel_exporter = setup_dual_mode
        
        v2_events = []
        async def capture_v2_event(event):
            v2_events.append(event)
        
        v2_tracer.add_event = capture_v2_event
        
        @trace_span("otel_only_operation", otel_only=True)
        async def otel_only_function(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 3
        
        result = await otel_only_function(7)
        assert result == 21
        
        # Check no v2 events created
        assert len(v2_events) == 0
        
        # Check OTel span created
        otel_exporter.shutdown()
        spans = otel_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "otel_only_operation"