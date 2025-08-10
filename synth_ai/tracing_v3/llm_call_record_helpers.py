"""Helper functions for creating and populating LLMCallRecord instances.

This module provides utilities to convert vendor responses to LLMCallRecord
format and compute aggregates from call records.
"""

import uuid
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from synth_ai.tracing_v3.lm_call_record_abstractions import (
    LLMCallRecord,
    LLMUsage,
    LLMRequestParams,
    LLMMessage,
    LLMContentPart,
    ToolCallSpec,
    ToolCallResult,
    LLMChunk,
)
from synth_ai.lm.vendors.base import BaseLMResponse


def create_llm_call_record_from_response(
    response: BaseLMResponse,
    model_name: str,
    provider: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.8,
    request_params: Optional[Dict[str, Any]] = None,
    tools: Optional[List] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    latency_ms: Optional[int] = None,
) -> LLMCallRecord:
    """Create an LLMCallRecord from a vendor response.
    
    Args:
        response: The vendor response object
        model_name: Name of the model used
        provider: Provider name (e.g., 'openai', 'anthropic')
        messages: Input messages sent to the model
        temperature: Temperature parameter used
        request_params: Additional request parameters
        tools: Tools provided to the model
        started_at: When the request started
        completed_at: When the request completed
        latency_ms: End-to-end latency in milliseconds
    
    Returns:
        A populated LLMCallRecord instance
    """
    # Generate call ID
    call_id = str(uuid.uuid4())
    
    # Determine API type from response
    api_type = "chat_completions"  # Default
    if hasattr(response, 'api_type'):
        if response.api_type == "responses":
            api_type = "responses"
        elif response.api_type == "completions":
            api_type = "completions"
    
    # Convert input messages to LLMMessage format
    input_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Handle different content formats
        if isinstance(content, str):
            parts = [LLMContentPart(type="text", text=content)]
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(LLMContentPart(type="text", text=item.get("text", "")))
                    elif item.get("type") == "image_url":
                        parts.append(LLMContentPart(
                            type="image",
                            uri=item.get("image_url", {}).get("url", ""),
                            mime_type="image/jpeg"
                        ))
                    elif item.get("type") == "image":
                        parts.append(LLMContentPart(
                            type="image",
                            data=item.get("source", {}),
                            mime_type=item.get("source", {}).get("media_type", "image/jpeg")
                        ))
                else:
                    parts.append(LLMContentPart(type="text", text=str(item)))
        else:
            parts = [LLMContentPart(type="text", text=str(content))]
        
        input_messages.append(LLMMessage(role=role, parts=parts))
    
    # Extract output messages from response
    output_messages = []
    output_text = None
    
    if hasattr(response, 'raw_response'):
        # Extract assistant message
        output_text = response.raw_response
        output_messages.append(
            LLMMessage(
                role="assistant",
                parts=[LLMContentPart(type="text", text=output_text)]
            )
        )
    
    # Extract tool calls if present
    output_tool_calls = []
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for idx, tool_call in enumerate(response.tool_calls):
            if isinstance(tool_call, dict):
                output_tool_calls.append(
                    ToolCallSpec(
                        name=tool_call.get("function", {}).get("name", ""),
                        arguments_json=tool_call.get("function", {}).get("arguments", "{}"),
                        call_id=tool_call.get("id", f"tool_{idx}"),
                        index=idx
                    )
                )
    
    # Extract usage information
    usage = None
    if hasattr(response, 'usage') and response.usage:
        usage = LLMUsage(
            input_tokens=response.usage.get("input_tokens"),
            output_tokens=response.usage.get("output_tokens"),
            total_tokens=response.usage.get("total_tokens"),
            cost_usd=response.usage.get("cost_usd"),
            # Additional token accounting if available
            reasoning_tokens=response.usage.get("reasoning_tokens"),
            reasoning_input_tokens=response.usage.get("reasoning_input_tokens"),
            reasoning_output_tokens=response.usage.get("reasoning_output_tokens"),
            cache_write_tokens=response.usage.get("cache_write_tokens"),
            cache_read_tokens=response.usage.get("cache_read_tokens"),
        )
    
    # Build request parameters
    params = LLMRequestParams(
        temperature=temperature,
        top_p=request_params.get("top_p") if request_params else None,
        max_tokens=request_params.get("max_tokens") if request_params else None,
        stop=request_params.get("stop") if request_params else None,
        raw_params=request_params or {}
    )
    
    # Handle response-specific fields
    finish_reason = None
    if hasattr(response, 'finish_reason'):
        finish_reason = response.finish_reason
    elif hasattr(response, 'stop_reason'):
        finish_reason = response.stop_reason
    
    # Create the call record
    record = LLMCallRecord(
        call_id=call_id,
        api_type=api_type,
        provider=provider,
        model_name=model_name,
        started_at=started_at or datetime.utcnow(),
        completed_at=completed_at or datetime.utcnow(),
        latency_ms=latency_ms,
        request_params=params,
        input_messages=input_messages,
        input_text=None,  # For completions API
        tool_choice="auto" if tools else None,
        output_messages=output_messages,
        output_text=output_text,
        output_tool_calls=output_tool_calls,
        usage=usage,
        finish_reason=finish_reason,
        outcome="success",
        metadata={
            "has_tools": tools is not None,
            "num_tools": len(tools) if tools else 0,
        }
    )
    
    # Store response ID if available (for Responses API)
    if hasattr(response, 'response_id') and response.response_id:
        record.metadata["response_id"] = response.response_id
        record.provider_request_id = response.response_id
    
    return record


def compute_aggregates_from_call_records(call_records: List[LLMCallRecord]) -> Dict[str, Any]:
    """Compute aggregate statistics from a list of LLMCallRecord instances.
    
    Args:
        call_records: List of LLMCallRecord instances
    
    Returns:
        Dictionary containing aggregated statistics
    """
    aggregates = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cost_usd": 0.0,
        "latency_ms": 0,
        "models_used": set(),
        "providers_used": set(),
        "tool_calls_count": 0,
        "error_count": 0,
        "success_count": 0,
        "call_count": len(call_records)
    }
    
    for record in call_records:
        # Token aggregation
        if record.usage:
            if record.usage.input_tokens:
                aggregates["input_tokens"] += record.usage.input_tokens
            if record.usage.output_tokens:
                aggregates["output_tokens"] += record.usage.output_tokens
            if record.usage.total_tokens:
                aggregates["total_tokens"] += record.usage.total_tokens
            if record.usage.reasoning_tokens:
                aggregates["reasoning_tokens"] += record.usage.reasoning_tokens
            if record.usage.cost_usd:
                aggregates["cost_usd"] += record.usage.cost_usd
        
        # Latency aggregation
        if record.latency_ms:
            aggregates["latency_ms"] += record.latency_ms
        
        # Model and provider tracking
        if record.model_name:
            aggregates["models_used"].add(record.model_name)
        if record.provider:
            aggregates["providers_used"].add(record.provider)
        
        # Tool calls
        aggregates["tool_calls_count"] += len(record.output_tool_calls)
        
        # Success/error tracking
        if record.outcome == "error":
            aggregates["error_count"] += 1
        elif record.outcome == "success":
            aggregates["success_count"] += 1
    
    # Convert sets to lists for JSON serialization
    aggregates["models_used"] = list(aggregates["models_used"])
    aggregates["providers_used"] = list(aggregates["providers_used"])
    
    # Compute averages
    if aggregates["call_count"] > 0:
        aggregates["avg_latency_ms"] = aggregates["latency_ms"] / aggregates["call_count"]
        aggregates["avg_input_tokens"] = aggregates["input_tokens"] / aggregates["call_count"]
        aggregates["avg_output_tokens"] = aggregates["output_tokens"] / aggregates["call_count"]
    
    return aggregates


def create_llm_call_record_from_streaming(
    chunks: List[LLMChunk],
    model_name: str,
    provider: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.8,
    request_params: Optional[Dict[str, Any]] = None,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
) -> LLMCallRecord:
    """Create an LLMCallRecord from streaming chunks.
    
    This function reconstructs a complete LLMCallRecord from streaming
    response chunks, useful for Responses API or streaming Chat Completions.
    
    Args:
        chunks: List of LLMChunk instances from streaming
        model_name: Name of the model used
        provider: Provider name
        messages: Input messages sent to the model
        temperature: Temperature parameter used
        request_params: Additional request parameters
        started_at: When the request started
        completed_at: When the request completed
    
    Returns:
        A populated LLMCallRecord instance
    """
    # Reconstruct output text from chunks
    output_text = "".join(
        chunk.delta_text for chunk in chunks 
        if chunk.delta_text
    )
    
    # Calculate latency from chunk timestamps
    latency_ms = None
    if chunks and started_at:
        last_chunk_time = chunks[-1].received_at
        latency_ms = int((last_chunk_time - started_at).total_seconds() * 1000)
    
    # Convert input messages
    input_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if isinstance(content, str):
            parts = [LLMContentPart(type="text", text=content)]
        else:
            parts = [LLMContentPart(type="text", text=str(content))]
        
        input_messages.append(LLMMessage(role=role, parts=parts))
    
    # Create output message
    output_messages = [
        LLMMessage(
            role="assistant",
            parts=[LLMContentPart(type="text", text=output_text)]
        )
    ]
    
    # Build request parameters
    params = LLMRequestParams(
        temperature=temperature,
        raw_params=request_params or {}
    )
    
    # Create the call record
    record = LLMCallRecord(
        call_id=str(uuid.uuid4()),
        api_type="responses",  # Streaming typically from Responses API
        provider=provider,
        model_name=model_name,
        started_at=started_at or datetime.utcnow(),
        completed_at=completed_at or datetime.utcnow(),
        latency_ms=latency_ms,
        request_params=params,
        input_messages=input_messages,
        output_messages=output_messages,
        output_text=output_text,
        chunks=chunks,
        outcome="success",
        metadata={
            "chunk_count": len(chunks),
            "streaming": True
        }
    )
    
    return record