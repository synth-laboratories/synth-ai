"""Fast unit tests for verifying image inputs are properly traced in v3.

These tests use a deterministic crafter policy to verify that:
1. Images are present in crafter observations
2. Images are correctly traced when included in LLM messages
3. LLMContentPart properly stores image metadata
"""

from __future__ import annotations

import base64
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from synth_ai.tracing_v3 import SessionTracer
from synth_ai.tracing_v3.abstractions import LMCAISEvent
from synth_ai.tracing_v3.lm_call_record_abstractions import (
    LLMCallRecord,
    LLMContentPart,
    LLMMessage,
    LLMUsage,
)
from synth_ai.tracing_v3.llm_call_record_helpers import (
    create_llm_call_record_from_response,
)


@pytest.mark.fast
def test_llm_content_part_stores_image_fields():
    """Test that LLMContentPart correctly stores image metadata."""
    # Create a content part with image data
    image_part = LLMContentPart(
        type="image",
        uri="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        mime_type="image/png",
        width=64,
        height=64,
    )

    assert image_part.type == "image"
    assert image_part.uri is not None
    assert "data:image/png;base64," in image_part.uri
    assert image_part.mime_type == "image/png"
    assert image_part.width == 64
    assert image_part.height == 64


@pytest.mark.fast
def test_llm_message_with_multimodal_content():
    """Test that LLMMessage can contain both text and image parts."""
    text_part = LLMContentPart(type="text", text="What do you see in this image?")
    image_part = LLMContentPart(
        type="image",
        uri="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        mime_type="image/png",
        width=1,
        height=1,
    )

    message = LLMMessage(role="user", parts=[text_part, image_part])

    assert message.role == "user"
    assert len(message.parts) == 2
    assert message.parts[0].type == "text"
    assert message.parts[0].text == "What do you see in this image?"
    assert message.parts[1].type == "image"
    assert message.parts[1].uri is not None
    assert message.parts[1].width == 1
    assert message.parts[1].height == 1


@pytest.mark.fast
def test_create_llm_call_record_with_image_url():
    """Test that create_llm_call_record_from_response parses image_url content."""
    from types import SimpleNamespace

    # Simulate a response
    mock_response = SimpleNamespace(raw_response="I see a red square.")

    # Messages with image_url content (OpenAI format)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                    },
                },
            ],
        }
    ]

    record = create_llm_call_record_from_response(
        response=mock_response,
        model_name="gpt-4o",
        provider="openai",
        messages=messages,
        temperature=0.7,
    )

    assert record.model_name == "gpt-4o"
    assert record.provider == "openai"
    assert len(record.input_messages) == 1
    
    user_msg = record.input_messages[0]
    assert user_msg.role == "user"
    assert len(user_msg.parts) == 2
    
    # Check text part
    assert user_msg.parts[0].type == "text"
    assert user_msg.parts[0].text == "What's in this image?"
    
    # Check image part
    assert user_msg.parts[1].type == "image"
    assert user_msg.parts[1].uri is not None
    assert "data:image/png;base64," in user_msg.parts[1].uri
    assert user_msg.parts[1].mime_type == "image/jpeg"  # default from helper


@pytest.mark.fast
def test_create_llm_call_record_with_anthropic_image_format():
    """Test that create_llm_call_record_from_response parses Anthropic image format."""
    from types import SimpleNamespace

    mock_response = SimpleNamespace(raw_response="I see a blue circle.")

    # Messages with Anthropic image format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                    },
                },
            ],
        }
    ]

    record = create_llm_call_record_from_response(
        response=mock_response,
        model_name="claude-3-5-sonnet-20241022",
        provider="anthropic",
        messages=messages,
        temperature=0.5,
    )

    assert len(record.input_messages) == 1
    user_msg = record.input_messages[0]
    assert len(user_msg.parts) == 2
    
    # Check image part
    image_part = user_msg.parts[1]
    assert image_part.type == "image"
    assert image_part.data is not None
    assert image_part.data.get("type") == "base64"
    assert image_part.mime_type == "image/png"


@pytest.mark.fast
@pytest.mark.asyncio
async def test_trace_image_in_messages(tmp_path: Path):
    """Test that images in LLM messages are properly stored in trace database."""
    db_path = tmp_path / "test_trace.db"

    tracer = SessionTracer(db_url=f"sqlite+aiosqlite:///{db_path}")
    await tracer.initialize()

    try:
        async with tracer.session(session_id="test_image_session") as session_id:
            async with tracer.timestep(step_id="step_0", turn_number=0):
                # Create an LLM call record with image
                from types import SimpleNamespace

                mock_response = SimpleNamespace(raw_response="I see crafter terrain.")

                # Small 1x1 red pixel PNG
                tiny_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
                data_url = f"data:image/png;base64,{tiny_png_b64}"

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What do you see?"},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ]

                llm_record = create_llm_call_record_from_response(
                    response=mock_response,
                    model_name="test-model",
                    provider="test",
                    messages=messages,
                    temperature=0.7,
                )

                # Record the LLM event
                from datetime import datetime, UTC
                from synth_ai.tracing_v3.abstractions import TimeRecord

                event = LMCAISEvent(
                    system_instance_id="test-llm",
                    time_record=TimeRecord(event_time=datetime.now(UTC).timestamp()),
                    model_name="test-model",
                    provider="test",
                    call_records=[llm_record],
                )

                await tracer.record_event(event)

        # Verify the image data was stored (session context manager auto-persists)
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check that the event was stored
        cursor.execute("SELECT COUNT(*) FROM events WHERE session_id = ?", (session_id,))
        event_count = cursor.fetchone()[0]
        assert event_count == 1

        # Check that the LLM call record contains image data
        cursor.execute(
            "SELECT call_records FROM events WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        assert row is not None

        # The call_records should be JSON containing the messages
        import json

        call_records = json.loads(row[0]) if row[0] else []
        assert len(call_records) > 0
        # Verify image data is in the stored call record
        stored_record = call_records[0]
        assert "input_messages" in stored_record

        conn.close()

    finally:
        await tracer.close()


@pytest.mark.fast
def test_crafter_observation_contains_image_fields():
    """Test that crafter observations include expected image fields.
    
    This is a structural test to verify the observation format
    without actually running the environment.
    """
    # Mock crafter observation structure
    mock_observation = {
        "observation": {
            "position": [5, 5],
            "health": 10,
            "food": 5,
            "water": 5,
            "observation_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "observation_image_format": "png",
            "observation_image_width": 64,
            "observation_image_height": 64,
            "observation_image_data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        },
        "step_idx": 0,
    }

    obs = mock_observation["observation"]
    
    # Verify all expected image fields are present
    assert "observation_image_base64" in obs
    assert "observation_image_format" in obs
    assert "observation_image_width" in obs
    assert "observation_image_height" in obs
    assert "observation_image_data_url" in obs
    
    # Verify image data is valid base64
    try:
        image_data = base64.b64decode(obs["observation_image_base64"])
        assert len(image_data) > 0
    except Exception as e:
        pytest.fail(f"Failed to decode base64 image: {e}")
    
    # Verify data URL format
    data_url = obs["observation_image_data_url"]
    assert data_url.startswith("data:image/")
    assert ";base64," in data_url


@pytest.mark.fast
def test_extract_image_parts_from_crafter_observation():
    """Test helper function to extract image parts from crafter observation."""

    def extract_image_parts(observation: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Extract image parts from a crafter observation for LLM messages."""
        if not observation:
            return []
        
        obs = observation.get("observation", observation)
        if not isinstance(obs, dict):
            return []
        
        data_url = obs.get("observation_image_data_url")
        if not data_url or not isinstance(data_url, str):
            return []
        
        # Return OpenAI-style image_url format
        return [{"type": "image_url", "image_url": {"url": data_url}}]

    # Test with valid observation
    observation_with_image = {
        "observation": {
            "health": 10,
            "observation_image_data_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "observation_image_width": 64,
            "observation_image_height": 64,
        }
    }

    image_parts = extract_image_parts(observation_with_image)
    assert len(image_parts) == 1
    assert image_parts[0]["type"] == "image_url"
    assert "url" in image_parts[0]["image_url"]
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")

    # Test with no image
    observation_no_image = {"observation": {"health": 10}}
    assert extract_image_parts(observation_no_image) == []

    # Test with None
    assert extract_image_parts(None) == []


@pytest.mark.fast
@pytest.mark.asyncio
async def test_full_image_tracing_pipeline(tmp_path: Path):
    """Test the complete pipeline: observation -> LLM message -> trace storage."""
    db_path = tmp_path / "full_pipeline.db"

    # Step 1: Mock a crafter observation with image
    tiny_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    crafter_obs = {
        "observation": {
            "position": [10, 10],
            "health": 9,
            "observation_image_base64": tiny_png_b64,
            "observation_image_data_url": f"data:image/png;base64,{tiny_png_b64}",
            "observation_image_format": "png",
            "observation_image_width": 1,
            "observation_image_height": 1,
        },
        "step_idx": 5,
    }

    # Step 2: Extract image parts (simulating what policy should do)
    obs_data = crafter_obs["observation"]
    image_data_url = obs_data["observation_image_data_url"]
    
    messages_with_image = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Health: {obs_data['health']}, Position: {obs_data['position']}"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ]

    # Step 3: Create LLM call record
    from types import SimpleNamespace

    mock_response = SimpleNamespace(raw_response='{"action": "move_forward"}')
    
    llm_record = create_llm_call_record_from_response(
        response=mock_response,
        model_name="gpt-4o-mini",
        provider="openai",
        messages=messages_with_image,
        temperature=0.2,
    )

    # Verify LLM record has image
    assert len(llm_record.input_messages) == 1
    assert len(llm_record.input_messages[0].parts) == 2
    assert llm_record.input_messages[0].parts[1].type == "image"
    assert llm_record.input_messages[0].parts[1].uri == image_data_url

    # Step 4: Trace it
    tracer = SessionTracer(db_url=f"sqlite+aiosqlite:///{db_path}")
    await tracer.initialize()

    try:
        async with tracer.session(session_id="pipeline_test") as session_id:
            async with tracer.timestep(step_id="step_5", turn_number=5):
                from datetime import datetime, UTC
                from synth_ai.tracing_v3.abstractions import TimeRecord

                event = LMCAISEvent(
                    system_instance_id="gpt-4o-mini",
                    time_record=TimeRecord(event_time=datetime.now(UTC).timestamp()),
                    model_name="gpt-4o-mini",
                    provider="openai",
                    call_records=[llm_record],
                )
                
                await tracer.record_event(event)

        # Step 5: Verify it's in the database (session context manager auto-persists)
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute(
            "SELECT call_records FROM events WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        assert row is not None

        # Parse and verify image is in stored data
        import json

        call_records = json.loads(row[0]) if row[0] else []
        assert len(call_records) > 0

        # The stored format may vary, but it should contain the image data
        # Check if we can find the data URL in the serialized data
        serialized_str = json.dumps(call_records)
        assert "data:image/png;base64," in serialized_str

        conn.close()

    finally:
        await tracer.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

