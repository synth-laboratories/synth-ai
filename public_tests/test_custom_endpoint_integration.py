import os
import pytest
import json
from unittest.mock import patch, MagicMock
import responses
from synth_ai import LM
from synth_ai.lm.vendors.supported.custom_endpoint import CustomEndpointAPI
from synth_ai.lm.tools.base import BaseTool
from pydantic import BaseModel


@pytest.mark.slow
class TestCustomEndpointIntegration:
    """Test suite for custom OpenAI-compatible endpoints."""

    @responses.activate
    @pytest.mark.slow
    def test_basic_completion(self):
        """Test basic completion without tools."""
        responses.add(
            responses.POST,
            "https://test-org--test-model.modal.run/chat/completions",
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I'll help you gather wood efficiently.",
                        }
                    }
                ]
            },
            status=200,
        )

        lm = LM(
            model_name="test-org--test-model.modal.run",
            formatting_model_name="gpt-4o-mini",
            temperature=0.7,
        )
        response = lm.respond_sync(
            system_message="You are a Crafter agent", user_message="Help me gather wood"
        )

        assert "wood efficiently" in response.raw_response
        assert response.tool_calls is None

    @responses.activate
    @pytest.mark.slow
    def test_tool_calling(self):
        """Test tool call extraction and validation."""
        responses.add(
            responses.POST,
            "https://test-org--test-model.modal.run/chat/completions",
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": 'I\'ll move to gather wood.\n\n```json\n{"tool_call": {"name": "move", "arguments": {"direction": "north"}}}\n```',
                        }
                    }
                ]
            },
            status=200,
        )

        # Define tool with proper pydantic model
        class MoveArgs(BaseModel):
            direction: str

        move_tool = BaseTool(name="move", description="Move in a direction", arguments=MoveArgs)

        lm = LM(
            model_name="test-org--test-model.modal.run",
            formatting_model_name="gpt-4o-mini",
            temperature=0.7,
        )
        response = lm.respond_sync(
            system_message="You are a Crafter agent", user_message="Move north", tools=[move_tool]
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "move"
        assert response.tool_calls[0]["arguments"]["direction"] == "north"

    @responses.activate
    @pytest.mark.slow
    def test_generic_domain_endpoint(self):
        """Test that generic domain endpoints work."""
        responses.add(
            responses.POST,
            "https://api.example.com/chat/completions",
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "Hello from custom endpoint!"}}
                ]
            },
            status=200,
        )

        lm = LM(model_name="api.example.com", formatting_model_name="gpt-4o-mini", temperature=0.5)
        response = lm.respond_sync(
            system_message="You are a helpful assistant", user_message="Hello"
        )

        assert "Hello from custom endpoint!" in response.raw_response

    @pytest.mark.slow
    def test_url_validation(self):
        """Test URL validation and security checks."""

        # Valid URLs
        CustomEndpointAPI("valid-org--valid-model.modal.run")
        CustomEndpointAPI("api.example.com")
        CustomEndpointAPI("subdomain.example.com/v1/api")

        # Invalid URLs should raise ValueError
        with pytest.raises(ValueError):
            CustomEndpointAPI("file://etc/passwd")

        with pytest.raises(ValueError):
            CustomEndpointAPI("localhost--model.modal.run")

        with pytest.raises(ValueError):
            CustomEndpointAPI("192.168.1.1/api")

        with pytest.raises(ValueError):
            CustomEndpointAPI("a" * 300)  # Too long

    @pytest.mark.skip(reason="Test requires actual endpoint - removed to avoid hitting cloud")
    @pytest.mark.slow
    def test_temperature_override(self):
        """Test environment variable temperature override."""
        pass

    @pytest.mark.skip(reason="Test requires actual endpoint - removed to avoid hitting cloud")
    @pytest.mark.slow
    def test_auth_token(self):
        """Test authentication token handling."""
        pass


# Integration test with real Modal endpoint (requires deployed app)
@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("MODAL_TEST_URL"), reason="Modal test URL not set")
@pytest.mark.slow
def test_modal_qwen_hello():
    """Live test with deployed Modal Qwen app."""
    url = os.environ["MODAL_TEST_URL"]  # e.g. "your-org--qwen-test.modal.run"

    lm = LM(
        model_name=url,
        formatting_model_name=url,  # Use same endpoint for formatting to avoid synth_sdk dependency
        temperature=0.0,
    )

    result = lm.respond_sync(
        system_message="You are a friendly assistant.", user_message="Hello, world!"
    )

    assert isinstance(result.raw_response, str)
    assert len(result.raw_response) > 0
    print("Assistant says:", result.raw_response)
