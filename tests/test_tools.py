import json
import os
from typing import List

import openai
import pytest

# Uses OpenAI clients; treat as integration when selected
pytestmark = pytest.mark.integration
from pydantic import BaseModel, Field

from synth_ai.zyk import LM
from synth_ai.lm.tools.base import BaseTool
from synth_ai.lm.vendors.core.anthropic_api import AnthropicAPI
from synth_ai.lm.vendors.openai_standard import OpenAIStandard


class WeatherParams(BaseModel):
    location: str
    unit: str


class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get the weather for a location"
    arguments: type[BaseModel] = WeatherParams

    def run(self, location: str, unit: str) -> str:
        return f"The weather in {location} is sunny and 20 {unit}"


weather_tool = WeatherTool()


class TestParams(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age in years")
    hobbies: List[str] = Field(description="List of the person's hobbies", default_factory=list)


class TestTool(BaseTool):
    name: str = "test_tool"
    description: str = "Store information about a person including their name, age, and hobbies. Always include hobbies as a list, even if empty."
    arguments: type[BaseModel] = TestParams

    def run(self, name: str, age: int, hobbies: List[str]) -> str:
        return f"Stored information for {name}, age {age}, with hobbies: {', '.join(hobbies) if hobbies else 'none'}"


# OpenAI Tests
@pytest.mark.slow
def test_weather_tool_oai():
    client = OpenAIStandard(
        sync_client=openai.OpenAI(),
        async_client=openai.AsyncOpenAI(),
        used_for_structured_outputs=False,
        exceptions_to_retry=[],
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that uses tools when appropriate.",
        },
        {
            "role": "user",
            "content": "What's the weather in Paris? Use the tools and explain your reasoning. Units local to the country, please!",
        },
    ]

    response = client._hit_api_sync(
        model="gpt-4o-mini",
        messages=messages,
        tools=[weather_tool],
        lm_config={"temperature": 0},
    )

    assert response.tool_calls is not None


@pytest.mark.slow
def test_weather_tool_oai_lm():
    lm = LM(
        model_name="gpt-4o-mini",
        formatting_model_name="gpt-4o-mini",
        temperature=0,
    )

    response = lm.respond_sync(
        system_message="You are a helpful assistant that uses tools when appropriate.",
        user_message="What's the weather in Paris? Use the tools and explain your reasoning. Units local to the country, please!",
        tools=[weather_tool],
    )

    assert response.tool_calls is not None


# Anthropic Tests
@pytest.mark.slow
def test_weather_tool_anthropic_direct():
    client = AnthropicAPI(
        used_for_structured_outputs=False,
        exceptions_to_retry=[],
    )

    response = client._hit_api_sync(
        model="claude-3-haiku-20240307",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that uses tools when appropriate.",
            },
            {
                "role": "user",
                "content": "What's the weather in Paris? Use the tools and explain your reasoning. Units local to the country, please!",
            },
        ],
        tools=[weather_tool],
        lm_config={"temperature": 0},
    )

    assert response.tool_calls is not None


@pytest.mark.slow
def test_weather_tool_anthropic_lm():
    lm = LM(
        model_name="claude-3-haiku-20240307",
        formatting_model_name="claude-3-haiku-20240307",
        temperature=0,
    )

    response = lm.respond_sync(
        system_message="You are a helpful assistant that uses tools when appropriate.",
        user_message="What's the weather in Paris? Use the tools and explain your reasoning. Units local to the country, please!",
        tools=[weather_tool],
    )

    assert response.tool_calls is not None


# Test tool validation
def test_tool_validation():
    tool = WeatherTool()
    result = tool.run(location="Paris", unit="celsius")
    assert "Paris" in result
    assert "celsius" in result

    # Test the schema generation
    schema = tool.to_openai_tool()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "get_weather"
    assert schema["function"]["description"] == "Get the weather for a location"

    # Test tool call extraction
    function_name = "get_weather"
    arguments = json.dumps({"location": "Paris", "unit": "celsius"})
    assert isinstance(arguments, str)
    assert "location" in arguments
    assert "Paris" in arguments


if __name__ == "__main__":
    test_weather_tool_oai_lm()
