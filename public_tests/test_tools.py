import json
import os
from typing import List

import openai
import pytest
from pydantic import BaseModel, Field

from synth_ai.zyk import LM
from synth_ai.zyk.lms.tools.base import BaseTool
from synth_ai.zyk.lms.vendors.core.anthropic_api import AnthropicAPI
from synth_ai.zyk.lms.vendors.core.gemini_api import GeminiAPI
from synth_ai.zyk.lms.vendors.core.mistral_api import MistralAPI
from synth_ai.zyk.lms.vendors.openai_standard import OpenAIStandard


class WeatherParams(BaseModel):
    location: str
    unit: str


weather_tool = BaseTool(
    name="get_weather",
    description="Get current temperature for a given location.",
    arguments=WeatherParams,
    strict=True,
)


class TestToolArguments(BaseModel):
    name: str = Field(..., description="The name of the person")
    age: int = Field(..., description="The age of the person")
    hobbies: List[str] = Field(
        default_factory=list,
        description="List of the person's hobbies (use empty list if not specified)",
    )


class TestTool(BaseTool):
    name: str = "test_tool"
    arguments: type = TestToolArguments
    description: str = "Store information about a person including their name, age, and hobbies. Always include hobbies as a list, even if empty."


# OpenAI Tests
def test_weather_tool_oai_direct():
    client = OpenAIStandard(
        sync_client=openai.OpenAI(),
        async_client=openai.AsyncOpenAI(),
        used_for_structured_outputs=False,
        exceptions_to_retry=[],
    )

    response = client._hit_api_sync(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "What's the weather in Paris? Use the tools and explain your reasoning. Units local to the country, please!",
            }
        ],
        tools=[weather_tool],
        lm_config={
            "temperature": 0,
        },
    )

    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["function"]["name"] == "get_weather"
    assert "arguments" in response.tool_calls[0]["function"]
    assert isinstance(response.tool_calls[0]["function"]["arguments"], str)


def test_weather_tool_oai_lm():
    lm = LM(
        model_name="gpt-4o-mini", formatting_model_name="gpt-4o-mini", temperature=0
    )

    response = lm.respond_sync(
        system_message="",
        user_message="What's the weather in Paris? Use the tools and explain your reasoning. Units local to the country, please!",
        tools=[weather_tool],
    )

    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["function"]["name"] == "get_weather"
    assert "arguments" in response.tool_calls[0]["function"]
    assert isinstance(response.tool_calls[0]["function"]["arguments"], str)


# Anthropic Tests
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
        lm_config={
            "temperature": 0,
        },
    )

    assert response.tool_calls is not None
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["function"]["name"] == "get_weather"
    assert "arguments" in response.tool_calls[0]["function"]
    arguments = response.tool_calls[0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert "location" in arguments
    assert "Paris" in arguments


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
    assert len(response.tool_calls) > 0
    assert response.tool_calls[0]["function"]["name"] == "get_weather"
    assert "arguments" in response.tool_calls[0]["function"]
    arguments = response.tool_calls[0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert "location" in arguments
    assert "Paris" in arguments


# Gemini Tests
def test_weather_tool_gemini_direct():
    client = GeminiAPI(
        used_for_structured_outputs=False,
        exceptions_to_retry=[],
    )

    response = client._hit_api_sync(
        model="gemini-2.0-flash",
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
        lm_config={
            "temperature": 0,
            "tool_config": {"function_calling_config": {"mode": "any"}},
        },
    )

    assert response.tool_calls is not None


def test_weather_tool_gemini_lm():
    lm = LM(
        model_name="gemini-2.0-flash",
        formatting_model_name="gemini-2.0-flash",
        temperature=0,
    )

    lm.lm_config["tool_config"] = {"function_calling_config": {"mode": "ALWAYS"}}

    response = lm.respond_sync(
        system_message="You are a helpful assistant that uses tools when appropriate.",
        user_message="What's the weather in Paris? Use the tools and explain your reasoning. Units local to the country, please!",
        tools=[weather_tool],
    )

    assert response.tool_calls is not None


# Mistral Tests
@pytest.mark.asyncio
async def test_mistral_tool_async():
    if not os.getenv("MISTRAL_API_KEY"):
        pytest.skip("MISTRAL_API_KEY not set")

    client = MistralAPI()
    tool = TestTool()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that stores information about people using the test_tool function. ",
        },
        {
            "role": "user",
            "content": "Please store information for John who is 30 years old.",
        },
    ]

    response = await client._hit_api_async(
        model="mistral-small-latest",
        messages=messages,
        lm_config={"temperature": 0},
        tools=[tool],
    )

    assert response.tool_calls is not None, "No tool calls were made"
    assert len(response.tool_calls) == 1, "Expected exactly one tool call"
    tool_call = response.tool_calls[0]
    assert tool_call["type"] == "function", "Tool call type should be function"
    assert tool_call["function"]["name"] == "test_tool", "Wrong function called"

    args = json.loads(tool_call["function"]["arguments"])
    assert args["name"] == "John", "Wrong name in arguments"
    assert args["age"] == 30, "Wrong age in arguments"
    assert "hobbies" in args, "Missing hobbies field"
    assert isinstance(args["hobbies"], list), "Hobbies should be a list"


def test_mistral_tool_sync():
    if not os.getenv("MISTRAL_API_KEY"):
        pytest.skip("MISTRAL_API_KEY not set")

    client = MistralAPI()
    tool = TestTool()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that stores information about people using the test_tool function. When given information about a person, always use the test_tool function to store it. Always include hobbies as an empty list if no hobbies are mentioned.",
        },
        {
            "role": "user",
            "content": "Please store information for John who is 30 years old.",
        },
    ]

    response = client._hit_api_sync(
        model="mistral-small-latest",
        messages=messages,
        lm_config={"temperature": 0},
        tools=[tool],
    )

    assert response.tool_calls is not None, "No tool calls were made"
    assert len(response.tool_calls) == 1, "Expected exactly one tool call"
    tool_call = response.tool_calls[0]
    assert tool_call["type"] == "function", "Tool call type should be function"
    assert tool_call["function"]["name"] == "test_tool", "Wrong function called"

    args = json.loads(tool_call["function"]["arguments"])
    assert args["name"] == "John", "Wrong name in arguments"
    assert args["age"] == 30, "Wrong age in arguments"
    assert "hobbies" in args, "Missing hobbies field"
    assert isinstance(args["hobbies"], list), "Hobbies should be a list"


def test_mistral_tool_schema():
    tool = TestTool()
    schema = tool.to_mistral_tool()

    assert schema["type"] == "function", "Missing type field"
    assert "function" in schema, "Missing function wrapper"
    function = schema["function"]
    assert function["name"] == "test_tool"
    assert (
        function["description"]
        == "Store information about a person including their name, age, and hobbies. Always include hobbies as a list, even if empty."
    )
    assert "parameters" in function
    assert not function["parameters"].get("additionalProperties", True)

    params = function["parameters"]
    assert "name" in params["properties"]
    assert params["properties"]["name"]["type"] == "string"
    assert "age" in params["properties"]
    assert params["properties"]["age"]["type"] == "integer"
    assert "hobbies" in params["properties"]
    assert params["properties"]["hobbies"]["type"] == "array"
    assert params["properties"]["hobbies"]["items"]["type"] == "string"


if __name__ == "__main__":
    test_weather_tool_oai_lm()
