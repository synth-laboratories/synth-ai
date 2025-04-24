from pydantic import BaseModel

from synth_ai.zyk.lms.core.main import LM
from synth_ai.zyk.lms.tools.base import BaseTool
from synth_ai.zyk.lms.vendors.supported.deepseek import DeepSeekAPI


class WeatherParams(BaseModel):
    location: str


weather_tool = BaseTool(
    name="get_weather",
    description="Get current temperature for a given location.",
    arguments=WeatherParams,
)


def test_weather_tool_direct():
    client = DeepSeekAPI()

    response = client._hit_api_sync(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that uses tools when appropriate.",
            },
            {
                "role": "user",
                "content": "What's the weather in Paris? Use the tools and explain your reasoning.",
            },
        ],
        tools=[weather_tool],
        lm_config={
            "temperature": 0,
        },
    )

    # Check that we got a tool call
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["function"]["name"] == "get_weather"
    assert "Paris" in response.tool_calls[0]["function"]["arguments"]


def test_weather_tool_lm():
    lm = LM(
        model_name="deepseek-chat",
        formatting_model_name="deepseek-chat",
        temperature=0,
    )

    response = lm.respond_sync(
        system_message="You are a helpful assistant that uses tools when appropriate.",
        user_message="What's the weather in Paris? Use the tools and explain your reasoning.",
        tools=[weather_tool],
    )

    # Check that we got a tool call
    assert response.tool_calls is not None
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["function"]["name"] == "get_weather"
    assert "Paris" in response.tool_calls[0]["function"]["arguments"]
