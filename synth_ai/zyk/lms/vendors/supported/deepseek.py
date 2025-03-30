import os
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI, OpenAI

from synth_ai.zyk.lms.tools.base import BaseTool
from synth_ai.zyk.lms.vendors.openai_standard import OpenAIStandard


class DeepSeekAPI(OpenAIStandard):
    def __init__(self):
        # print("Setting up DeepSeek API")
        self.sync_client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.async_client = AsyncOpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    def _convert_tools_to_openai_format(self, tools: List[BaseTool]) -> List[Dict]:
        return [tool.to_openai_tool() for tool in tools]

    async def _private_request_async(
        self,
        messages: List[Dict],
        temperature: float = 0,
        model_name: str = "deepseek-chat",
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
        lm_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[List[Dict]]]:
        request_params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            request_params["tools"] = self._convert_tools_to_openai_format(tools)

        response = await self.async_client.chat.completions.create(**request_params)
        message = response.choices[0].message

        return message.content, message.tool_calls if hasattr(
            message, "tool_calls"
        ) else None

    def _private_request_sync(
        self,
        messages: List[Dict],
        temperature: float = 0,
        model_name: str = "deepseek-chat",
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
        lm_config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Optional[List[Dict]]]:
        request_params = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }

        if tools:
            request_params["tools"] = self._convert_tools_to_openai_format(tools)

        response = self.sync_client.chat.completions.create(**request_params)
        message = response.choices[0].message

        return message.content, message.tool_calls if hasattr(
            message, "tool_calls"
        ) else None
