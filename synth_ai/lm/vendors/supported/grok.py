import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI

from synth_ai.lm.tools.base import BaseTool
from synth_ai.lm.vendors.openai_standard import OpenAIStandard


class GrokAPI(OpenAIStandard):
    """
    Vendor shim for xAI Grok models.

    It re-uses ``OpenAIStandard`` because xAI's REST endpoints are
    OpenAI-compatible, including the ``tools`` / ``tool_choice`` function-calling
    interface.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.x.ai/v1",
    ) -> None:
        api_key = api_key or os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("Set the XAI_API_KEY environment variable or pass api_key explicitly.")

        super().__init__(
            sync_client=OpenAI(api_key=api_key, base_url=base_url),
            async_client=AsyncOpenAI(api_key=api_key, base_url=base_url),
            used_for_structured_outputs=True,
        )

    async def _hit_api_async(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
    ):
        if not model:
            raise ValueError("Model name is required for Grok API calls")

        return await super()._hit_api_async(
            model,
            messages,
            lm_config,
            use_ephemeral_cache_only=use_ephemeral_cache_only,
            reasoning_effort=reasoning_effort,
            tools=tools,
        )

    def _hit_api_sync(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        lm_config: Dict[str, Any],
        use_ephemeral_cache_only: bool = False,
        reasoning_effort: str = "high",
        tools: Optional[List[BaseTool]] = None,
    ):
        if not model:
            raise ValueError("Model name is required for Grok API calls")

        return super()._hit_api_sync(
            model,
            messages,
            lm_config,
            use_ephemeral_cache_only=use_ephemeral_cache_only,
            reasoning_effort=reasoning_effort,
            tools=tools,
        )
