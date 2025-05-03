from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from synth_ai.zyk.lms.core.exceptions import StructuredOutputCoercionFailureException
from synth_ai.zyk.lms.core.vendor_clients import (
    anthropic_naming_regexes,
    get_client,
    openai_naming_regexes,
)
from synth_ai.zyk.lms.structured_outputs.handler import StructuredOutputHandler
from synth_ai.zyk.lms.vendors.base import VendorBase
from synth_ai.zyk.lms.tools.base import BaseTool
from synth_ai.zyk.lms.config import reasoning_models

def build_messages(
    sys_msg: str,
    user_msg: str,
    images_bytes: List = [],
    model_name: Optional[str] = None,
) -> List[Dict]:
    if len(images_bytes) > 0 and any(
        regex.match(model_name) for regex in openai_naming_regexes
    ):
        return [
            {"role": "system", "content": sys_msg},
            {
                "role": "user",
                "content": [{"type": "text", "text": user_msg}]
                + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_bytes}"},
                    }
                    for image_bytes in images_bytes
                ],
            },
        ]
    elif len(images_bytes) > 0 and any(
        regex.match(model_name) for regex in anthropic_naming_regexes
    ):
        system_info = {"role": "system", "content": sys_msg}
        user_info = {
            "role": "user",
            "content": [{"type": "text", "text": user_msg}]
            + [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_bytes,
                    },
                }
                for image_bytes in images_bytes
            ],
        }
        return [system_info, user_info]
    elif len(images_bytes) > 0:
        raise ValueError("Images are not yet supported for this model")
    else:
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]


class LM:
    # if str
    model_name: str
    client: VendorBase
    lm_config: Dict[str, Any]
    structured_output_handler: StructuredOutputHandler

    def __init__(
        self,
        model_name: str,
        formatting_model_name: str,
        temperature: float,
        max_retries: Literal["None", "Few", "Many"] = "Few",
        structured_output_mode: Literal[
            "stringified_json", "forced_json"
        ] = "stringified_json",
        synth_logging: bool = True,
    ):
        # print("Structured output mode", structured_output_mode)
        self.client = get_client(
            model_name,
            with_formatting=structured_output_mode == "forced_json",
            synth_logging=synth_logging,
        )
        # print(self.client.__class__)

        formatting_client = get_client(formatting_model_name, with_formatting=True)

        max_retries_dict = {"None": 0, "Few": 2, "Many": 5}
        self.structured_output_handler = StructuredOutputHandler(
            self.client,
            formatting_client,
            structured_output_mode,
            {"max_retries": max_retries_dict.get(max_retries, 2)},
        )
        self.backup_structured_output_handler = StructuredOutputHandler(
            self.client,
            formatting_client,
            "forced_json",
            {"max_retries": max_retries_dict.get(max_retries, 2)},
        )
        # Override temperature to 1 for reasoning models
        effective_temperature = 1.0 if model_name in reasoning_models else temperature
        self.lm_config = {"temperature": effective_temperature}
        self.model_name = model_name

    def respond_sync(
        self,
        system_message: Optional[str] = None,
        user_message: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        images_as_bytes: List[Any] = [],
        response_model: Optional[BaseModel] = None,
        use_ephemeral_cache_only: bool = False,
        tools: Optional[List[BaseTool]] = None,
        reasoning_effort: str = "low",
    ):
        assert (system_message is None) == (
            user_message is None
        ), "Must provide both system_message and user_message or neither"
        assert (
            (messages is None) != (system_message is None)
        ), "Must provide either messages or system_message/user_message pair, but not both"
        assert not (response_model and tools), "Cannot provide both response_model and tools"
        if messages is None:
            messages = build_messages(
                system_message, user_message, images_as_bytes, self.model_name
            )
        result = None
        if response_model:
            try:
                result = self.structured_output_handler.call_sync(
                    messages,
                    model=self.model_name,
                    lm_config=self.lm_config,
                    response_model=response_model,
                    use_ephemeral_cache_only=use_ephemeral_cache_only,
                    reasoning_effort=reasoning_effort,
                )
            except StructuredOutputCoercionFailureException:
                # print("Falling back to backup handler")
                result = self.backup_structured_output_handler.call_sync(
                    messages,
                    model=self.model_name,
                    lm_config=self.lm_config,
                    response_model=response_model,
                    use_ephemeral_cache_only=use_ephemeral_cache_only,
                    reasoning_effort=reasoning_effort,
                )
        else:
            result = self.client._hit_api_sync(
                messages=messages,
                model=self.model_name,
                lm_config=self.lm_config,
                use_ephemeral_cache_only=use_ephemeral_cache_only,
                tools=tools,
                reasoning_effort=reasoning_effort,
            )
        assert isinstance(result.raw_response, str), "Raw response must be a string"
        assert (isinstance(result.structured_output, BaseModel) or result.structured_output is None), "Structured output must be a Pydantic model or None"
        assert (isinstance(result.tool_calls, list) or result.tool_calls is None), "Tool calls must be a list or None"
        return result

    async def respond_async(
        self,
        system_message: Optional[str] = None,
        user_message: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        images_as_bytes: List[Any] = [],
        response_model: Optional[BaseModel] = None,
        use_ephemeral_cache_only: bool = False,
        tools: Optional[List[BaseTool]] = None,
        reasoning_effort: str = "low",
    ):
        # "In respond_async")
        assert (system_message is None) == (
            user_message is None
        ), "Must provide both system_message and user_message or neither"
        assert (
            (messages is None) != (system_message is None)
        ), "Must provide either messages or system_message/user_message pair, but not both"

        assert not (response_model and tools), "Cannot provide both response_model and tools"
        if messages is None:
            messages = build_messages(
                system_message, user_message, images_as_bytes, self.model_name
            )
        result = None
        if response_model:
            try:
                print("Trying structured output handler")
                result = await self.structured_output_handler.call_async(
                    messages,
                    model=self.model_name,
                    lm_config=self.lm_config,
                    response_model=response_model,
                    use_ephemeral_cache_only=use_ephemeral_cache_only,
                    reasoning_effort=reasoning_effort,
                )
            except StructuredOutputCoercionFailureException:
                print("Falling back to backup handler")
                result = await self.backup_structured_output_handler.call_async(
                    messages,
                    model=self.model_name,
                    lm_config=self.lm_config,
                    response_model=response_model,
                    use_ephemeral_cache_only=use_ephemeral_cache_only,
                    reasoning_effort=reasoning_effort,
                )
        else:
            #print("Calling API no response model")
            result = await self.client._hit_api_async(
                messages=messages,
                model=self.model_name,
                lm_config=self.lm_config,
                use_ephemeral_cache_only=use_ephemeral_cache_only,
                tools=tools,
                reasoning_effort=reasoning_effort,
            )
        assert isinstance(result.raw_response, str), "Raw response must be a string"
        assert (isinstance(result.structured_output, BaseModel) or result.structured_output is None), "Structured output must be a Pydantic model or None"
        assert (isinstance(result.tool_calls, list) or result.tool_calls is None), "Tool calls must be a list or None"
        return result

if __name__ == "__main__":
    import asyncio

    # Update json instructions to handle nested pydantic?
    class Thought(BaseModel):
        argument_keys: List[str] = Field(description="The keys of the arguments")
        argument_values: List[str] = Field(
            description="Stringified JSON for the values of the arguments"
        )

    class TestModel(BaseModel):
        emotion: str = Field(description="The emotion expressed")
        concern: str = Field(description="The concern expressed")
        action: str = Field(description="The action to be taken")
        thought: Thought = Field(description="The thought process")

        class Config:
            schema_extra = {"required": ["thought", "emotion", "concern", "action"]}

    lm = LM(
        model_name="gpt-4o-mini",
        formatting_model_name="gpt-4o-mini",
        temperature=1,
        max_retries="Few",
        structured_output_mode="forced_json",
    )
    print(
        asyncio.run(
            lm.respond_async(
                system_message="You are a helpful assistant ",
                user_message="Hello, how are you?",
                images_as_bytes=[],
                response_model=TestModel,
                use_ephemeral_cache_only=False,
            )
        )
    )
