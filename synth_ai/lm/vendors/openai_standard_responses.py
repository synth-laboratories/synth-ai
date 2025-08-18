"""
OpenAI Responses API extensions for OpenAIStandard vendor.

This module contains the Responses API and Harmony encoding methods
that extend the OpenAIStandard class functionality.
"""

import uuid
from typing import Any

from synth_ai.lm.tools.base import BaseTool
from synth_ai.lm.vendors.base import BaseLMResponse


def _silent_backoff_handler(_details):
    """No-op handler to keep stdout clean while still allowing visibility via logging if desired."""
    pass


DEFAULT_EXCEPTIONS_TO_RETRY = (
    Exception,  # Will be more specific when imported
)


class OpenAIResponsesAPIMixin:
    """Mixin class providing Responses API functionality for OpenAI vendors."""

    async def _hit_api_async_responses(
        self,
        model: str,
        messages: list[dict[str, Any]],
        lm_config: dict[str, Any],
        previous_response_id: str | None = None,
        use_ephemeral_cache_only: bool = False,
        tools: list[BaseTool] | None = None,
    ) -> BaseLMResponse:
        """Use OpenAI Responses API for supported models."""

        print(f"🔍 RESPONSES API: Called for model {model}")
        print(f"🔍 RESPONSES API: previous_response_id = {previous_response_id}")

        # Check if the client has responses attribute
        if not hasattr(self.async_client, "responses"):
            print("🔍 RESPONSES API: Client doesn't have responses attribute, using fallback")
            # Fallback - use chat completions with simulated response_id
            response = await self._hit_api_async(
                model=model,
                messages=messages,
                lm_config=lm_config,
                use_ephemeral_cache_only=use_ephemeral_cache_only,
                tools=tools,
            )

            # Add Responses API fields
            if not response.response_id:
                import uuid

                response.response_id = str(uuid.uuid4())
            response.api_type = "responses"
            return response

        # Use the official Responses API
        try:
            # Common API call params for Responses API
            api_params = {
                "model": model,
            }

            # For Responses API, we use 'input' parameter
            if previous_response_id:
                # Continue existing thread
                api_params["previous_response_id"] = previous_response_id
                # Only pass the new user input
                if messages and len(messages) > 0:
                    # Get the last user message content
                    last_message = messages[-1]
                    api_params["input"] = last_message.get("content", "")
            else:
                # Start new thread - combine system and user messages into input
                if messages and len(messages) > 0:
                    # Combine messages into a single input string
                    input_parts = []
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "system":
                            input_parts.append(f"System: {content}")
                        elif role == "user":
                            input_parts.append(f"User: {content}")
                        elif role == "assistant":
                            input_parts.append(f"Assistant: {content}")
                    api_params["input"] = "\n".join(input_parts)

            # Add tools if provided
            if tools and all(isinstance(tool, BaseTool) for tool in tools):
                api_params["tools"] = [tool.to_openai_tool() for tool in tools]
            elif tools:
                api_params["tools"] = tools

            # Add other parameters from lm_config if needed
            if "max_tokens" in lm_config:
                api_params["max_tokens"] = lm_config["max_tokens"]

            print(f"🔍 RESPONSES API: Calling with params: {list(api_params.keys())}")

            # Call the Responses API
            response = await self.async_client.responses.create(**api_params)

            print(f"🔍 RESPONSES API: Response received, type: {type(response)}")

            # Extract fields from response
            output_text = getattr(response, "output_text", getattr(response, "content", ""))
            reasoning_obj = getattr(response, "reasoning", None)
            response_id = getattr(response, "id", None)

            # Debug reasoning type (only first time)
            if reasoning_obj and not hasattr(self, "_reasoning_logged"):
                print(f"🔍 RESPONSES API: Reasoning type: {type(reasoning_obj)}")
                print(
                    f"🔍 RESPONSES API: Reasoning attributes: {[x for x in dir(reasoning_obj) if not x.startswith('_')]}"
                )
                self._reasoning_logged = True

            # Handle reasoning - it might be an object or a string
            reasoning = None
            if reasoning_obj:
                if isinstance(reasoning_obj, str):
                    # Synth backend returns full reasoning as string
                    reasoning = reasoning_obj
                else:
                    # OpenAI returns a Reasoning object
                    # Try to get summary first, but preserve entire object if no summary
                    if hasattr(reasoning_obj, "summary") and reasoning_obj.summary:
                        reasoning = reasoning_obj.summary
                    else:
                        # Preserve the full object structure as JSON
                        # This includes effort level and any other fields
                        if hasattr(reasoning_obj, "model_dump_json"):
                            reasoning = reasoning_obj.model_dump_json()
                        elif hasattr(reasoning_obj, "to_dict"):
                            import json

                            reasoning = json.dumps(reasoning_obj.to_dict())
                        else:
                            reasoning = str(reasoning_obj)

            # Handle tool calls if present
            tool_calls = None
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response.tool_calls
                ]

            print(f"🔍 RESPONSES API: Extracted response_id = {response_id}")

            return BaseLMResponse(
                raw_response=output_text,
                response_id=response_id,
                reasoning=reasoning,
                api_type="responses",
                tool_calls=tool_calls,
            )

        except (AttributeError, Exception) as e:
            print(f"🔍 RESPONSES API: Error calling Responses API: {e}")
            # No fallback - raise the error
            raise

    async def _hit_api_async_harmony(
        self,
        model: str,
        messages: list[dict[str, Any]],
        lm_config: dict[str, Any],
        previous_response_id: str | None = None,
        use_ephemeral_cache_only: bool = False,
        tools: list[BaseTool] | None = None,
    ) -> BaseLMResponse:
        """Use Harmony encoding for OSS-GPT models."""
        if not self.harmony_available:
            raise ImportError(
                "openai-harmony package required for OSS-GPT models. Install with: pip install openai-harmony"
            )

        from openai_harmony import Conversation, Message, Role

        # Convert messages to Harmony format
        harmony_messages = []
        for msg in messages:
            role = (
                Role.SYSTEM
                if msg["role"] == "system"
                else (Role.USER if msg["role"] == "user" else Role.ASSISTANT)
            )
            content = msg["content"]
            # Handle multimodal content
            if isinstance(content, list):
                # Extract text content for now
                text_parts = [
                    part.get("text", "") for part in content if part.get("type") == "text"
                ]
                content = " ".join(text_parts)
            harmony_messages.append(Message.from_role_and_content(role, content))

        conv = Conversation.from_messages(harmony_messages)
        tokens = self.harmony_enc.render_conversation_for_completion(conv, Role.ASSISTANT)

        # For now, we'll need to integrate with Synth GPU endpoint
        # This would require the actual endpoint to be configured
        # Placeholder for actual Synth GPU call
        import os

        import aiohttp

        synth_gpu_endpoint = os.getenv("SYNTH_GPU_HARMONY_ENDPOINT")
        if not synth_gpu_endpoint:
            raise ValueError("SYNTH_GPU_HARMONY_ENDPOINT environment variable not set")

        async with aiohttp.ClientSession() as session, session.post(
                f"{synth_gpu_endpoint}/v1/completions",
                json={
                    "model": model,
                    "prompt": tokens,
                    "max_tokens": lm_config.get("max_tokens", 4096),
                    "temperature": lm_config.get("temperature", 0.8),
                },
            ) as resp:
                result = await resp.json()

        # Parse response using Harmony
        response_tokens = result.get("choices", [{}])[0].get("text", "")
        parsed = self.harmony_enc.parse_messages_from_completion_tokens(
            response_tokens, Role.ASSISTANT
        )

        if parsed:
            assistant_msg = (
                parsed[-1].content_text()
                if hasattr(parsed[-1], "content_text")
                else str(parsed[-1])
            )
        else:
            assistant_msg = response_tokens

        return BaseLMResponse(
            raw_response=assistant_msg,
            response_id=previous_response_id or str(uuid.uuid4()),
            api_type="harmony",
        )
