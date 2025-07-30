"""
Drop-in replacement for anthropic.Client to log requests with Langfuse and track messages using Synth SDK.
Analogous to the modified OpenAI version.
"""

import logging
import types
from dataclasses import dataclass
from typing import Optional

try:
    import anthropic
except ImportError:
    raise ModuleNotFoundError(
        "Please install anthropic to use this feature: 'pip install anthropic'"
    )

try:
    from anthropic import AsyncClient, Client
except ImportError:
    Client = None
    AsyncClient = None

from langfuse import Langfuse
from langfuse.client import StatefulGenerationClient
from langfuse.decorators import langfuse_context
from langfuse.utils import _get_timestamp
from langfuse.utils.langfuse_singleton import LangfuseSingleton
from wrapt import wrap_function_wrapper

from synth_ai.lm.provider_support.suppress_logging import *
from synth_ai.tracing_v1.trackers import (
    synth_tracker_async,
    synth_tracker_sync,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust as needed

# CREDIT TO LANGFUSE FOR OPEN-SOURCING THE CODE THAT THIS IS BASED ON
# USING WITH MIT LICENSE PERMISSION
# https://langfuse.com


@dataclass
class AnthropicDefinition:
    module: str
    object: str
    method: str
    sync: bool


ANTHROPIC_METHODS = [
    AnthropicDefinition(
        module="anthropic.Client",
        object="completions",
        method="create",
        sync=True,
    ),
    AnthropicDefinition(
        module="anthropic.Client",
        object="completions",
        method="stream",
        sync=True,
    ),
    AnthropicDefinition(
        module="anthropic.AsyncClient",
        object="completions",
        method="create",
        sync=False,
    ),
    AnthropicDefinition(
        module="anthropic.AsyncClient",
        object="completions",
        method="stream",
        sync=False,
    ),
    AnthropicDefinition(
        module="anthropic.Client",
        object="messages",
        method="create",
        sync=True,
    ),
    AnthropicDefinition(
        module="anthropic.AsyncClient",
        object="messages",
        method="create",
        sync=False,
    ),
]


class AnthropicArgsExtractor:
    def __init__(
        self,
        name=None,
        metadata=None,
        trace_id=None,
        session_id=None,
        user_id=None,
        tags=None,
        parent_observation_id=None,
        langfuse_prompt=None,
        **kwargs,
    ):
        self.args = {
            "name": name,
            "metadata": metadata,
            "trace_id": trace_id,
            "session_id": session_id,
            "user_id": user_id,
            "tags": tags,
            "parent_observation_id": parent_observation_id,
            "langfuse_prompt": langfuse_prompt,
        }
        self.kwargs = kwargs

    def get_langfuse_args(self):
        return {**self.args, **self.kwargs}

    def get_anthropic_args(self):
        return self.kwargs


def _langfuse_wrapper(func):
    def _with_langfuse(anthropic_resource, initialize):
        def wrapper(wrapped, instance, args, kwargs):
            return func(anthropic_resource, initialize, wrapped, args, kwargs)

        return wrapper

    return _with_langfuse


def _extract_anthropic_prompt(kwargs: dict) -> str:
    """Return the user prompt if present, else empty."""
    logger.debug(f"Extracting prompt from kwargs: {kwargs}")

    # Handle Messages API format
    if "messages" in kwargs:
        messages = kwargs["messages"]
        logger.debug(f"Found messages format: {messages}")
        # Extract the last user message
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        return user_messages[-1] if user_messages else ""

    # Handle Completions API format
    return kwargs.get("prompt", "")


def _extract_anthropic_completion(response):
    """Extract final completion, model, usage from the anthropic response."""
    if not response:
        return None, "<NoneType response returned from Anthropic>", None

    model = getattr(response, "model", None)
    raw_usage = getattr(response, "usage", None)

    # Handle content which might be a TextBlock or list of TextBlocks
    content = getattr(response, "content", None) or getattr(response, "completion", None)
    if isinstance(content, list):
        # Handle list of TextBlocks
        completion = " ".join(
            block.text if hasattr(block, "text") else str(block) for block in content
        )
    elif hasattr(content, "text"):
        # Handle single TextBlock
        completion = content.text
    else:
        completion = str(content) if content is not None else ""

    # Convert Anthropic usage format to Langfuse format
    if raw_usage:
        usage = {
            "promptTokens": getattr(raw_usage, "input_tokens", 0),
            "completionTokens": getattr(raw_usage, "output_tokens", 0),
            "totalTokens": getattr(raw_usage, "total_tokens", 0),
        }
    else:
        usage = {"promptTokens": 0, "completionTokens": 0, "totalTokens": 0}

    return model, completion, usage


def _extract_streamed_anthropic_response(items):
    """Extract final completion, model, usage from streamed anthropic response."""
    if not items:
        return None, "<Empty response from Anthropic>", None

    last_item = items[-1]
    model = getattr(last_item, "model", None)
    raw_usage = getattr(last_item, "usage", None)

    # Combine all content pieces, handling TextBlocks
    completion_parts = []
    for item in items:
        content = getattr(item, "content", None) or getattr(item, "completion", None)
        if isinstance(content, list):
            # Handle list of TextBlocks
            completion_parts.extend(
                block.text if hasattr(block, "text") else str(block) for block in content
            )
        elif hasattr(content, "text"):
            # Handle single TextBlock
            completion_parts.append(content.text)
        elif content:
            completion_parts.append(str(content))

    completion = " ".join(completion_parts)

    # Convert usage format
    if raw_usage:
        usage = {
            "promptTokens": getattr(raw_usage, "input_tokens", 0),
            "completionTokens": getattr(raw_usage, "output_tokens", 0),
            "totalTokens": getattr(raw_usage, "total_tokens", 0),
        }
    else:
        usage = {"promptTokens": 0, "completionTokens": 0, "totalTokens": 0}

    return model, completion, usage


def _get_langfuse_data_from_kwargs(anthropic_resource, langfuse: Langfuse, start_time, kwargs):
    name = kwargs.get("name", "Anthropic-generation")
    if name is not None and not isinstance(name, str):
        raise TypeError("name must be a string")

    decorator_context_observation_id = langfuse_context.get_current_observation_id()
    decorator_context_trace_id = langfuse_context.get_current_trace_id()

    trace_id = kwargs.get("trace_id", None) or decorator_context_trace_id
    if trace_id is not None and not isinstance(trace_id, str):
        raise TypeError("trace_id must be a string")

    session_id = kwargs.get("session_id", None)
    if session_id is not None and not isinstance(session_id, str):
        raise TypeError("session_id must be a string")

    user_id = kwargs.get("user_id", None)
    if user_id is not None and not isinstance(user_id, str):
        raise TypeError("user_id must be a string")

    tags = kwargs.get("tags", None)
    if tags is not None and (
        not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags)
    ):
        raise TypeError("tags must be a list of strings")

    if decorator_context_trace_id:
        langfuse_context.update_current_trace(session_id=session_id, user_id=user_id, tags=tags)

    parent_observation_id = kwargs.get("parent_observation_id", None) or (
        decorator_context_observation_id
        if decorator_context_observation_id != decorator_context_trace_id
        else None
    )
    if parent_observation_id is not None and not isinstance(parent_observation_id, str):
        raise TypeError("parent_observation_id must be a string")
    if parent_observation_id is not None and trace_id is None:
        raise ValueError("parent_observation_id requires trace_id to be set")

    metadata = kwargs.get("metadata", {})
    if metadata is not None and not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")

    # Collect user prompt and model from arguments
    prompt = _extract_anthropic_prompt(kwargs)
    model = kwargs.get("model", None)
    # If user supplied inputs for a model in some nested structure, consider hooking in here.

    # Basic hyperparams
    model_params = {
        "temperature": kwargs.get("temperature", 1.0),
        "max_tokens": kwargs.get("max_tokens_to_sample", None),
        "top_p": kwargs.get("top_p", None),
    }

    is_nested_trace = False
    if trace_id:
        is_nested_trace = True
        langfuse.trace(id=trace_id, session_id=session_id, user_id=user_id, tags=tags)
    else:
        trace_instance = langfuse.trace(
            session_id=session_id,
            user_id=user_id,
            tags=tags,
            name=name,
            input=prompt,
            metadata=metadata,
        )
        trace_id = trace_instance.id

    langfuse_prompt = kwargs.get("langfuse_prompt", None)

    return (
        {
            "name": name,
            "metadata": metadata,
            "trace_id": trace_id,
            "parent_observation_id": parent_observation_id,
            "user_id": user_id,
            "start_time": start_time,
            "input": prompt,
            "model_params": model_params,
            "prompt": langfuse_prompt,
            "model": model,
        },
        is_nested_trace,
    )


def _create_langfuse_update(
    completion,
    generation: StatefulGenerationClient,
    completion_start_time,
    model=None,
    usage=None,
    model_params=None,
):
    update = {
        "end_time": _get_timestamp(),
        "output": completion,
        "completion_start_time": completion_start_time,
    }
    if model:
        if not model_params:
            model_params = {}
        model_params["model_name"] = model
    if model_params is not None:
        update["model_params"] = model_params
    if usage is not None:
        update["usage"] = usage
    generation.update(**update)


@_langfuse_wrapper
def _wrap(anthropic_resource: AnthropicDefinition, initialize, wrapped, args, kwargs):
    # print("\n=== WRAP START ===")
    # print(f"WRAP: Args: {args}")
    # print(f"WRAP: Kwargs: {kwargs}")

    new_langfuse = initialize()
    start_time = _get_timestamp()
    arg_extractor = AnthropicArgsExtractor(*args, **kwargs)
    generation_data, is_nested_trace = _get_langfuse_data_from_kwargs(
        anthropic_resource, new_langfuse, start_time, arg_extractor.get_langfuse_args()
    )
    generation = new_langfuse.generation(**generation_data)

    try:
        anthropic_response = wrapped(*args, **arg_extractor.get_anthropic_args())

        # If it's a streaming call, returns a generator
        if isinstance(anthropic_response, types.GeneratorType):
            return LangfuseAnthropicResponseGeneratorSync(
                response=anthropic_response,
                generation=generation,
                langfuse=new_langfuse,
                is_nested_trace=is_nested_trace,
                kwargs=arg_extractor.get_anthropic_args(),
            )
        else:
            model, completion, usage = _extract_anthropic_completion(anthropic_response)
            # Synth tracking
            if "messages" in arg_extractor.get_anthropic_args():
                # print("\nWRAP: Messages API path")
                system_content = arg_extractor.get_anthropic_args().get("system")
                original_messages = arg_extractor.get_anthropic_args()["messages"]
                # print(f"WRAP: Original messages: {original_messages}")
                # print(f"WRAP: System content: {system_content}")

                if system_content:
                    messages = [{"role": "system", "content": system_content}] + original_messages
                else:
                    messages = original_messages

                # print(f"WRAP: Final messages to track: {messages}")
                # print("WRAP: About to call track_lm")
                synth_tracker_sync.track_lm(
                    messages=messages,
                    model_name=model,
                    model_params=generation_data.get("model_params", {}),
                    finetune=False,
                )
                # print("WRAP: Finished track_lm call")

                # Track assistant output
                assistant_msg = [{"role": "assistant", "content": completion}]
                # rint("About to track LM output")
                # print("Assistant message: %s", assistant_msg)

                synth_tracker_sync.track_lm_output(
                    messages=assistant_msg,
                    model_name=model,
                    finetune=False,
                )
                # print("Finished tracking LM output")

            elif "prompt" in arg_extractor.get_anthropic_args():
                # print("\nWRAP: Completions API path")
                user_prompt = arg_extractor.get_anthropic_args().get("prompt", "")
                # print(f"WRAP: User prompt: {user_prompt}")
                messages = [{"role": "user", "content": user_prompt}]
                # print(f"WRAP: Messages created: {messages}")
                assistant_msg = [{"role": "assistant", "content": completion}]

                # print("About to track LM call with model: %s", model)
                # print("User prompt: %s", user_prompt)
                # print("Messages to track: %s", messages)
                # print("Model params: %s", generation_data.get("model_params", {}))

                synth_tracker_sync.track_lm(
                    messages=messages,
                    model_name=model,
                    model_params=generation_data.get("model_params", {}),
                    finetune=False,
                )

                # print("About to track LM output")
                # print("Assistant message: %s", assistant_msg)

                synth_tracker_sync.track_lm_output(
                    messages=assistant_msg,
                    model_name=model,
                    finetune=False,
                )
                # print("Finished tracking LM output")

            # Complete the generation update
            _create_langfuse_update(
                completion,
                generation,
                start_time,
                model=model,
                usage=usage,
                model_params=generation_data.get("model_params", {}),
            )
            if not is_nested_trace:
                new_langfuse.trace(id=generation.trace_id, output=completion)

        return anthropic_response
    except Exception as ex:
        model_params = generation_data.get("model_params", {})
        generation.update(
            end_time=_get_timestamp(),
            status_message=str(ex),
            level="ERROR",
            model_params=model_params,
            usage={"promptTokens": 0, "completionTokens": 0, "totalTokens": 0},
        )
        raise ex


@_langfuse_wrapper
async def _wrap_async(anthropic_resource: AnthropicDefinition, initialize, wrapped, args, kwargs):
    # print("\n=== WRAP_ASYNC START ===")
    # print(f"WRAP_ASYNC: Args: {args}")
    # print(f"WRAP_ASYNC: Kwargs: {kwargs}")

    new_langfuse = initialize()
    start_time = _get_timestamp()
    arg_extractor = AnthropicArgsExtractor(*args, **kwargs)

    # Initialize tracker if needed
    if not hasattr(synth_tracker_async, "_local") or not getattr(
        synth_tracker_async._local, "initialized", False
    ):
        synth_tracker_async.initialize()
        # print("WRAP_ASYNC: Initialized async tracker")

    generation_data, is_nested_trace = _get_langfuse_data_from_kwargs(
        anthropic_resource, new_langfuse, start_time, arg_extractor.get_langfuse_args()
    )
    generation = new_langfuse.generation(**generation_data)

    try:
        logger.debug("About to call wrapped function")
        response = await wrapped(*args, **kwargs)
        logger.debug(f"Got response: {response}")

        model, completion, usage = _extract_anthropic_completion(response)
        logger.debug(f"Extracted completion - Model: {model}, Usage: {usage}")

        # Synth tracking
        if "messages" in arg_extractor.get_anthropic_args():
            # logger.debug("WRAP_ASYNC: Messages API path detected")
            system_content = arg_extractor.get_anthropic_args().get("system")
            original_messages = arg_extractor.get_anthropic_args()["messages"]
            # logger.debug("WRAP_ASYNC: Original messages: %s", original_messages)
            # logger.debug("WRAP_ASYNC: System content: %s", system_content)

            if system_content:
                messages = [{"role": "system", "content": system_content}] + original_messages
            else:
                messages = original_messages

            # logger.debug("WRAP_ASYNC: About to track messages: %s", messages)
            synth_tracker_async.track_lm(
                messages=messages,
                model_name=model,
                model_params=generation_data.get("model_params", {}),
                finetune=False,
            )

            # Track assistant output
            assistant_msg = [{"role": "assistant", "content": completion}]
            logger.debug("Tracking assistant message: %s", assistant_msg)
            synth_tracker_async.track_lm_output(
                messages=assistant_msg,
                model_name=model,
                finetune=False,
            )
        elif "prompt" in arg_extractor.get_anthropic_args():
            # Handle Completions API format
            user_prompt = arg_extractor.get_anthropic_args().get("prompt", "")
            messages = [{"role": "user", "content": user_prompt}]
            assistant_msg = [{"role": "assistant", "content": completion}]

            logger.debug("About to track async LM call with model: %s", model)
            logger.debug("User prompt: %s", user_prompt)
            logger.debug("Messages to track: %s", messages)
            logger.debug("Model params: %s", generation_data.get("model_params", {}))

            # Track input
            # SynthTracker.track_lm(
            #     messages=messages,
            #     model_name=model,
            #     model_params=generation_data.get("model_params", {}),
            #     finetune=False,
            # )

            logger.debug("About to track async LM output")
            logger.debug("Assistant message: %s", assistant_msg)

            # Track output
            # SynthTracker.track_lm_output(
            #     messages=assistant_msg,
            #     model_name=model,
            #     finetune=False,
            # )

        # Complete the generation update
        _create_langfuse_update(
            completion,
            generation,
            start_time,
            model=model,
            usage=usage,
            model_params=generation_data.get("model_params", {}),
        )
        if not is_nested_trace:
            new_langfuse.trace(id=generation.trace_id, output=completion)

        return response

    except Exception as ex:
        model_params = generation_data.get("model_params", {})
        generation.update(
            end_time=_get_timestamp(),
            status_message=str(ex),
            level="ERROR",
            model_params=model_params,
            usage={"promptTokens": 0, "completionTokens": 0, "totalTokens": 0},
        )
        raise ex


class LangfuseAnthropicResponseGeneratorSync:
    def __init__(self, *, response, generation, langfuse, is_nested_trace, kwargs):
        self.response = response
        self.generation = generation
        self.langfuse = langfuse
        self.is_nested_trace = is_nested_trace
        self.kwargs = kwargs
        self.items = []
        self.completion_start_time = None

    def __iter__(self):
        try:
            for chunk in self.response:
                self.items.append(chunk)
                if self.completion_start_time is None:
                    self.completion_start_time = _get_timestamp()
                yield chunk
        finally:
            self._finalize()

    def __next__(self):
        try:
            chunk = next(self.response)
            self.items.append(chunk)
            if self.completion_start_time is None:
                self.completion_start_time = _get_timestamp()
            return chunk
        except StopIteration:
            self._finalize()
            raise

    def _finalize(self):
        print("\n=== FINALIZE START ===")
        print(f"FINALIZE: Self kwargs: {self.kwargs}")
        model, completion, usage = _extract_streamed_anthropic_response(self.items)

        if "messages" in self.kwargs:
            print("\nFINALIZE: Messages API path")
            system_content = self.kwargs.get("system")
            original_messages = self.kwargs["messages"]
            print(f"FINALIZE: Original messages: {original_messages}")
            print(f"FINALIZE: System content: {system_content}")

            if system_content:
                messages = [{"role": "system", "content": system_content}] + original_messages
            else:
                messages = original_messages

            print(f"FINALIZE: Final messages to track: {messages}")
            print("FINALIZE: About to call track_lm")
            # synth_tracker_sync.track_lm(
            #     messages=messages,
            #     model_name=model,
            #     model_params=self.generation.model_params or {},
            #     finetune=False,
            # )
            print("FINALIZE: Finished track_lm call")

            # Track assistant output
            assistant_msg = [{"role": "assistant", "content": completion}]
            print("Tracking assistant message: %s", assistant_msg)
            # synth_tracker_sync.track_lm_output(
            #     messages=assistant_msg,
            #     model_name=model,
            #     finetune=False,
            # )
        elif "prompt" in self.kwargs:
            print("\nFINALIZE: Completions API path")
            user_prompt = self.kwargs.get("prompt", "")
            print(f"FINALIZE: User prompt: {user_prompt}")
            messages = [{"role": "user", "content": user_prompt}]
            print(f"FINALIZE: Messages created: {messages}")
            assistant_msg = [{"role": "assistant", "content": completion}]

            # synth_tracker_sync.track_lm(
            #     messages=messages,
            #     model_name=model,
            #     model_params=self.generation.model_params or {},
            #     finetune=False,
            # )

            # synth_tracker_sync.track_lm_output(
            #     messages=assistant_msg,
            #     model_name=model,
            #     finetune=False,
            # )

        if not self.is_nested_trace:
            self.langfuse.trace(id=self.generation.trace_id, output=completion)
        _create_langfuse_update(
            completion,
            self.generation,
            self.completion_start_time,
            model=model,
            usage=usage,
            model_params=self.generation.model_params,
        )


class LangfuseAnthropicResponseGeneratorAsync:
    def __init__(self, *, response, generation, langfuse, is_nested_trace, kwargs):
        self.response = response
        self.generation = generation
        self.langfuse = langfuse
        self.is_nested_trace = is_nested_trace
        self.kwargs = kwargs
        self.items = []
        self.completion_start_time = None

    async def __aiter__(self):
        try:
            async for chunk in self.response:
                self.items.append(chunk)
                if self.completion_start_time is None:
                    self.completion_start_time = _get_timestamp()
                yield chunk
        finally:
            await self._finalize()

    async def __anext__(self):
        try:
            chunk = await self.response.__anext__()
            self.items.append(chunk)
            if self.completion_start_time is None:
                self.completion_start_time = _get_timestamp()
            return chunk
        except StopAsyncIteration:
            await self._finalize()
            raise

    async def _finalize(self):
        print("\n=== FINALIZE START ===")
        if not synth_tracker_async:
            print("ERROR: synth_tracker_async is not initialized!")
            raise RuntimeError("synth_tracker_async must be initialized before use")

        print(f"FINALIZE: Self kwargs: {self.kwargs}")
        model, completion, usage = _extract_streamed_anthropic_response(self.items)

        if "messages" in self.kwargs:
            print("\nFINALIZE: Messages API path")
            system_content = self.kwargs.get("system")
            original_messages = self.kwargs["messages"]
            print(f"FINALIZE: Original messages: {original_messages}")
            print(f"FINALIZE: System content: {system_content}")

            if system_content:
                messages = [{"role": "system", "content": system_content}] + original_messages
            else:
                messages = original_messages

            print(f"FINALIZE: Final messages to track: {messages}")
            print("FINALIZE: About to call track_lm")
            synth_tracker_async.track_lm(
                messages=messages,
                model_name=model,
                model_params=self.generation.model_params or {},
                finetune=False,
            )
            print("FINALIZE: Finished track_lm call")

            # Track assistant output
            assistant_msg = [{"role": "assistant", "content": completion}]
            print("Tracking assistant message: %s", assistant_msg)
            synth_tracker_async.track_lm_output(
                messages=assistant_msg,
                model_name=model,
                finetune=False,
            )
        elif "prompt" in self.kwargs:
            print("\nFINALIZE: Completions API path")
            user_prompt = self.kwargs.get("prompt", "")
            print(f"FINALIZE: User prompt: {user_prompt}")
            messages = [{"role": "user", "content": user_prompt}]
            print(f"FINALIZE: Messages created: {messages}")
            assistant_msg = [{"role": "assistant", "content": completion}]

            synth_tracker_async.track_lm(
                messages=messages,
                model_name=model,
                model_params=self.generation.model_params or {},
                finetune=False,
            )

            synth_tracker_async.track_lm_output(
                messages=assistant_msg,
                model_name=model,
                finetune=False,
            )

        if not self.is_nested_trace:
            self.langfuse.trace(id=self.generation.trace_id, output=completion)
        _create_langfuse_update(
            completion,
            self.generation,
            self.completion_start_time,
            model=model,
            usage=usage,
            model_params=self.generation.model_params,
        )

    async def close(self):
        await self.response.aclose()


class AnthropicLangfuse:
    _langfuse: Optional[Langfuse] = None

    def initialize(self):
        self._langfuse = LangfuseSingleton().get(
            public_key=getattr(anthropic, "langfuse_public_key", None),
            secret_key=getattr(anthropic, "langfuse_secret_key", None),
            host=getattr(anthropic, "langfuse_host", None),
            debug=getattr(anthropic, "langfuse_debug", None),
            enabled=getattr(anthropic, "langfuse_enabled", True),
            sdk_integration="anthropic",
            sample_rate=getattr(anthropic, "langfuse_sample_rate", None),
        )
        return self._langfuse

    def flush(self):
        if self._langfuse is not None:
            self._langfuse.flush()

    def langfuse_auth_check(self):
        if self._langfuse is None:
            self.initialize()
        return self._langfuse.auth_check()

    def register_tracing(self):
        # Patch anthropic.Client to wrap both completions and messages methods
        original_client_init = anthropic.Client.__init__

        def new_client_init(instance, *args, **kwargs):
            logger.debug("Initializing new Anthropic Client with tracing")
            original_client_init(instance, *args, **kwargs)

            # Wrap completions methods
            comp_obj = getattr(instance, "completions", None)
            if comp_obj is not None:
                logger.debug("Found completions object, wrapping methods")
                # Wrap 'create' method if available.
                if hasattr(comp_obj, "create"):
                    wrap_function_wrapper(
                        comp_obj,
                        "create",
                        _wrap(
                            next(
                                r
                                for r in ANTHROPIC_METHODS
                                if r.method == "create" and r.module == "anthropic.Client"
                            ),
                            self.initialize,
                        ),
                    )
                # Wrap 'stream' method only if it exists.
                if hasattr(comp_obj, "stream"):
                    wrap_function_wrapper(
                        comp_obj,
                        "stream",
                        _wrap(
                            next(
                                r
                                for r in ANTHROPIC_METHODS
                                if r.method == "stream" and r.module == "anthropic.Client"
                            ),
                            self.initialize,
                        ),
                    )

            # Wrap messages methods
            msg_obj = getattr(instance, "messages", None)
            if msg_obj is not None:
                logger.debug("Found messages object, wrapping methods")
                if hasattr(msg_obj, "create"):
                    wrap_function_wrapper(
                        msg_obj,
                        "create",
                        _wrap(
                            next(
                                r
                                for r in ANTHROPIC_METHODS
                                if r.method == "create"
                                and r.module == "anthropic.Client"
                                and r.object == "messages"
                            ),
                            self.initialize,
                        ),
                    )

        anthropic.Client.__init__ = new_client_init

        # Patch anthropic.AsyncClient similarly.
        original_async_init = anthropic.AsyncClient.__init__

        def new_async_init(instance, *args, **kwargs):
            logger.debug("Initializing new Async Anthropic Client with tracing")
            original_async_init(instance, *args, **kwargs)

            # Wrap completions methods
            comp_obj = getattr(instance, "completions", None)
            if comp_obj is not None:
                logger.debug("Found async completions object, wrapping methods")
                if hasattr(comp_obj, "create"):
                    wrap_function_wrapper(
                        comp_obj,
                        "create",
                        _wrap_async(
                            next(
                                r
                                for r in ANTHROPIC_METHODS
                                if r.method == "create" and r.module == "anthropic.AsyncClient"
                            ),
                            self.initialize,
                        ),
                    )
                if hasattr(comp_obj, "stream"):
                    wrap_function_wrapper(
                        comp_obj,
                        "stream",
                        _wrap_async(
                            next(
                                r
                                for r in ANTHROPIC_METHODS
                                if r.method == "stream" and r.module == "anthropic.AsyncClient"
                            ),
                            self.initialize,
                        ),
                    )

            # Wrap messages methods
            msg_obj = getattr(instance, "messages", None)
            if msg_obj is not None:
                logger.debug("Found async messages object, wrapping methods")
                if hasattr(msg_obj, "create"):
                    logger.debug("Wrapping async messages.create method")
                    wrap_function_wrapper(
                        msg_obj,
                        "create",
                        _wrap_async(
                            next(
                                r
                                for r in ANTHROPIC_METHODS
                                if r.method == "create"
                                and r.module == "anthropic.AsyncClient"
                                and r.object == "messages"
                            ),
                            self.initialize,
                        ),
                    )

        anthropic.AsyncClient.__init__ = new_async_init

        setattr(anthropic, "langfuse_public_key", None)
        setattr(anthropic, "langfuse_secret_key", None)
        setattr(anthropic, "langfuse_host", None)
        setattr(anthropic, "langfuse_debug", None)
        setattr(anthropic, "langfuse_enabled", True)
        setattr(anthropic, "langfuse_sample_rate", None)
        setattr(anthropic, "langfuse_auth_check", self.langfuse_auth_check)
        setattr(anthropic, "flush_langfuse", self.flush)


modifier = AnthropicLangfuse()
modifier.register_tracing()


# DEPRECATED: Use `anthropic.langfuse_auth_check()` instead
def auth_check():
    if modifier._langfuse is None:
        modifier.initialize()
    return modifier._langfuse.auth_check()


# Rename Client to Anthropic and AsyncClient to AsyncAnthropic for better clarity
Anthropic = Client
AsyncAnthropic = AsyncClient
