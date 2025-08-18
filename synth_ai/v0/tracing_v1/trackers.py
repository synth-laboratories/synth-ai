import asyncio
import contextvars
from typing import Any, Literal

from pydantic import BaseModel

from .config import VALID_TYPES, Message, ModelParams
from .local import _local

# Existing SynthTrackerSync and SynthTrackerAsync classes...


class SynthTrackerSync:
    """Tracker for synchronous functions.

    Purpose is to annotate the inside of your sync functions to track intermediate values.
    Decorator @trace_event_sync is used to annotate the functions and track the inputs and outputs.
    This tracker is instead used to access the data inside of decorated functions.
    """

    _local = _local

    @classmethod
    def initialize(cls) -> None:
        cls._local.initialized = True
        cls._local.inputs = []
        cls._local.outputs = []

    @classmethod
    def track_lm(
        cls,
        messages: list[dict[str, str]],
        model_name: str,
        model_params: dict[str, str | int | float] | None = None,
        finetune: bool = False,
    ):
        # print("Tracking LM call in sync context - ",messages)  # Added logging
        if getattr(cls._local, "initialized", False):
            cls._local.inputs.append(
                {
                    "origin": "agent",
                    "messages": messages,
                    "model_name": model_name,
                    "model_params": model_params,
                    "finetune": finetune,
                }
            )
        else:
            pass

    @classmethod
    def track_state(
        cls,
        variable_name: str,
        variable_value: BaseModel | str | dict | int | float | bool | list | None,
        origin: Literal["agent", "environment"],
        annotation: str | None = None,
    ):
        # Skip if value is not a trackable type instead of raising error
        if not isinstance(variable_value, VALID_TYPES):
            return

        if getattr(cls._local, "initialized", False):
            if isinstance(variable_value, BaseModel):
                variable_value = variable_value.model_dump()
            cls._local.outputs.append(
                {
                    "origin": origin,
                    "variable_name": variable_name,
                    "variable_value": variable_value,
                    "annotation": annotation,
                }
            )
            # logger.debug(f"Tracked state: {variable_name}")
        else:
            pass
            # raise RuntimeError(
            #     "Trace not initialized. Use within a function decorated with @trace_event_sync."
            # )

    @classmethod
    def get_traced_data(cls) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return getattr(cls._local, "inputs", []), getattr(cls._local, "outputs", [])

    @classmethod
    def finalize(cls) -> None:
        # Clean up the thread-local storage
        cls._local.initialized = False
        cls._local.inputs = []
        cls._local.outputs = []
        # logger.debug("Finalized trace data")

    @classmethod
    def track_lm_output(
        cls,
        messages: list[dict[str, str]],
        model_name: str,
        finetune: bool = False,
    ):
        """
        Tracks 'messages' as if they were output from the LLM.
        """
        if getattr(cls._local, "initialized", False):
            cls._local.outputs.append(
                {
                    "origin": "agent",
                    "messages": messages,
                    "model_name": model_name,
                    "finetune": finetune,
                }
            )
        else:
            pass


# Context variables for asynchronous tracing
trace_inputs_var = contextvars.ContextVar("trace_inputs", default=None)
trace_outputs_var = contextvars.ContextVar("trace_outputs", default=None)
trace_initialized_var = contextvars.ContextVar("trace_initialized", default=False)


class SynthTrackerAsync:
    """Tracker for synchronous functions.

    Purpose is to annotate the inside of your sync functions to track intermediate values.
    Decorator @trace_event_sync is used to annotate the functions and track the inputs and outputs.
    This tracker is instead used to access the data inside of decorated functions.
    """

    @classmethod
    def initialize(cls) -> None:
        trace_initialized_var.set(True)
        trace_inputs_var.set([])  # List of tuples: (origin, var)
        trace_outputs_var.set([])  # List of tuples: (origin, var)
        # logger.debug("AsyncTrace initialized")

    @classmethod
    def track_lm(
        cls,
        messages: list[dict[str, str]],
        model_name: str,
        model_params: dict[str, str | int | float] | None = None,
        finetune: bool = False,
    ):
        # print("Tracking LM call in async context")  # Added logging
        if trace_initialized_var.get():
            trace_inputs = trace_inputs_var.get()
            trace_inputs.append(
                {
                    "origin": "agent",
                    "messages": messages,
                    "model_name": model_name,
                    "model_params": model_params,
                    "finetune": finetune,
                }
            )
            trace_inputs_var.set(trace_inputs)
        else:
            pass
            # raise RuntimeError(
            #     "Trace not initialized. Use within a function decorated with @trace_event_async."
            # )

    @classmethod
    def track_state(
        cls,
        variable_name: str,
        variable_value: BaseModel | str | dict | int | float | bool | list | None,
        origin: Literal["agent", "environment"],
        annotation: str | None = None,
        io_type: Literal["input", "output"] = "output",
    ):
        # Skip if value is not a trackable type instead of raising error
        if not isinstance(variable_value, VALID_TYPES):
            return

        if trace_initialized_var.get():
            if isinstance(variable_value, BaseModel):
                variable_value = variable_value.model_dump()
            trace_outputs = trace_outputs_var.get()
            if io_type == "input":
                trace_inputs = trace_inputs_var.get()
                trace_inputs.append(
                    {
                        "origin": origin,
                        "variable_name": variable_name,
                        "variable_value": variable_value,
                        "annotation": annotation,
                    }
                )
                trace_inputs_var.set(trace_inputs)
            else:
                trace_outputs.append(
                    {
                        "origin": origin,
                        "variable_name": variable_name,
                        "variable_value": variable_value,
                        "annotation": annotation,
                    }
                )
                trace_outputs_var.set(trace_outputs)
            # logger.debug(f"Tracked state: {variable_name}")
        else:
            pass
            # raise RuntimeError(
            #     "Trace not initialized. Use within a function decorated with @trace_event_async."
            # )

    @classmethod
    def get_traced_data(cls) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        traced_inputs = trace_inputs_var.get()
        traced_outputs = trace_outputs_var.get()
        return traced_inputs, traced_outputs

    @classmethod
    def finalize(cls) -> None:
        trace_initialized_var.set(False)
        trace_inputs_var.set([])
        trace_outputs_var.set([])
        # logger.debug("Finalized async trace data")

    @classmethod
    def track_lm_output(
        cls,
        messages: list[dict[str, str]],
        model_name: str,
        finetune: bool = False,
    ):
        """
        Tracks 'messages' as if they were output from the LLM.
        """
        if trace_initialized_var.get():
            trace_outputs = trace_outputs_var.get()
            trace_outputs.append(
                {
                    "origin": "agent",
                    "messages": messages,
                    "model_name": model_name,
                    "finetune": finetune,
                }
            )
            trace_outputs_var.set(trace_outputs)
        else:
            pass


# Make traces available globally
synth_tracker_sync = SynthTrackerSync
synth_tracker_async = SynthTrackerAsync


class SynthTracker:
    """Tracker for synchronous and asynchronous functions. Intelligently chooses between sync and async trackers.

    Purpose is to annotate the inside of your sync and async functions to track intermediate values.
    Decorators @trace_event_sync and @trace_event_async are used to annotate the functions and track the inputs and outputs.
    This tracker is instead used to access the data inside of decorated functions.
    """

    @classmethod
    def is_called_by_async(cls) -> bool:
        try:
            asyncio.get_running_loop()  # Attempt to get the running event loop
            # Debug logging disabled: print("DEBUG: Running in async context")
            return True  # If successful, we are in an async context
        except RuntimeError:
            # Debug logging disabled: print("DEBUG: Running in sync context")
            return False  # If there's no running event loop, we are in a sync context

    @classmethod
    def track_lm(
        cls,
        messages: list[dict[str, str]],
        model_name: str,
        model_params: dict[str, str | int | float] | None = None,
        finetune: bool = False,
    ):
        # Debug logging disabled: print("DEBUG: Tracking LM call")
        """
        Track a language model interaction within the current trace.
        Automatically detects whether to use sync or async tracking.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries containing the conversation.
                Each message should have:
                - 'role': str - The role of the speaker (e.g., "user", "assistant", "system")
                - 'content': str - The content of the message

            model_name (str): Name of the language model being used
                Examples: "gpt-4o-mini", "gpt-4o-mini", "claude-3-opus-20240229"

            finetune (bool, optional): Whether this interaction is part of a fine-tuning process.
                Defaults to False.

        Raises:
            RuntimeError: If called outside a traced context (use with @trace_event_sync
                or @trace_event_async decorator)
            TypeError: If messages or model_name are not of the correct type

        Example:
            ```python
            @trace_event_sync(origin="agent", event_type="chat")
            def process_chat(self, user_input: str):
                messages = [
                    {"role": "user", "content": user_input}
                ]
                SynthTracker.track_lm(
                    messages=messages,
                    model_name="gpt-4o-mini"
                )
            ```
        """
        if cls.is_called_by_async() and trace_initialized_var.get():
            # print("DEBUG: Tracking LM call in async context")  # Added logging
            synth_tracker_async.track_lm(
                messages,
                model_name,
                model_params,
                finetune,
            )
        elif getattr(synth_tracker_sync._local, "initialized", False):
            # print("DEBUG: Tracking LM call in sync context")  # Added logging
            synth_tracker_sync.track_lm(
                messages,
                model_name,
                model_params,
                finetune,
            )
        else:
            # Debug logging disabled: print("DEBUG: Skipping LM tracking - not initialized")  # Added logging
            pass

    @classmethod
    def track_state(
        cls,
        variable_name: str,
        variable_value: BaseModel | str | dict | int | float | bool | list | None,
        origin: Literal["agent", "environment"],
        annotation: str | None = None,
    ):
        """
        Track a state change or variable value within the current trace.
        Automatically detects whether to use sync or async tracking.

        Args:
            variable_name (str): Name of the variable or state being tracked

            variable_value (Union[BaseModel, str, dict, int, float, bool, list, None]):
                Value to track. Must be one of the supported types:
                - BaseModel (Pydantic models)
                - Basic Python types (str, dict, int, float, bool, list)
                - None

            origin (Literal["agent", "environment"]): Source of the state change
                - "agent": Changes from the AI/agent system
                - "environment": Changes from external sources/environment

            annotation (Optional[str], optional): Additional notes about the state change.
                Defaults to None.

        Raises:
            RuntimeError: If called outside a traced context (use with @trace_event_sync
                or @trace_event_async decorator)
            TypeError: If variable_value is not one of the supported types
            ValueError: If origin is not "agent" or "environment"

        Example:
            ```python
            @trace_event_sync(origin="agent", event_type="process")
            def update_state(self, new_value: dict):
                SynthTracker.track_state(
                    variable_name="system_state",
                    variable_value=new_value,
                    origin="agent",
                    annotation="Updated after processing"
                )
            ```
        """
        if cls.is_called_by_async() and trace_initialized_var.get():
            # logger.debug("Using async tracker to track state")
            synth_tracker_async.track_state(variable_name, variable_value, origin, annotation)
        elif getattr(synth_tracker_sync._local, "initialized", False):
            # logger.debug("Using sync tracker to track state")
            synth_tracker_sync.track_state(variable_name, variable_value, origin, annotation)
        else:
            # raise RuntimeError("Trace not initialized in track_state.")
            pass

    @classmethod
    def get_traced_data(
        cls,
        async_sync: Literal[
            "async", "sync", ""
        ] = "",  # Force only async or sync data to be returned
    ) -> tuple[list, list]:
        traced_inputs, traced_outputs = [], []
        # Debug logging disabled: print(
        #     f"\nDEBUG: Getting traced data with async_sync='{async_sync}'"
        # )  # Added logging

        if async_sync in ["async", ""]:
            # Debug logging disabled: print("DEBUG: Retrieving async traced data")
            traced_inputs_async, traced_outputs_async = synth_tracker_async.get_traced_data()
            # Debug logging disabled: print(
            #     f"DEBUG: Found {len(traced_inputs_async)} async inputs and {len(traced_outputs_async)} async outputs"
            # )  # Added logging
            traced_inputs.extend(traced_inputs_async)
            traced_outputs.extend(traced_outputs_async)

        if async_sync in ["sync", ""]:
            # Debug logging disabled: print("DEBUG: Retrieving sync traced data")
            traced_inputs_sync, traced_outputs_sync = synth_tracker_sync.get_traced_data()
            # Debug logging disabled: print(
            #     f"DEBUG: Found {len(traced_inputs_sync)} sync inputs and {len(traced_outputs_sync)} sync outputs"
            # )  # Added logging
            traced_inputs.extend(traced_inputs_sync)
            traced_outputs.extend(traced_outputs_sync)

        # Debug logging disabled: print(
        #     f"DEBUG: Final combined data: {len(traced_inputs)} inputs and {len(traced_outputs)} outputs\n"
        # )  # Added logging
        return traced_inputs, traced_outputs

    @classmethod
    def track_lm_output(
        cls,
        messages: list[dict[str, str]],
        model_name: str,
        finetune: bool = False,
    ):
        """
        Tracks 'messages' as if they were output from the LLM.
        Automatically detects whether to use sync or async tracking.
        """
        if cls.is_called_by_async() and trace_initialized_var.get():
            # Debug logging disabled: print("DEBUG: Tracking LM output in async context")
            synth_tracker_async.track_lm_output(
                messages=messages,
                model_name=model_name,
                finetune=finetune,
            )
        elif getattr(synth_tracker_sync._local, "initialized", False):
            # Debug logging disabled: print("DEBUG: Tracking LM output in sync context")
            synth_tracker_sync.track_lm_output(
                messages=messages,
                model_name=model_name,
                finetune=finetune,
            )
        else:
            # Debug logging disabled: print("DEBUG: Skipping LM output tracking - not initialized")
            pass


def track_messages_sync(
    input_messages: list[Message],
    output_messages: list[Message],
    model_name: str,
    model_params: ModelParams | None = None,
    finetune: bool = False,
) -> None:
    """Track both input and output messages in a conversation synchronously.

    Args:
        input_messages: List of input messages (e.g., user messages)
        output_messages: List of output messages (e.g., assistant responses)
        model_name: Name of the language model being used
        model_params: Optional parameters used for the model
        finetune: Whether this conversation should be used for fine-tuning
    """
    # Track input messages
    synth_tracker_sync.track_lm(
        messages=input_messages,
        model_name=model_name,
        model_params=model_params,
        finetune=finetune,
    )

    # Track output messages
    synth_tracker_sync.track_lm_output(
        messages=output_messages,
        model_name=model_name,
        finetune=finetune,
    )


async def track_messages_async(
    input_messages: list[Message],
    output_messages: list[Message],
    model_name: str,
    model_params: ModelParams | None = None,
    finetune: bool = False,
) -> None:
    """Track both input and output messages in a conversation asynchronously.

    Args:
        input_messages: List of input messages (e.g., user messages)
        output_messages: List of output messages (e.g., assistant responses)
        model_name: Name of the language model being used
        model_params: Optional parameters used for the model
        finetune: Whether this conversation should be used for fine-tuning
    """
    # Track input messages
    synth_tracker_async.track_lm(
        messages=input_messages,
        model_name=model_name,
        model_params=model_params,
        finetune=finetune,
    )

    # Track output messages
    synth_tracker_async.track_lm_output(
        messages=output_messages,
        model_name=model_name,
        finetune=finetune,
    )
