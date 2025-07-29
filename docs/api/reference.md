# API Reference

*Generated on 2025-07-29 14:11:10*

This document contains the API reference for the core `synth_ai` modules.

## Installation

```bash
pip install synth-ai
```

## Quick Start

```python
from synth_ai.lm.core.main_v2 import LM

# Initialize LM
lm = LM(model_name="gpt-4o-mini")

# Generate response
response = lm.respond("Hello, world!")
print(response.raw_response)
```

## synth_ai.lm.core.main_v2

Enhanced LM class with native v2 tracing support.

This module extends the LM class to support v2 tracing through decorators,
enabling clean integration without modifying provider wrappers.

### Classes

### LM

Enhanced Language Model interface with native v2 tracing support.

Args:
model_name: The name of the model to use.
formatting_model_name: The model to use for formatting structured outputs.
temperature: The temperature setting for the model (0.0 to 1.0).
max_retries: Number of retries for API calls ("None", "Few", or "Many").
structured_output_mode: Mode for structured outputs ("stringified_json" or "forced_json").
synth_logging: Whether to enable Synth v1 logging (for backwards compatibility).
provider: Optional provider override.
session_tracer: Optional v2 SessionTracer instance for tracing.
system_id: Optional system ID for v2 tracing (defaults to "lm_{model_name}").
enable_v2_tracing: Whether to enable v2 tracing (defaults to True).

#### Methods

**__init__**(self, model_name: str, formatting_model_name: str, temperature: float, max_retries: Literal['None', 'Few', 'Many'] = 'Few', structured_output_mode: Literal['stringified_json', 'forced_json'] = 'stringified_json', synth_logging: bool = True, provider: Union[Literal['openai', 'anthropic', 'groq', 'gemini', 'deepseek', 'grok', 'mistral', 'openrouter', 'together', 'synth'], str, NoneType] = None, session_tracer: Optional[synth_ai.tracing_v2.session_tracer.SessionTracer] = None, system_id: Optional[str] = None, enable_v2_tracing: bool = True)

**respond_async**(self, system_message: Optional[str] = None, user_message: Optional[str] = None, messages: Optional[List[Dict]] = None, images_as_bytes: List[Any] = [], response_model: Optional[pydantic.main.BaseModel] = None, use_ephemeral_cache_only: bool = False, tools: Optional[List[synth_ai.lm.tools.base.BaseTool]] = None, reasoning_effort: str = 'low', turn_number: Optional[int] = None)

**respond_sync**(self, system_message: Optional[str] = None, user_message: Optional[str] = None, messages: Optional[List[Dict]] = None, images_as_bytes: List[Any] = [], response_model: Optional[pydantic.main.BaseModel] = None, use_ephemeral_cache_only: bool = False, tools: Optional[List[synth_ai.lm.tools.base.BaseTool]] = None, reasoning_effort: str = 'low', turn_number: Optional[int] = None)

#### Properties

**__weakref__**

    list of weak references to the object


### LMTracingContext

Context manager for LM with v2 tracing.

#### Methods

**__enter__**(self)

**__exit__**(self, *args)

**__init__**(self, lm: synth_ai.lm.core.main_v2.LM, session_tracer: synth_ai.tracing_v2.session_tracer.SessionTracer)

#### Properties

**__weakref__**

    list of weak references to the object


### Functions

### build_messages(sys_msg: str, user_msg: str, images_bytes: List = [], model_name: Optional[str] = None) -> List[Dict]


### extract_lm_attributes(args, kwargs, result=None, error=None) -> Dict[str, Any]

Custom attribute extraction for LM calls following OTel GenAI conventions.



## synth_ai.tracing_v2.session_tracer

SessionTracer for capturing LLM calls from langfuse OTEL and converting to CAIS events.

This module provides utilities to:
1. Capture langfuse OTEL spans during LLM calls
2. Convert them to CAISEvent objects
3. Manage SessionTrace objects for rollouts
4. Save traces to disk for analysis

### Classes

### EnvironmentEvent

Environment system event (e.g., state transition, reward signal).

#### Methods

**__eq__**(self, other)

**__init__**(self, time_record: Optional[synth_ai.tracing_v2.session_tracer.TimeRecord] = None, system_instance_id: str = 'environment', system_state_before: Optional[Dict[str, Any]] = None, system_state_after: Optional[Dict[str, Any]] = None, reward: Optional[float] = None, terminated: Optional[bool] = None, observation_diff: Optional[Dict[str, Any]] = None, metadata: Dict[str, Any] = <factory>, event_metadata: List[Any] = <factory>) -> None

**__repr__**(self)

**to_dict**(self) -> Dict[str, Any]


### LMCAISEvent

Extended CAIS event for LM-specific interactions with additional fields.

#### Methods

**__eq__**(self, other)

**__init__**(self, time_record: synth_ai.tracing_v2.abstractions.TimeRecord = <factory>, system_instance_id: Optional[Any] = None, system_state_before: Optional[Any] = None, system_state_after: Optional[Any] = None, llm_call_records: List[Any] = <factory>, metadata: Optional[Dict[str, Any]] = <factory>, event_metadata: List[synth_ai.tracing_v2.abstractions.EventMetadata] = <factory>, span_id: Optional[str] = None, trace_id: Optional[str] = None, model_name: Optional[str] = None, prompt_tokens: Optional[int] = None, completion_tokens: Optional[int] = None, total_tokens: Optional[int] = None, cost: Optional[float] = None, latency_ms: Optional[float] = None) -> None

**__repr__**(self)

**to_dict**(self) -> Dict[str, Any]


### RuntimeEvent

Runtime system event (e.g., action execution, message routing).

#### Methods

**__eq__**(self, other)

**__init__**(self, time_record: Optional[synth_ai.tracing_v2.session_tracer.TimeRecord] = None, system_instance_id: str = 'runtime', system_state_before: Optional[Dict[str, Any]] = None, system_state_after: Optional[Dict[str, Any]] = None, actions: Optional[List[Any]] = None, metadata: Dict[str, Any] = <factory>, event_metadata: List[Any] = <factory>) -> None

**__repr__**(self)

**to_dict**(self) -> Dict[str, Any]


### SessionEvent

Base class for session events (system state changes).

#### Methods

**__eq__**(self, other)

**__init__**(self, time_record: Optional[synth_ai.tracing_v2.session_tracer.TimeRecord] = None) -> None

**__repr__**(self)

**to_dict**(self) -> Dict[str, Any]

#### Properties

**__weakref__**

    list of weak references to the object


### SessionEventMessage

Message entering or leaving a Markov blanket.

#### Methods

**__eq__**(self, other)

**__init__**(self, content: Any, timestamp: str = <factory>, message_type: str = 'unknown', time_record: Optional[synth_ai.tracing_v2.session_tracer.TimeRecord] = None) -> None

**__repr__**(self)

**to_dict**(self) -> Dict[str, Any]

#### Properties

**__weakref__**

    list of weak references to the object


### SessionMetadum

Base class for session metadata.

#### Methods

**__eq__**(self, other)

**__init__**(self, metadata_type: str, data: Dict[str, Any] = <factory>, timestamp: str = <factory>) -> None

**__repr__**(self)

**to_dict**(self) -> Dict[str, Any]

#### Properties

**__weakref__**

    list of weak references to the object


### SessionTimeStep

A single timestep in a session containing events and messages.

#### Methods

**__eq__**(self, other)

**__init__**(self, step_id: str, events: List[synth_ai.tracing_v2.session_tracer.SessionEvent] = <factory>, step_messages: List[synth_ai.tracing_v2.session_tracer.SessionEventMessage] = <factory>, timestamp: str = <factory>, step_metadata: Dict[str, Any] = <factory>) -> None

**__repr__**(self)

**add_event**(self, event: synth_ai.tracing_v2.session_tracer.SessionEvent)

    Add an event to this timestep.

**add_message**(self, message: synth_ai.tracing_v2.session_tracer.SessionEventMessage)

    Add a message to this timestep.

**to_dict**(self) -> Dict[str, Any]

#### Properties

**__weakref__**

    list of weak references to the object


### SessionTrace

Complete trace of a session with timesteps and metadata.

#### Methods

**__eq__**(self, other)

**__init__**(self, session_id: str, session_time_steps: List[synth_ai.tracing_v2.session_tracer.SessionTimeStep] = <factory>, session_metadata: List[synth_ai.tracing_v2.session_tracer.SessionMetadum] = <factory>, message_history: List[synth_ai.tracing_v2.session_tracer.SessionEventMessage] = <factory>, event_history: List[synth_ai.tracing_v2.session_tracer.SessionEvent] = <factory>, created_at: str = <factory>) -> None

**__repr__**(self)

**add_event**(self, event: synth_ai.tracing_v2.session_tracer.SessionEvent)

    Add an event to the global event history.

**add_message**(self, message: synth_ai.tracing_v2.session_tracer.SessionEventMessage)

    Add a message to the global message history.

**add_metadata**(self, metadata: synth_ai.tracing_v2.session_tracer.SessionMetadum)

    Add metadata to the session.

**add_timestep**(self, timestep: synth_ai.tracing_v2.session_tracer.SessionTimeStep)

    Add a timestep to the session.

**to_dict**(self) -> Dict[str, Any]

    Convert to dictionary for serialization.

#### Properties

**__weakref__**

    list of weak references to the object


### SessionTracer

Main class for tracing LLM interactions in sessions.

#### Methods

**__enter__**(self)

    Context manager entry.

**__exit__**(self, exc_type, exc_val, exc_tb)

    Context manager exit.

**__init__**(self, traces_dir: str = 'traces', hooks: Optional[List[synth_ai.tracing_v2.hooks.TraceHook]] = None, duckdb_path: Optional[str] = None, experiment_id: Optional[str] = None)

**_extract_completion_from_attrs**(self, attrs: Dict) -> Optional[Any]

    Extract completion content from OTEL attributes.

**_extract_prompt_from_attrs**(self, attrs: Dict) -> Optional[Any]

    Extract prompt content from OTEL attributes.

**_run_hooks_on_event**(self, event: synth_ai.tracing_v2.session_tracer.SessionEvent)

    Run all registered hooks on an event and collect metadata.

**add_session_metadata**(self, metadata_type: str, data: Dict[str, Any])

    Add metadata to the current session.

**advance_turn**(self)

    Advance to the next turn.

**capture_llm_call**(self, system_id: str = 'llm_agent', system_state_before: Optional[Dict] = None, system_state_after: Optional[Dict] = None) -> Optional[synth_ai.tracing_v2.session_tracer.LMCAISEvent]

    Capture the current langfuse OTEL span and convert to CAISEvent.
    Records prompt message, then CAISEvent, then completion message in temporal order.

**capture_llm_call_from_generation**(self, generation, system_id: str = 'llm_agent', system_state_before: Optional[Dict] = None, system_state_after: Optional[Dict] = None, messages: Optional[List] = None, response=None) -> Optional[synth_ai.tracing_v2.session_tracer.LMCAISEvent]

    Capture LLM call from langfuse generation object.
    Records prompt message, then completion message, then CAISEvent in temporal order.

**close**(self)

    Close resources including DuckDB connection.

**end_session**(self, save: bool = True, upload_to_db: bool = True) -> Optional[pathlib.Path]

    End the current session and optionally save it.

**record_event**(self, event: synth_ai.tracing_v2.session_tracer.SessionEvent) -> synth_ai.tracing_v2.session_tracer.SessionEvent

    Record an event in both the current timestep and global history.

**record_message**(self, message: synth_ai.tracing_v2.session_tracer.SessionEventMessage) -> synth_ai.tracing_v2.session_tracer.SessionEventMessage

    Record a message in both the current timestep and global history.

**save_session**(self, filename: Optional[str] = None) -> pathlib.Path

    Save the current session trace to disk.

**start_session**(self, session_id: Optional[str] = None) -> synth_ai.tracing_v2.session_tracer.SessionTrace

    Start a new session trace.

**start_timestep**(self, step_id: Optional[str] = None, **metadata) -> synth_ai.tracing_v2.session_tracer.SessionTimeStep

    Start a new timestep in the current session.

#### Properties

**__weakref__**

    list of weak references to the object


### TimeRecord

Time record for events and messages.

#### Methods

**__eq__**(self, other)

**__init__**(self, event_time: str, message_time: int) -> None

**__repr__**(self)

**to_dict**(self) -> Dict[str, Any]

#### Properties

**__weakref__**

    list of weak references to the object


### Functions

### create_session_tracer(traces_dir: str = 'traces', hooks: Optional[List[synth_ai.tracing_v2.hooks.TraceHook]] = None) -> synth_ai.tracing_v2.session_tracer.SessionTracer

Create a new session tracer.


### extract_model_info_from_attrs(attrs: Dict[str, Any]) -> Dict[str, Optional[Any]]

Generic helper to extract model/token info for various providers.


### guess_provider_from_attrs(attrs: Dict[str, Any]) -> Optional[str]

Guess the provider based on known attribute keys - supports OpenAI, Azure OpenAI, and Anthropic.



## synth_ai.tracing_v2.duckdb.manager

DuckDB integration for tracing_v2 system.
Provides efficient storage and analytics for trace data.

### Classes

### DuckDBTraceManager

Manages DuckDB storage for trace data.

#### Methods

**__enter__**(self)

    Context manager entry.

**__exit__**(self, exc_type, exc_val, exc_tb)

    Context manager exit.

**__init__**(self, db_path: Optional[str] = None, config: Optional[synth_ai.tracing_v2.storage.config.DuckDBConfig] = None, skip_schema_init: bool = False)

    Initialize DuckDB manager with database path or config.

    Args:
    db_path: Path to database file (overrides config)
    config: DuckDB configuration object
    skip_schema_init: Skip schema initialization (for concurrent connections)

**_connect**(self)

    Establish connection to DuckDB using singleton pattern.

**_create_analytics_views**(self)

    Create materialized views for analytics.

**_insert_event**(self, session_id: str, event: Union[synth_ai.tracing_v2.abstractions.CAISEvent, synth_ai.tracing_v2.session_tracer.LMCAISEvent, synth_ai.tracing_v2.session_tracer.EnvironmentEvent, synth_ai.tracing_v2.session_tracer.RuntimeEvent], timestep_id_map: Dict[str, int])

    Insert an event into the database.

**_insert_message**(self, session_id: str, message: synth_ai.tracing_v2.session_tracer.SessionEventMessage, timestep_id_map: Dict[str, int])

    Insert a message into the database.

**_insert_session_trace_bulk**(self, trace)

    Insert a session trace optimized for bulk operations (with locking).

**_insert_session_trace_bulk_internal**(self, trace)

    Insert a session trace optimized for bulk operations.

**_insert_session_trace_internal**(self, trace)

    Internal implementation of session trace insertion.

**_insert_timestep**(self, session_id: str, timestep: synth_ai.tracing_v2.session_tracer.SessionTimeStep, index: int) -> int

    Insert a timestep and return its ID.

**_prepare_event_data**(self, session_id: str, event: Union[synth_ai.tracing_v2.abstractions.CAISEvent, synth_ai.tracing_v2.session_tracer.LMCAISEvent, synth_ai.tracing_v2.session_tracer.EnvironmentEvent, synth_ai.tracing_v2.session_tracer.RuntimeEvent], timestep_id_map: Dict[str, int]) -> Optional[List[Any]]

    Prepare event data for bulk insert.

**_serialize_metadata**(self, metadata: synth_ai.tracing_v2.abstractions.SessionMetadum) -> Dict[str, Any]

    Serialize session metadata to dictionary.

**batch_upload**(self, traces: List[synth_ai.tracing_v2.session_tracer.SessionTrace], batch_size: Optional[int] = None)

    Upload multiple traces efficiently in batches.

**close**(self)

    Close DuckDB connection using reference counting.

**create_experiment**(self, experiment_id: str, name: str, description: str = '', system_versions: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]

    Create a new experiment with optional system versions.

**create_system**(self, system_id: str, name: str, description: str = '') -> Dict[str, Any]

    Create a new system.

**create_system_version**(self, version_id: str, system_id: str, branch: str, commit: str, description: str = '') -> Dict[str, Any]

    Create a new system version.

**export_to_parquet**(self, output_dir: str)

    Export all tables to Parquet format.

**get_expensive_calls**(self, cost_threshold: float) -> pandas.core.frame.DataFrame

    Get expensive LLM calls above threshold.

**get_experiment_sessions**(self, experiment_id: str) -> pandas.core.frame.DataFrame

    Get all sessions for an experiment.

**get_experiments_by_system_version**(self, system_version_id: str) -> pandas.core.frame.DataFrame

    Get all experiments using a specific system version.

**get_model_usage**(self, start_date: Optional[datetime.datetime] = None, end_date: Optional[datetime.datetime] = None) -> pandas.core.frame.DataFrame

    Get model usage statistics.

**get_session_summary**(self, session_id: Optional[str] = None) -> pandas.core.frame.DataFrame

    Get session summary information.

**init_schema**(self)

    Initialize database schema.

**insert_session_trace**(self, trace)

    Insert a complete session trace into DuckDB.

**link_session_to_experiment**(self, session_id: str, experiment_id: str)

    Link a session to an experiment.

**query_traces**(self, query: str, params: Optional[List[Any]] = None) -> pandas.core.frame.DataFrame

    Execute analytical query and return results as DataFrame.


### Functions

### convert_datetime_for_duckdb(dt: Union[datetime.datetime, str, float, NoneType]) -> Optional[str]

Convert datetime to DuckDB-compatible timestamp string.


### safe_json_serialize(obj: Any) -> str

Safely serialize objects to JSON, handling datetime objects.



## synth_ai.tracing_v2.decorators

V3 Decorator Implementation for Synth-AI Tracing (Improved)

This module provides decorators that emit OpenTelemetry spans while maintaining
compatibility with v2 SessionTracer patterns. Designed for < 5% overhead and
proper context propagation across async/thread boundaries.

Key improvements:
- Relies on OTel SDK sampling instead of double sampling
- Uses OTel events for prompt/response bodies
- Adds PII masking capability
- Includes cost tracking for AI calls
- Better resource attributes
- Early exit when tracing disabled

### Classes

### ContextPropagatingExecutor

ThreadPoolExecutor that automatically propagates OTel context.

#### Methods

**submit**(self, fn, *args, **kwargs)

    Submit with context propagation.


### Functions

### add_otel_events(span: opentelemetry.trace.span.Span, messages: Optional[List[Dict[str, str]]] = None, completion: Optional[str] = None) -> None

Add OTel events for prompts and completions per GenAI spec.


### calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]

Calculate cost in USD for AI model usage.


### capture_state(args: tuple, kwargs: dict, attrs_fn: Optional[Callable] = None, result: Any = None, error: Optional[Exception] = None, is_before: bool = True) -> Dict[str, Any]

Capture state for tracing, with optional custom attribute extraction.


### create_v2_event(session_tracer: synth_ai.tracing_v2.session_tracer.SessionTracer, event_type: type, before_state: Dict[str, Any], after_state: Dict[str, Any], duration_ns: int, metadata: Optional[Dict[str, Any]] = None) -> None

Create a v2 event for compatibility.


### extract_ai_attributes(args, kwargs, result=None, error=None) -> Dict[str, Any]

Extract attributes for AI/LLM calls following OTel GenAI conventions.


### extract_env_attributes(args, kwargs, result=None, error=None) -> Dict[str, Any]

Extract attributes for environment operations.


### extract_runtime_attributes(args, kwargs, result=None, error=None) -> Dict[str, Any]

Extract attributes for runtime operations.


### get_active_session_tracer() -> Optional[synth_ai.tracing_v2.session_tracer.SessionTracer]

Get the active v2 session tracer from context.


### get_system_id() -> Optional[str]

Get the system ID from context.


### get_turn_number() -> Optional[int]

Get the current turn number from context.


### mask_pii(text: str, patterns: Optional[List[Tuple[str, str]]] = None) -> str

Mask PII in text using regex patterns.


### set_active_session_tracer(tracer: synth_ai.tracing_v2.session_tracer.SessionTracer) -> None

Set the active v2 session tracer for the current context.


### set_system_id(system_id: str) -> None

Set the system ID for the current context.


### set_turn_number(turn: int) -> None

Set the current turn number for the context.


### setup_flush_handlers()

Ensure traces are flushed on exit.


### setup_otel_tracer()

Set up OTel tracer with proper resource attributes and batching.


### trace_span(name: Union[str, Callable[..., str]], kind: opentelemetry.trace.SpanKind = <SpanKind.INTERNAL: 0>, attrs_fn: Optional[Callable[[tuple, dict, Any, Optional[Exception]], Dict[str, Any]]] = None, event_type: Optional[type] = None, v2_only: bool = False, otel_only: bool = False) -> Callable[[~F], ~F]

Unified decorator for OTel spans + v2 events.

Args:
name: Span name or callable to generate it
kind: OTel span kind
attrs_fn: Function to extract attributes
event_type: V2 event class (CAISEvent, etc.)
v2_only: Only emit v2 events (no OTel)
otel_only: Only emit OTel spans (no v2)


### truncate_and_mask(data: Any, max_bytes: Optional[int] = None) -> Any

Truncate large payloads and mask PII.



## synth_ai.environments.environment.core

### Classes

### Environment

Base class for all environments in the Synth AI framework.

This class provides the fundamental structure for all environment types,
including a name attribute system that supports both automatic naming
based on the class name and manual name assignment.

The Environment class serves as the foundation for more specialized
environment types like StatefulEnvironment, providing common functionality
and ensuring consistent interfaces across all environment implementations.

Attributes:
_default_name: Class-level default name derived from the class name
_name: Instance-level name override (optional)

Example:
>>> class MyCustomEnv(Environment):
...     pass
>>> env = MyCustomEnv()
>>> print(env.name)  # "MyCustomEnv"
>>> env.name = "custom_environment"
>>> print(env.name)  # "custom_environment"



## synth_ai.environments.environment.registry

Environment Registry Module

This module provides a global registry system for environment types in the Synth AI framework.
The registry allows environments to be registered by name and retrieved dynamically,
enabling flexible environment management and discovery.

The registry supports:
- Dynamic environment registration at runtime
- Type-safe environment retrieval
- Environment discovery and listing
- Centralized environment management

Example:
>>> from synth_ai.environments.environment.registry import register_environment
>>> from myproject.environments import MyCustomEnvironment

>>> # Register a custom environment
>>> register_environment("my_env", MyCustomEnvironment)

>>> # List available environments
>>> available_envs = list_supported_env_types()
>>> print("Available environments:", available_envs)

>>> # Get environment class for instantiation
>>> env_cls = get_environment_cls("my_env")
>>> env_instance = env_cls(task_config)

### Functions

### get_environment_cls(env_type: str) -> Type[synth_ai.environments.stateful.core.StatefulEnvironment]

Retrieve a registered environment class by name.

This function looks up an environment class in the global registry
and returns it for instantiation. This enables dynamic environment
creation based on string identifiers, which is useful for:
- Configuration-driven environment selection
- API endpoints that accept environment type parameters
- Plugin systems and modular architectures
- Testing frameworks that need to test multiple environment types

Args:
env_type: The name of the environment type to retrieve. Must match
a name that was previously registered using register_environment().

Returns:
Type[StatefulEnvironment]: The environment class that can be instantiated
with appropriate task parameters.

Raises:
ValueError: If env_type is not found in the registry. The error message
will include the invalid type name and suggest checking available types.

Example:
>>> # Retrieve and instantiate an environment
>>> env_cls = get_environment_cls("CartPole")
>>> environment = env_cls(task_instance)
>>>
>>> # Use in configuration-driven scenarios
>>> config = {"environment_type": "Sokoban", "difficulty": "easy"}
>>> env_cls = get_environment_cls(config["environment_type"])
>>> env = env_cls(create_task(config))

See Also:
list_supported_env_types(): Get all available environment type names
register_environment(): Register new environment types


### list_supported_env_types() -> List[str]

List all registered environment type names.

This function returns a list of all environment names that have been
registered in the global registry. It's useful for:
- Displaying available options to users
- Validating environment type parameters
- Building dynamic UIs or configuration tools
- Debugging and development

Returns:
List[str]: Sorted list of all registered environment type names.
Returns an empty list if no environments have been registered.

Example:
>>> # Check what environments are available
>>> available_envs = list_supported_env_types()
>>> print("Supported environments:")
>>> for env_type in available_envs:
...     print(f"  - {env_type}")

>>> # Validate user input
>>> user_choice = input("Choose environment: ")
>>> if user_choice not in list_supported_env_types():
...     print(f"Error: {user_choice} not available")

>>> # Build configuration options
>>> config_schema = {
...     "environment_type": {
...         "enum": list_supported_env_types(),
...         "description": "Type of environment to use"
...     }
... }

Note:
The returned list is sorted alphabetically for consistent ordering.
This function returns a copy of the environment names, so modifying
the returned list will not affect the registry.


### register_environment(name: str, cls: Type[synth_ai.environments.stateful.core.StatefulEnvironment]) -> None

Register an environment class under a unique name.

This function adds an environment class to the global registry, making it
available for dynamic instantiation by name. This is particularly useful
for building flexible systems where environment types are determined at
runtime or configured through external settings.

Args:
name: Unique identifier for the environment. This name will be used
to retrieve the environment class later. Names should be descriptive
and follow a consistent naming convention (e.g., "CartPole", "Sokoban").
cls: The environment class to register. Must be a subclass of
StatefulEnvironment and implement all required abstract methods.

Raises:
TypeError: If cls is not a subclass of StatefulEnvironment
ValueError: If name is empty or None

Example:
>>> class MyGameEnvironment(StatefulEnvironment):
...     # Implementation of abstract methods
...     pass

>>> register_environment("my_game", MyGameEnvironment)
>>>
>>> # Now the environment can be retrieved by name
>>> env_cls = get_environment_cls("my_game")
>>> env = env_cls(task_config)

Note:
Environment names are case-sensitive. It's recommended to use
consistent naming conventions (e.g., lowercase with underscores
or CamelCase) across your application.



## synth_ai.environments.environment.tools

### Classes

### AbstractTool

Abstract base class for all environment tools.

Tools are the primary mechanism for agents to interact with environments.
Each tool represents a specific action or capability that an agent can
invoke, such as moving, picking up items, or examining objects.

Tools define their own call and result schemas using Pydantic models,
enabling automatic validation and documentation generation. This ensures
that agents provide valid inputs and receive structured outputs.

The tool system supports:
- Type-safe argument validation
- Automatic error handling and reporting
- Consistent result formatting
- Dynamic tool registration and discovery

Attributes:
name: Unique identifier for this tool (used in EnvToolCall.tool)
call_schema: Pydantic model defining valid arguments for this tool
result_schema: Pydantic model defining the structure of results

Example:
>>> class MoveTool(AbstractTool):
...     name = "move"
...     call_schema = MoveArgs  # Pydantic model with 'direction' field
...
...     async def __call__(self, call: EnvToolCall) -> ToolResult:
...         direction = call.args['direction']
...         # Perform movement logic
...         if valid_move:
...             return ToolResult(ok=True, payload=new_position)
...         else:
...             return ToolResult(ok=False, error="Invalid move")

>>> # Register and use the tool
>>> move_tool = MoveTool()
>>> register_tool(move_tool)
>>> call = EnvToolCall(tool="move", args={"direction": "north"})
>>> result = await move_tool(call)

#### Methods

**__call__**(self, call: 'EnvToolCall') -> 'ToolResult'

    Execute the tool with the given tool call.

    This method contains the core logic for the tool's functionality.
    It should:
    1. Validate the tool call arguments (using call_schema)
    2. Perform the requested action
    3. Return a ToolResult with appropriate success/failure status

    Args:
    call: The tool call containing the tool name and arguments

    Returns:
    ToolResult: Result of tool execution with success status,
    payload data, and any error messages

    Raises:
    ValidationError: If call arguments don't match call_schema
    EnvironmentError: If tool execution fails due to environment state

    Example:
    >>> call = EnvToolCall(tool="move", args={"direction": "east"})
    >>> result = await tool(call)
    >>> if result.ok:
    ...     print(f"Moved to: {result.payload['position']}")
    ... else:
    ...     print(f"Move failed: {result.error}")

#### Properties

**__weakref__**

    list of weak references to the object


### EnvToolCall

Represents an agent-requested call to an environment tool.

This class encapsulates an action that an AI agent wants to perform
in an environment. Tool calls consist of a tool name and a dictionary
of arguments that will be passed to the tool for execution.

The tool call system provides a standardized way for agents to interact
with environments, making it easy to:
- Validate agent actions before execution
- Log and trace agent behavior
- Implement complex multi-step actions
- Handle errors and provide feedback

Attributes:
tool: The name of the tool to invoke (must be registered in environment)
args: Arguments to pass to the tool, with argument names as keys

Example:
>>> # Simple movement action
>>> move_call = EnvToolCall(tool="move", args={"direction": "north"})

>>> # Complex action with multiple parameters
>>> craft_call = EnvToolCall(
...     tool="craft_item",
...     args={"item": "sword", "materials": ["iron", "wood"], "quantity": 1}
... )

>>> # Action with no arguments
>>> look_call = EnvToolCall(tool="look", args={})

#### Properties

**__weakref__**

    list of weak references to the object


### ToolResult

Represents the result of executing an environment tool.

This class standardizes the response format for all tool executions,
providing a consistent interface for success/failure status, return
values, and error information. This makes it easy for agents and
environment systems to handle tool execution results uniformly.

Attributes:
ok: Whether the tool execution was successful
payload: The return value from the tool (None if failed or no return value)
error: Error message if execution failed (None if successful)

Example:
>>> # Successful tool execution
>>> success_result = ToolResult(
...     ok=True,
...     payload={"new_position": [5, 3], "items_collected": ["key"]},
...     error=None
... )

>>> # Failed tool execution
>>> error_result = ToolResult(
...     ok=False,
...     payload=None,
...     error="Cannot move north: wall blocking path"
... )

>>> # Success with no return value
>>> simple_success = ToolResult(ok=True)

#### Properties

**__weakref__**

    list of weak references to the object


### Functions

### register_tool(tool: 'AbstractTool') -> 'None'

Register a tool instance for use in environments.

This function adds a tool to the global registry, making it available
for environments to use when processing agent tool calls. Tools must
be registered before they can be invoked by agents.

The registry uses the tool's name attribute as the key, so tool names
must be unique across all registered tools.

Args:
tool: The tool instance to register. Must have a unique name.

Raises:
ValueError: If a tool with the same name is already registered
TypeError: If tool is not an instance of AbstractTool

Example:
>>> class LookTool(AbstractTool):
...     name = "look"
...     # ... implementation

>>> look_tool = LookTool()
>>> register_tool(look_tool)
>>>
>>> # Now agents can use: EnvToolCall(tool="look", args={})

Note:
Tools are typically registered during environment initialization
or module import. Once registered, tools remain available for
the duration of the application session.



## synth_ai.environments.stateful.core

### Classes

### StatefulEnvironment

Abstract base class for stateful environments in the Synth AI framework.

This class defines the interface for environments that maintain state between
interactions and support agent-environment interactions through tool calls.
StatefulEnvironments are designed to work with AI agents that can observe
the environment, take actions through tool calls, and receive feedback.

The environment follows a standard lifecycle:
1. Initialize - Set up initial state and return first observation
2. Step - Process agent tool calls and return new observations
3. Checkpoint - Save current state for potential restoration
4. Terminate - Clean up and finalize the environment

All methods are async to support non-blocking operations and integration
with modern async/await patterns in AI applications.

Example:
>>> class MyGameEnv(StatefulEnvironment):
...     async def initialize(self):
...         # Set up game state
...         return initial_observation
...
...     async def step(self, tool_calls):
...         # Process player actions
...         return new_observation
>>>
>>> env = MyGameEnv(task)
>>> obs = await env.initialize()
>>> result = await env.step([tool_call])

#### Methods

**checkpoint**(self) -> synth_ai.environments.environment.shared_engine.InternalObservation

    Create a checkpoint of the current environment state.

    This method saves the current state of the environment for potential
    restoration later. It's useful for:
    - Implementing save/load functionality
    - Creating branching scenarios for exploration
    - Debugging and development
    - Rollback mechanisms for error recovery

    Returns:
    InternalObservation: Current state observation, potentially including
    checkpoint metadata or state identifiers.

    Example:
    >>> checkpoint_obs = await env.checkpoint()
    >>> checkpoint_id = checkpoint_obs.metadata.get('checkpoint_id')

**initialize**(self) -> synth_ai.environments.environment.shared_engine.InternalObservation

    Initialize the environment and return the initial observation.

    This method sets up the environment's initial state, loads any
    necessary resources, and prepares the environment for agent interaction.
    It should be called once before any step() calls.

    Returns:
    InternalObservation: The initial state observation that the agent
    will use to understand the environment and plan its first action.

    Raises:
    EnvironmentError: If initialization fails due to invalid configuration
    or resource unavailability.

    Example:
    >>> env = MyEnvironment(task)
    >>> initial_obs = await env.initialize()
    >>> print(initial_obs.observation)  # Agent-visible state

**step**(self, tool_calls: List[synth_ai.environments.environment.tools.EnvToolCall]) -> synth_ai.environments.environment.shared_engine.InternalObservation

    Execute tool calls and return the resulting observation.

    This is the main interaction method where agents submit actions
    (as tool calls) and receive feedback from the environment. The method:
    1. Validates the tool calls (may call validate_tool_calls)
    2. Executes the actions in the environment
    3. Updates the environment state
    4. Returns the new observation for the agent

    Args:
    tool_calls: List of tool calls representing the agent's actions.
    Each tool call specifies a tool name and arguments.

    Returns:
    InternalObservation: The new state observation after executing
    the tool calls, including any changes, rewards, or feedback.

    Raises:
    ValidationError: If tool calls are invalid or cannot be executed.
    EnvironmentError: If execution fails due to environment state issues.

    Example:
    >>> tool_calls = [
    ...     EnvToolCall(tool="move", args={"direction": "north"}),
    ...     EnvToolCall(tool="pick_up", args={"item": "key"})
    ... ]
    >>> obs = await env.step(tool_calls)
    >>> print(obs.observation.get('player_location'))

**terminate**(self) -> synth_ai.environments.environment.shared_engine.InternalObservation

    Terminate the environment and return the final observation.

    This method performs cleanup operations, saves any persistent state,
    and prepares the final observation that summarizes the environment's
    end state. It should be called when the episode or session is complete.

    Returns:
    InternalObservation: The final state observation, typically including
    summary information, final scores, or completion status.

    Example:
    >>> final_obs = await env.terminate()
    >>> print(final_obs.observation.get('final_score'))

**validate_tool_calls**(self, tool_calls: synth_ai.environments.environment.tools.EnvToolCall)

    Validate that tool calls are properly formatted and executable.

    This method checks tool calls before execution to ensure they:
    - Reference valid tools available in this environment
    - Provide required arguments with correct types
    - Follow any environment-specific constraints or rules

    Args:
    tool_calls: The tool call(s) to validate. Can be a single call
    or a list of calls depending on the environment's capabilities.

    Raises:
    ValidationError: If tool calls are invalid, malformed, or not
    supported by this environment.
    TypeError: If tool_calls is not the expected type.

    Example:
    >>> tool_call = EnvToolCall(tool="move", args={"direction": "north"})
    >>> env.validate_tool_calls(tool_call)  # Raises if invalid


