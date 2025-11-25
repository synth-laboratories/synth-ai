"""Tracing v3 - Turso/sqld based tracing implementation.

This module provides a modern, async-first tracing system for capturing and storing
detailed execution traces from AI systems. It's designed to handle high-throughput
scenarios with proper async/await patterns throughout.

Architecture Overview:
---------------------
The v3 tracing system is built on several key components:

1. **Data Abstractions** (`abstractions.py`):
   - Dataclass-based models for traces, events, and messages
   - Type-safe representations of session data
   - Support for multiple event types (Runtime, Environment, LM/CAIS)

2. **Session Tracer** (`session_tracer.py`):
   - Main interface for creating and managing trace sessions
   - Async context managers for session and timestep management
   - Automatic event and message recording with proper ordering

3. **Async Storage** (`turso/manager.py`):
   - SQLAlchemy async engine with Turso/sqld backend
   - Batch insert capabilities for high-throughput scenarios
   - Analytics views for querying trace data

4. **Decorators** (`decorators.py`):
   - Context-aware decorators using asyncio's ContextVar
   - Automatic LLM call tracing with token/cost tracking
   - Session and turn number propagation across async boundaries

5. **Hook System** (`hooks.py`):
   - Extensible hook points throughout the tracing lifecycle
   - Support for both sync and async hook callbacks
   - Pre/post processing of events and messages

6. **Replica Sync** (`replica_sync.py`):
   - Optional background sync with remote Turso database
   - Local embedded SQLite for low-latency writes
   - Configurable sync intervals

Key Features:
------------
- **Async-First**: All database operations are async, preventing blocking
- **Context Propagation**: Session/turn info flows through async call chains
- **Type Safety**: Full typing support with dataclasses and type hints
- **Extensibility**: Hook system allows custom processing logic
- **Performance**: Batch operations and connection pooling for efficiency

Usage Example:
-------------
    from synth_ai.core.tracing_v3 import SessionTracer

    tracer = SessionTracer()
    await tracer.initialize()

    async with tracer.session() as session_id:
        async with tracer.timestep("step1", turn_number=1):
            # Record events during execution
            await tracer.record_event(RuntimeEvent(...))
            await tracer.record_message("User input", "user")

Configuration:
-------------
The system uses environment variables for configuration:
- TURSO_LOCAL_DB_URL: Local SQLite database URL
- TURSO_POOL_SIZE: Connection pool size (default: 8)
- TURSO_ECHO_SQL: Enable SQL logging (default: false)
- SQLD_DB_PATH: Path to SQLite database file

See `config.py` for full configuration options.
"""

from .abstractions import (
    BaseEvent,
    EnvironmentEvent,
    RuntimeEvent,
    SessionEventMarkovBlanketMessage,
    SessionMessageContent,
    SessionTimeStep,
    SessionTrace,
    TimeRecord,
)
from .config import TursoConfig
from .llm_call_record_helpers import BaseLMResponse
from .session_tracer import SessionTracer

__all__ = [
    "SessionTracer",
    "SessionTrace",
    "SessionTimeStep",
    "BaseEvent",
    "RuntimeEvent",
    "EnvironmentEvent",
    "SessionEventMarkovBlanketMessage",
    "SessionMessageContent",
    "TimeRecord",
    "TursoConfig",
    "BaseLMResponse",
]
