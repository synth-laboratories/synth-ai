"""Main SessionTracer class for tracing v3."""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager

from .abstractions import SessionTrace, SessionTimeStep, BaseEvent, SessionEventMarkovBlanketMessage, TimeRecord
from .decorators import set_session_id, set_turn_number, set_session_tracer, SessionContext
from .turso.manager import AsyncSQLTraceManager
from .config import CONFIG
from .hooks import HookManager, GLOBAL_HOOKS
from .utils import generate_session_id


class SessionTracer:
    """Async session tracer with Turso/sqld backend."""

    def __init__(
        self,
        hooks: Optional[HookManager] = None,
        db_url: Optional[str] = None,
        auto_save: bool = True,
    ):
        """Initialize session tracer.

        Args:
            hooks: Hook manager instance (uses global hooks if not provided)
            db_url: Database URL (uses config default if not provided)
            auto_save: Whether to automatically save sessions on end
        """
        self.hooks = hooks or GLOBAL_HOOKS
        self._current_trace: Optional[SessionTrace] = None
        self._lock = asyncio.Lock()
        self.db_url = db_url or CONFIG.db_url
        self.db: Optional[AsyncSQLTraceManager] = None
        self.auto_save = auto_save
        self._current_step: Optional[SessionTimeStep] = None

    @property
    def current_session(self) -> Optional[SessionTrace]:
        """Get the current session trace."""
        return self._current_trace

    @property
    def current_step(self) -> Optional[SessionTimeStep]:
        """Get the current timestep."""
        return self._current_step

    async def initialize(self):
        """Initialize the database connection."""
        if self.db is None:
            self.db = AsyncSQLTraceManager(self.db_url)
            await self.db.initialize()

    async def start_session(
        self, session_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new session.
        
        Creates a new tracing session and sets up the necessary context variables.
        This method is thread-safe and will raise an error if a session is already
        active to prevent accidental session mixing.
        
        The session ID is propagated through asyncio context variables, allowing
        nested async functions to access the tracer without explicit passing.

        Args:
            session_id: Optional session ID (generated if not provided).
                       Useful for correlating traces with external systems.
            metadata: Optional session metadata. Common keys include:
                     - 'user_id': User identifier
                     - 'experiment_id': Experiment this session belongs to
                     - 'model_config': Model configuration used
                     - 'environment': Environment or context info

        Returns:
            The session ID (either provided or generated)
            
        Raises:
            RuntimeError: If a session is already active
        """
        async with self._lock:
            if self._current_trace is not None:
                raise RuntimeError("Session already active. End current session first.")

            session_id = session_id or generate_session_id()
            set_session_id(session_id)
            set_session_tracer(self)

            self._current_trace = SessionTrace(
                session_id=session_id,
                created_at=datetime.utcnow(),
                session_time_steps=[],
                event_history=[],
                markov_blanket_message_history=[],
                metadata=metadata or {},
            )

            # Initialize DB if needed
            if self.auto_save and self.db is None:
                await self.initialize()

            # Trigger hooks
            await self.hooks.trigger(
                "session_start", session_id=session_id, metadata=metadata or {}
            )

            return session_id

    async def start_timestep(
        self,
        step_id: str,
        turn_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionTimeStep:
        """Start a new timestep.

        Args:
            step_id: Unique step identifier
            turn_number: Optional turn number
            metadata: Optional step metadata

        Returns:
            The created timestep
        """
        if self._current_trace is None:
            raise RuntimeError("No active session. Start a session first.")

        step = SessionTimeStep(
            step_id=step_id,
            step_index=len(self._current_trace.session_time_steps),
            timestamp=datetime.utcnow(),
            turn_number=turn_number,
            step_metadata=metadata or {},
        )

        self._current_trace.session_time_steps.append(step)
        self._current_step = step

        if turn_number is not None:
            set_turn_number(turn_number)

        # Trigger hooks
        await self.hooks.trigger(
            "timestep_start", step=step, session_id=self._current_trace.session_id
        )

        return step

    async def end_timestep(self, step_id: Optional[str] = None):
        """End the current or specified timestep."""
        if self._current_trace is None:
            raise RuntimeError("No active session")

        if step_id:
            # Find specific step
            step = next(
                (s for s in self._current_trace.session_time_steps if s.step_id == step_id), None
            )
            if not step:
                raise ValueError(f"Step {step_id} not found")
        else:
            step = self._current_step

        if step and step.completed_at is None:
            step.completed_at = datetime.utcnow()

            # Trigger hooks
            await self.hooks.trigger(
                "timestep_end", step=step, session_id=self._current_trace.session_id
            )

        if step == self._current_step:
            self._current_step = None

    async def record_event(self, event: BaseEvent):
        """Record an event.

        Args:
            event: The event to record
        """
        if self._current_trace is None:
            raise RuntimeError("No active session")

        # Add step_id to event metadata if in a timestep
        if self._current_step:
            event.metadata["step_id"] = self._current_step.step_id

        # Trigger pre-recording hooks
        await self.hooks.trigger("event_recorded", event_obj=event)

        # Add to histories
        self._current_trace.event_history.append(event)
        if self._current_step:
            self._current_step.events.append(event)

    async def record_message(
        self,
        content: str,
        message_type: str,
        event_time: Optional[float] = None,
        message_time: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a message.

        Args:
            content: Message content
            message_type: Type of message (user, assistant, system, etc.)
            event_time: Optional event timestamp
            message_time: Optional message time
            metadata: Optional message metadata
        """
        if self._current_trace is None:
            raise RuntimeError("No active session")

        msg = SessionEventMarkovBlanketMessage(
            content=content,
            message_type=message_type,
            time_record=TimeRecord(
                event_time=event_time or datetime.utcnow().timestamp(), message_time=message_time
            ),
            metadata=metadata or {},
        )

        # Add step_id to metadata if in a timestep
        if self._current_step:
            msg.metadata["step_id"] = self._current_step.step_id

        # Trigger hooks
        await self.hooks.trigger("message_recorded", message=msg)

        # Add to histories
        self._current_trace.markov_blanket_message_history.append(msg)
        if self._current_step:
            self._current_step.markov_blanket_messages.append(msg)

    async def end_session(self, save: bool = None) -> SessionTrace:
        """End the current session.

        Args:
            save: Whether to save the session (uses auto_save if not specified)

        Returns:
            The completed session trace
        """
        async with self._lock:
            if self._current_trace is None:
                raise RuntimeError("No active session")

            # End any open timesteps
            for step in self._current_trace.session_time_steps:
                if step.completed_at is None:
                    step.completed_at = datetime.utcnow()

            # Trigger pre-save hooks
            await self.hooks.trigger("before_save", session=self._current_trace)

            # Save if requested
            should_save = save if save is not None else self.auto_save
            if should_save and self.db:
                await self.db.insert_session_trace(self._current_trace)

                # Trigger post-save hooks
                await self.hooks.trigger("after_save", session=self._current_trace)

            # Trigger session end hooks
            await self.hooks.trigger("session_end", session=self._current_trace)

            # Clear state
            trace = self._current_trace
            self._current_trace = None
            self._current_step = None
            set_session_id(None)
            set_turn_number(None)
            set_session_tracer(None)

            return trace

    @asynccontextmanager
    async def session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        save: bool = None,
    ):
        """Context manager for a session.

        Example:
            async with tracer.session() as session_id:
                # Do work within session
                await tracer.record_event(event)
        """
        session_id = await self.start_session(session_id, metadata)
        try:
            yield session_id
        finally:
            await self.end_session(save=save)

    @asynccontextmanager
    async def timestep(
        self,
        step_id: str,
        turn_number: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for a timestep.

        Example:
            async with tracer.timestep("step1") as step:
                # Do work within timestep
                await tracer.record_event(event)
        """
        step = await self.start_timestep(step_id, turn_number, metadata)
        try:
            yield step
        finally:
            await self.end_timestep(step_id)

    async def get_session_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent session history from database."""
        if self.db is None:
            await self.initialize()

        query = "SELECT * FROM session_traces ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {limit}"

        df = await self.db.query_traces(query)
        return df.to_dict("records")

    async def close(self):
        """Close database connections."""
        if self.db:
            await self.db.close()
            self.db = None
