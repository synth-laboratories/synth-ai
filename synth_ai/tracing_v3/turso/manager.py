from __future__ import annotations
"""Async SQLAlchemy-based trace manager for Turso/sqld.

This module provides the database interface for the tracing system using
async SQLAlchemy with a Turso/sqld backend. It handles all database operations
including schema creation, session storage, and analytics queries.

Key Features:
------------
- Async-first design using aiosqlite for local SQLite
- Automatic schema creation and migration
- Batch insert capabilities for high-throughput scenarios
- Analytics views for efficient querying
- Connection pooling and retry logic

Performance Considerations:
--------------------------
- Uses NullPool for SQLite to avoid connection issues
- Implements busy timeout for concurrent access
- Batches inserts to reduce transaction overhead
- Creates indexes for common query patterns
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy import event
from sqlalchemy.orm import selectinload, sessionmaker
from sqlalchemy.pool import NullPool

from ..abstractions import (
    EnvironmentEvent,
    LMCAISEvent,
    RuntimeEvent,
    SessionTrace,
)
from ..config import CONFIG
from .models import (
    Base,
    analytics_views,
)
from .models import (
    Event as DBEvent,
)
from .models import (
    Experiment as DBExperiment,
)
from .models import (
    Message as DBMessage,
)
from .models import (
    SessionTimestep as DBSessionTimestep,
)
from .models import (
    SessionTrace as DBSessionTrace,
)
from .models import (
    OutcomeReward as DBOutcomeReward,
)
from .models import (
    EventReward as DBEventReward,
)

logger = logging.getLogger(__name__)


class AsyncSQLTraceManager:
    """Async trace storage manager using SQLAlchemy and Turso/sqld.

    Handles all database operations for the tracing system. Designed to work
    with both local SQLite (via aiosqlite) and remote Turso databases.

    The manager handles:
    - Connection lifecycle management
    - Schema creation and verification
    - Transaction management
    - Batch operations for efficiency
    - Analytics view creation
    """

    def __init__(self, db_url: str | None = None):
        self.db_url = db_url or CONFIG.db_url
        self.engine: AsyncEngine | None = None
        self.SessionLocal: sessionmaker | None = None
        self._schema_lock = asyncio.Lock()
        self._schema_ready = False

    async def initialize(self):
        """Initialize the database connection and schema.

        This method is idempotent and thread-safe. It:
        1. Creates the async engine with appropriate settings
        2. Verifies database file exists (for SQLite)
        3. Creates schema if needed
        4. Sets up analytics views

        The schema lock ensures only one worker creates the schema in
        concurrent scenarios.
        """
        if self.engine is None:
            logger.debug(f"🔗 Initializing database connection to: {self.db_url}")

            # For SQLite, use NullPool to avoid connection pool issues
            # SQLite doesn't handle concurrent connections well, so we create
            # a new connection for each operation
            if self.db_url.startswith("sqlite"):
                # Extract the file path from the URL
                db_path = self.db_url.replace("sqlite+aiosqlite:///", "")
                import os

                # Check if database file exists
                if not os.path.exists(db_path):
                    logger.debug(f"⚠️  Database file not found: {db_path}")
                    logger.debug(
                        "🔧 Make sure './serve.sh' is running to start the turso/sqld service"
                    )
                else:
                    logger.debug(f"✅ Found database file: {db_path}")

                # Set a high busy timeout to handle concurrent access
                # This allows SQLite to wait instead of immediately failing
                connect_args = {"timeout": 30.0}  # 30 second busy timeout
                self.engine = create_async_engine(
                    self.db_url,  # Use instance db_url, not CONFIG
                    poolclass=NullPool,  # No connection pooling for SQLite
                    connect_args=connect_args,
                    echo=CONFIG.echo_sql,
                )
                # Ensure PRAGMA foreign_keys=ON for every connection
                try:
                    @event.listens_for(self.engine.sync_engine, "connect")
                    def _set_sqlite_pragma(dbapi_connection, connection_record):  # type: ignore[no-redef]
                        try:
                            cursor = dbapi_connection.cursor()
                            cursor.execute("PRAGMA foreign_keys=ON")
                            cursor.close()
                        except Exception:
                            pass
                except Exception:
                    pass
            else:
                connect_args = CONFIG.get_connect_args()
                engine_kwargs = CONFIG.get_engine_kwargs()
                self.engine = create_async_engine(
                    self.db_url,  # Use instance db_url, not CONFIG
                    connect_args=connect_args,
                    **engine_kwargs,
                )

            self.SessionLocal = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )

        await self._ensure_schema()

    async def _ensure_schema(self):
        """Ensure database schema is created.

        Uses a lock to prevent race conditions when multiple workers start
        simultaneously. The checkfirst=True parameter handles cases where
        another worker already created the schema.
        """
        async with self._schema_lock:
            if self._schema_ready:
                return

            logger.debug("📊 Initializing database schema...")

            async with self.engine.begin() as conn:
                # Use a transaction to ensure atomic schema creation
                # checkfirst=True prevents errors if tables already exist
                try:
                    await conn.run_sync(
                        lambda sync_conn: Base.metadata.create_all(sync_conn, checkfirst=True)
                    )
                    # logger.info("✅ Database schema created/verified successfully")
                except Exception as e:
                    # If tables already exist, that's fine - another worker created them
                    if "already exists" not in str(e):
                        logger.error(f"❌ Failed to create database schema: {e}")
                        raise
                    else:
                        logger.debug("✅ Database schema already exists")

                # Enable foreign keys for SQLite - critical for data integrity
                # This must be done for each connection in SQLite
                if CONFIG.foreign_keys:
                    await conn.execute(text("PRAGMA foreign_keys = ON"))

                # Set journal mode
                if CONFIG.journal_mode:
                    await conn.execute(text(f"PRAGMA journal_mode = {CONFIG.journal_mode}"))

                # Create analytics views for efficient querying
                # These are materialized as views to avoid recalculation
                for view_name, view_sql in analytics_views.items():
                    try:
                        await conn.execute(text(view_sql))
                    except Exception as e:
                        # Views might already exist from another worker
                        if "already exists" not in str(e):
                            logger.warning(f"Could not create view {view_name}: {e}")

            self._schema_ready = True
            # logger.debug("🎯 Database ready for use!")

    @asynccontextmanager
    async def session(self):
        """Get an async database session."""
        if not self.SessionLocal:
            await self.initialize()
        async with self.SessionLocal() as session:
            yield session

    async def insert_session_trace(self, trace: SessionTrace) -> str:
        """Insert a complete session trace.

        This method handles the complex task of inserting a complete session
        with all its timesteps, events, and messages. It uses a single
        transaction for atomicity and flushes after timesteps to get their
        auto-generated IDs for foreign keys.

        Args:
            trace: The complete session trace to store

        Returns:
            The session ID

        Raises:
            IntegrityError: If session ID already exists (handled gracefully)
        """
        async with self.session() as sess:
            try:
                # Convert to cents for cost storage - avoids floating point
                # precision issues and allows for integer arithmetic
                def to_cents(cost: float | None) -> int | None:
                    return int(cost * 100) if cost is not None else None

                # Insert session
                db_session = DBSessionTrace(
                    session_id=trace.session_id,
                    created_at=trace.created_at,
                    num_timesteps=len(trace.session_time_steps),
                    num_events=len(trace.event_history),
                    num_messages=len(trace.markov_blanket_message_history),
                    session_metadata=trace.metadata or {},
                )
                sess.add(db_session)

                # Track timestep IDs for foreign keys - we need these to link
                # events and messages to their respective timesteps
                step_id_map: dict[str, int] = {}

                # Insert timesteps
                for step in trace.session_time_steps:
                    db_step = DBSessionTimestep(
                        session_id=trace.session_id,
                        step_id=step.step_id,
                        step_index=step.step_index,
                        turn_number=step.turn_number,
                        started_at=step.timestamp,
                        completed_at=step.completed_at,
                        num_events=len(step.events),
                        num_messages=len(step.markov_blanket_messages),
                        step_metadata=step.step_metadata or {},
                    )
                    sess.add(db_step)
                    # Flush to get the auto-generated ID without committing
                    # This allows us to use the ID for foreign keys while
                    # maintaining transaction atomicity
                    await sess.flush()  # Get the auto-generated ID
                    step_id_map[step.step_id] = db_step.id

                # Insert events - handle different event types with their
                # specific fields while maintaining a unified storage model
                for event in trace.event_history:
                    event_data = {
                        "session_id": trace.session_id,
                        "timestep_id": step_id_map.get(event.metadata.get("step_id")),
                        "system_instance_id": event.system_instance_id,
                        "event_time": event.time_record.event_time,
                        "message_time": event.time_record.message_time,
                        "event_metadata_json": event.metadata or {},
                        "event_extra_metadata": event.event_metadata,
                    }

                    if isinstance(event, LMCAISEvent):
                        # Serialize call_records if present
                        call_records_data = None
                        if event.call_records:
                            from dataclasses import asdict

                            call_records_data = [asdict(record) for record in event.call_records]

                        event_data.update(
                            {
                                "event_type": "cais",
                                "model_name": event.model_name,
                                "provider": event.provider,
                                "input_tokens": event.input_tokens,
                                "output_tokens": event.output_tokens,
                                "total_tokens": event.total_tokens,
                                "cost_usd": to_cents(event.cost_usd),
                                "latency_ms": event.latency_ms,
                                "span_id": event.span_id,
                                "trace_id": event.trace_id,
                                "system_state_before": event.system_state_before,
                                "system_state_after": event.system_state_after,
                                "call_records": call_records_data,  # Store in the proper column
                            }
                        )
                    elif isinstance(event, EnvironmentEvent):
                        event_data.update(
                            {
                                "event_type": "environment",
                                "reward": event.reward,
                                "terminated": event.terminated,
                                "truncated": event.truncated,
                                "system_state_before": event.system_state_before,
                                "system_state_after": event.system_state_after,
                            }
                        )
                    elif isinstance(event, RuntimeEvent):
                        event_data.update(
                            {
                                "event_type": "runtime",
                                "event_metadata_json": {**event.metadata, "actions": event.actions},
                            }
                        )
                    else:
                        event_data["event_type"] = event.__class__.__name__.lower()

                    db_event = DBEvent(**event_data)
                    sess.add(db_event)

                # Insert messages
                for msg in trace.markov_blanket_message_history:
                    db_msg = DBMessage(
                        session_id=trace.session_id,
                        timestep_id=step_id_map.get(msg.metadata.get("step_id"))
                        if hasattr(msg, "metadata")
                        else None,
                        message_type=msg.message_type,
                        content=msg.content,
                        event_time=msg.time_record.event_time,
                        message_time=msg.time_record.message_time,
                        message_metadata=msg.metadata if hasattr(msg, "metadata") else {},
                    )
                    sess.add(db_msg)

                # Commit the entire transaction atomically
                await sess.commit()
                return trace.session_id
            except IntegrityError as e:
                # Handle duplicate session IDs gracefully - this can happen
                # in distributed systems or retries. We return the existing
                # ID to maintain idempotency
                if "UNIQUE constraint failed: session_traces.session_id" in str(e):
                    await sess.rollback()
                    return trace.session_id  # Return existing ID
                raise

    async def get_session_trace(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve a session trace by ID."""
        async with self.session() as sess:
            result = await sess.execute(
                select(DBSessionTrace)
                .options(
                    selectinload(DBSessionTrace.timesteps),
                    selectinload(DBSessionTrace.events),
                    selectinload(DBSessionTrace.messages),
                )
                .where(DBSessionTrace.session_id == session_id)
            )
            session = result.scalar_one_or_none()

            if not session:
                return None

            return {
                "session_id": session.session_id,
                "created_at": session.created_at,
                "num_timesteps": session.num_timesteps,
                "num_events": session.num_events,
                "num_messages": session.num_messages,
                "metadata": session.session_metadata,
                "timesteps": [
                    {
                        "step_id": step.step_id,
                        "step_index": step.step_index,
                        "turn_number": step.turn_number,
                        "started_at": step.started_at,
                        "completed_at": step.completed_at,
                        "metadata": step.step_metadata,
                    }
                    for step in sorted(session.timesteps, key=lambda s: s.step_index)
                ],
            }

    async def query_traces(
        self, query: str, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        async with self.session() as sess:
            result = await sess.execute(text(query), params or {})
            rows = result.mappings().all()
            return pd.DataFrame(rows)

    async def get_model_usage(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        model_name: str | None = None,
    ) -> pd.DataFrame:
        """Get model usage statistics."""
        query = """
            SELECT * FROM model_usage_stats
            WHERE 1=1
        """
        params = {}

        if start_date:
            query += " AND last_used >= :start_date"
            params["start_date"] = start_date

        if end_date:
            query += " AND first_used <= :end_date"
            params["end_date"] = end_date

        if model_name:
            query += " AND model_name = :model_name"
            params["model_name"] = model_name

        query += " ORDER BY usage_count DESC"

        return await self.query_traces(query, params)

    async def create_experiment(
        self,
        experiment_id: str,
        name: str,
        description: str | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> str:
        """Create a new experiment."""
        async with self.session() as sess:
            experiment = DBExperiment(
                experiment_id=experiment_id,
                name=name,
                description=description,
                configuration=configuration or {},
            )
            sess.add(experiment)
            await sess.commit()
            return experiment_id

    async def link_session_to_experiment(self, session_id: str, experiment_id: str):
        """Link a session to an experiment."""
        async with self.session() as sess:
            await sess.execute(
                update(DBSessionTrace)
                .where(DBSessionTrace.session_id == session_id)
                .values(experiment_id=experiment_id)
            )
            await sess.commit()

    async def batch_insert_sessions(
        self, traces: list[SessionTrace], batch_size: int | None = None
    ) -> list[str]:
        """Batch insert multiple session traces.

        Processes traces in batches to balance memory usage and performance.
        Each batch is inserted in a separate transaction to avoid holding
        locks for too long.

        Args:
            traces: List of session traces to insert
            batch_size: Number of traces per batch (defaults to config)

        Returns:
            List of inserted session IDs
        """
        batch_size = batch_size or CONFIG.batch_size
        inserted_ids = []

        # Process in chunks to avoid memory issues with large datasets
        for i in range(0, len(traces), batch_size):
            batch = traces[i : i + batch_size]
            # Insert each trace in the batch - could be optimized further
            # with bulk inserts if needed
            for trace in batch:
                session_id = await self.insert_session_trace(trace)
                inserted_ids.append(session_id)

        return inserted_ids

    async def get_sessions_by_experiment(
        self, experiment_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get all sessions for an experiment."""
        async with self.session() as sess:
            query = (
                select(DBSessionTrace)
                .where(DBSessionTrace.experiment_id == experiment_id)
                .order_by(DBSessionTrace.created_at.desc())
            )

            if limit:
                query = query.limit(limit)

            result = await sess.execute(query)
            sessions = result.scalars().all()

            return [
                {
                    "session_id": s.session_id,
                    "created_at": s.created_at,
                    "num_timesteps": s.num_timesteps,
                    "num_events": s.num_events,
                    "num_messages": s.num_messages,
                    "metadata": s.metadata,
                }
                for s in sessions
            ]

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all related data."""
        async with self.session() as sess:
            # Get the session object to trigger cascade deletes
            result = await sess.execute(
                select(DBSessionTrace).where(DBSessionTrace.session_id == session_id)
            )
            session = result.scalar_one_or_none()

            if session:
                await sess.delete(session)
                await sess.commit()
                return True
            return False

    async def close(self):
        """Close the database connection.

        Properly disposes of the engine and all connections. This is important
        for cleanup, especially with SQLite which can leave lock files.
        """
        if self.engine:
            # Dispose of all connections in the pool
            await self.engine.dispose()
            # Clear all state to allow re-initialization if needed
            self.engine = None
            self.SessionLocal = None
            self._schema_ready = False

    # -------------------------------
    # Incremental insert helpers
    # -------------------------------

    async def ensure_session(self, session_id: str, *, created_at: datetime | None = None, metadata: dict[str, Any] | None = None):
        """Ensure a DB session row exists for session_id."""
        async with self.session() as sess:
            result = await sess.execute(select(DBSessionTrace).where(DBSessionTrace.session_id == session_id))
            existing = result.scalar_one_or_none()
            if existing:
                return
            row = DBSessionTrace(
                session_id=session_id,
                created_at=created_at or datetime.utcnow(),
                num_timesteps=0,
                num_events=0,
                num_messages=0,
                session_metadata=metadata or {},
            )
            sess.add(row)
            await sess.commit()

    async def ensure_timestep(self, session_id: str, *, step_id: str, step_index: int, turn_number: int | None = None, started_at: datetime | None = None, completed_at: datetime | None = None, metadata: dict[str, Any] | None = None) -> int:
        """Ensure a timestep row exists; return its DB id."""
        async with self.session() as sess:
            result = await sess.execute(
                select(DBSessionTimestep).where(DBSessionTimestep.session_id == session_id, DBSessionTimestep.step_id == step_id)
            )
            row = result.scalar_one_or_none()
            if row:
                return row.id
            row = DBSessionTimestep(
                session_id=session_id,
                step_id=step_id,
                step_index=step_index,
                turn_number=turn_number,
                started_at=started_at or datetime.utcnow(),
                completed_at=completed_at,
                num_events=0,
                num_messages=0,
                step_metadata=metadata or {},
            )
            sess.add(row)
            await sess.flush()
            # increment session num_timesteps
            await sess.execute(
                update(DBSessionTrace)
                .where(DBSessionTrace.session_id == session_id)
                .values(num_timesteps=DBSessionTrace.num_timesteps + 1)
            )
            await sess.commit()
            return row.id

    async def insert_message_row(self, session_id: str, *, timestep_db_id: int | None, message_type: str, content: str, event_time: float | None = None, message_time: int | None = None, metadata: dict[str, Any] | None = None) -> int:
        """Insert a message and return its id."""
        async with self.session() as sess:
            db_msg = DBMessage(
                session_id=session_id,
                timestep_id=timestep_db_id,
                message_type=message_type,
                content=content,
                event_time=event_time,
                message_time=message_time,
                message_metadata=metadata or {},
            )
            sess.add(db_msg)
            await sess.flush()
            # increment session num_messages
            await sess.execute(
                update(DBSessionTrace)
                .where(DBSessionTrace.session_id == session_id)
                .values(num_messages=DBSessionTrace.num_messages + 1)
            )
            await sess.commit()
            return db_msg.id

    async def insert_event_row(self, session_id: str, *, timestep_db_id: int | None, event: EnvironmentEvent | LMCAISEvent | RuntimeEvent, metadata_override: dict[str, Any] | None = None) -> int:
        """Insert an event and return its id."""
        def to_cents(cost: float | None) -> int | None:
            return int(cost * 100) if cost is not None else None

        event_data: dict[str, Any] = {
            "session_id": session_id,
            "timestep_id": timestep_db_id,
            "system_instance_id": event.system_instance_id,
            "event_time": event.time_record.event_time,
            "message_time": event.time_record.message_time,
            "event_metadata_json": metadata_override or event.metadata or {},
            "event_extra_metadata": getattr(event, "event_metadata", None),
        }
        if isinstance(event, LMCAISEvent):
            call_records_data = None
            if getattr(event, "call_records", None):
                from dataclasses import asdict

                call_records_data = [asdict(record) for record in event.call_records]
            event_data.update({
                "event_type": "cais",
                "model_name": event.model_name,
                "provider": event.provider,
                "input_tokens": event.input_tokens,
                "output_tokens": event.output_tokens,
                "total_tokens": event.total_tokens,
                "cost_usd": to_cents(event.cost_usd),
                "latency_ms": event.latency_ms,
                "span_id": event.span_id,
                "trace_id": event.trace_id,
                "system_state_before": event.system_state_before,
                "system_state_after": event.system_state_after,
                "call_records": call_records_data,
            })
        elif isinstance(event, EnvironmentEvent):
            event_data.update({
                "event_type": "environment",
                "reward": event.reward,
                "terminated": event.terminated,
                "truncated": event.truncated,
                "system_state_before": event.system_state_before,
                "system_state_after": event.system_state_after,
            })
        elif isinstance(event, RuntimeEvent):
            event_data.update({
                "event_type": "runtime",
                "event_metadata_json": {**(event.metadata or {}), "actions": event.actions},
            })
        else:
            event_data["event_type"] = event.__class__.__name__.lower()

        async with self.session() as sess:
            db_event = DBEvent(**event_data)
            sess.add(db_event)
            await sess.flush()
            # increment session num_events
            await sess.execute(
                update(DBSessionTrace)
                .where(DBSessionTrace.session_id == session_id)
                .values(num_events=DBSessionTrace.num_events + 1)
            )
            await sess.commit()
            return db_event.id

    # -------------------------------
    # Reward helpers
    # -------------------------------

    async def insert_outcome_reward(self, session_id: str, *, total_reward: int, achievements_count: int, total_steps: int, reward_metadata: dict | None = None) -> int:
        async with self.session() as sess:
            row = DBOutcomeReward(
                session_id=session_id,
                total_reward=total_reward,
                achievements_count=achievements_count,
                total_steps=total_steps,
                reward_metadata=reward_metadata or {},
            )
            sess.add(row)
            await sess.flush()
            await sess.commit()
            return row.id

    async def insert_event_reward(self, session_id: str, *, event_id: int, message_id: int | None = None, turn_number: int | None = None, reward_value: float = 0.0, reward_type: str | None = None, key: str | None = None, annotation: dict[str, Any] | None = None, source: str | None = None) -> int:
        async with self.session() as sess:
            row = DBEventReward(
                event_id=event_id,
                session_id=session_id,
                message_id=message_id,
                turn_number=turn_number,
                reward_value=reward_value,
                reward_type=reward_type,
                key=key,
                annotation=annotation or {},
                source=source,
            )
            sess.add(row)
            await sess.flush()
            await sess.commit()
            return row.id

    async def get_outcome_rewards(self) -> list[dict[str, Any]]:
        async with self.session() as sess:
            result = await sess.execute(select(DBOutcomeReward))
            rows = result.scalars().all()
            return [
                {
                    "id": r.id,
                    "session_id": r.session_id,
                    "total_reward": r.total_reward,
                    "achievements_count": r.achievements_count,
                    "total_steps": r.total_steps,
                    "created_at": r.created_at,
                }
                for r in rows
            ]

    async def get_outcome_rewards_by_min_reward(self, min_reward: int) -> list[str]:
        async with self.session() as sess:
            result = await sess.execute(
                select(DBOutcomeReward.session_id).where(DBOutcomeReward.total_reward >= min_reward)
            )
            return [row[0] for row in result.all()]
