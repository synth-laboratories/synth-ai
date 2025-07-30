"""Async SQLAlchemy-based trace manager for Turso/sqld."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union
import pandas as pd
from sqlalchemy import select, insert, update, delete, text, and_, or_, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
from sqlalchemy.orm import sessionmaker, selectinload, joinedload
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import IntegrityError
import logging

from ..config import CONFIG
from ..abstractions import SessionTrace, SessionTimeStep, BaseEvent, LMCAISEvent, EnvironmentEvent, RuntimeEvent
from ..utils import json_dumps
from .models import (
    Base, SessionTrace as DBSessionTrace, SessionTimestep as DBSessionTimestep,
    Event as DBEvent, Message as DBMessage, Experiment as DBExperiment,
    System as DBSystem, SystemVersion as DBSystemVersion, analytics_views
)

logger = logging.getLogger(__name__)


class AsyncSQLTraceManager:
    """Async trace storage manager using SQLAlchemy and Turso/sqld."""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or CONFIG.db_url
        self.engine: Optional[AsyncEngine] = None
        self.SessionLocal: Optional[sessionmaker] = None
        self._schema_lock = asyncio.Lock()
        self._schema_ready = False
        
    async def initialize(self):
        """Initialize the database connection and schema."""
        if self.engine is None:
            logger.info(f"🔗 Initializing database connection to: {self.db_url}")
            
            # For SQLite, use NullPool and set busy timeout
            if self.db_url.startswith("sqlite"):
                # Extract the file path from the URL
                db_path = self.db_url.replace("sqlite+aiosqlite:///", "")
                import os
                
                # Check if database file exists
                if not os.path.exists(db_path):
                    logger.warning(f"⚠️  Database file not found: {db_path}")
                    logger.warning("🔧 Make sure './serve.sh' is running to start the turso/sqld service")
                else:
                    logger.info(f"✅ Found database file: {db_path}")
                
                connect_args = {"timeout": 30.0}  # 30 second busy timeout
                self.engine = create_async_engine(
                    self.db_url,  # Use instance db_url, not CONFIG
                    poolclass=NullPool,  # No connection pooling for SQLite
                    connect_args=connect_args,
                    echo=CONFIG.echo_sql
                )
            else:
                connect_args = CONFIG.get_connect_args()
                engine_kwargs = CONFIG.get_engine_kwargs()
                self.engine = create_async_engine(
                    self.db_url,  # Use instance db_url, not CONFIG
                    connect_args=connect_args,
                    **engine_kwargs
                )
            
            self.SessionLocal = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
        await self._ensure_schema()
    
    async def _ensure_schema(self):
        """Ensure database schema is created."""
        async with self._schema_lock:
            if self._schema_ready:
                return
                
            logger.info("📊 Initializing database schema...")
            
            async with self.engine.begin() as conn:
                # Create tables with checkfirst=True to handle concurrent creation
                try:
                    await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn, checkfirst=True))
                    logger.info("✅ Database schema created/verified successfully")
                except Exception as e:
                    # If tables already exist, that's fine - another worker created them
                    if "already exists" not in str(e):
                        logger.error(f"❌ Failed to create database schema: {e}")
                        raise
                    else:
                        logger.info("✅ Database schema already exists")
                
                # Enable foreign keys for SQLite
                if CONFIG.foreign_keys:
                    await conn.execute(text("PRAGMA foreign_keys = ON"))
                    
                # Set journal mode
                if CONFIG.journal_mode:
                    await conn.execute(text(f"PRAGMA journal_mode = {CONFIG.journal_mode}"))
                
                # Create analytics views
                for view_name, view_sql in analytics_views.items():
                    try:
                        await conn.execute(text(view_sql))
                    except Exception as e:
                        # Views might already exist
                        if "already exists" not in str(e):
                            logger.warning(f"Could not create view {view_name}: {e}")
                    
            self._schema_ready = True
            logger.info("🎯 Database ready for use!")
    
    @asynccontextmanager
    async def session(self):
        """Get an async database session."""
        if not self.SessionLocal:
            await self.initialize()
        async with self.SessionLocal() as session:
            yield session
    
    async def insert_session_trace(self, trace: SessionTrace) -> str:
        """Insert a complete session trace."""
        async with self.session() as sess:
            try:
                # Convert to cents for cost storage
                def to_cents(cost: Optional[float]) -> Optional[int]:
                    return int(cost * 100) if cost is not None else None
                
                # Insert session
                db_session = DBSessionTrace(
                session_id=trace.session_id,
                created_at=trace.created_at,
                num_timesteps=len(trace.session_time_steps),
                num_events=len(trace.event_history),
                num_messages=len(trace.message_history),
                session_metadata=trace.metadata or {},
                )
                sess.add(db_session)
                
                # Track timestep IDs for foreign keys
                step_id_map: Dict[str, int] = {}
                
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
                        num_messages=len(step.step_messages),
                        step_metadata=step.step_metadata or {},
                    )
                    sess.add(db_step)
                    await sess.flush()  # Get the auto-generated ID
                    step_id_map[step.step_id] = db_step.id
                
                # Insert events
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
                        "event_metadata_json": {**event.metadata, "actions": event.actions},
                    })
                    else:
                        event_data["event_type"] = event.__class__.__name__.lower()
                    
                    db_event = DBEvent(**event_data)
                    sess.add(db_event)
                
                # Insert messages
                for msg in trace.message_history:
                    db_msg = DBMessage(
                        session_id=trace.session_id,
                        timestep_id=step_id_map.get(msg.metadata.get("step_id")) if hasattr(msg, 'metadata') else None,
                        message_type=msg.message_type,
                        content=msg.content,
                        event_time=msg.time_record.event_time,
                        message_time=msg.time_record.message_time,
                        message_metadata=msg.metadata if hasattr(msg, 'metadata') else {},
                    )
                    sess.add(db_msg)
                
                await sess.commit()
                return trace.session_id
            except IntegrityError as e:
                # Handle duplicate session IDs gracefully
                if "UNIQUE constraint failed: session_traces.session_id" in str(e):
                    await sess.rollback()
                    return trace.session_id  # Return existing ID
                raise
    
    async def get_session_trace(self, session_id: str) -> Optional[Dict[str, Any]]:
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
    
    async def query_traces(self, query: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        async with self.session() as sess:
            result = await sess.execute(text(query), params or {})
            rows = result.mappings().all()
            return pd.DataFrame(rows)
    
    async def get_model_usage(self, 
                            start_date: datetime = None,
                            end_date: datetime = None,
                            model_name: str = None) -> pd.DataFrame:
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
    
    async def create_experiment(self,
                              experiment_id: str,
                              name: str,
                              description: str = None,
                              configuration: Dict[str, Any] = None) -> str:
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
    
    async def batch_insert_sessions(self, 
                                  traces: List[SessionTrace],
                                  batch_size: int = None) -> List[str]:
        """Batch insert multiple session traces."""
        batch_size = batch_size or CONFIG.batch_size
        inserted_ids = []
        
        for i in range(0, len(traces), batch_size):
            batch = traces[i:i + batch_size]
            for trace in batch:
                session_id = await self.insert_session_trace(trace)
                inserted_ids.append(session_id)
                
        return inserted_ids
    
    async def get_sessions_by_experiment(self, 
                                       experiment_id: str,
                                       limit: int = None) -> List[Dict[str, Any]]:
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
                select(DBSessionTrace)
                .where(DBSessionTrace.session_id == session_id)
            )
            session = result.scalar_one_or_none()
            
            if session:
                await sess.delete(session)
                await sess.commit()
                return True
            return False
    
    async def close(self):
        """Close the database connection."""
        if self.engine:
            await self.engine.dispose()
            self.engine = None
            self.SessionLocal = None
            self._schema_ready = False