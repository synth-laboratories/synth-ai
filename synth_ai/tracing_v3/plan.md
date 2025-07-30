# Comprehensive Refactor Plan: DuckDB (tracing_v2) → Turso/sqld (tracing_v3)

## Overview
Migrate from DuckDB's embedded analytical engine to Turso's sqld daemon, which provides:
- Multi-writer MVCC (no "database is locked" errors)
- HTTP/gRPC access over localhost
- Pure SQLite 3 dialect
- Lightweight ~10MB daemon
- Built-in replication support

## Key Architecture Changes

### 1. Connection Model
- **FROM**: Direct embedded DuckDB with process-wide locks
- **TO**: Async HTTP client to sqld daemon at `http://127.0.0.1:8080`

### 2. SQL Dialect
- **FROM**: DuckDB-specific (sequences, JSON type)
- **TO**: SQLite 3 (AUTOINCREMENT, TEXT with JSON functions)

### 3. Schema Definition
- **FROM**: Raw SQL strings in schema.py
- **TO**: SQLAlchemy declarative models with type safety

### 4. Concurrency
- **FROM**: Single-writer with threading locks
- **TO**: Multi-writer MVCC via sqld daemon

### 5. Python Stack
- **FROM**: Synchronous DuckDB API
- **TO**: Async SQLAlchemy 2.0 with libsql-client

## Implementation Steps

### Phase 1: Infrastructure Setup
1. Create tracing_v3 directory structure mirroring v2
2. Add sqld daemon management utilities
3. Update dependencies (libsql-client, sqlalchemy-libsql)
4. Create configuration for Turso connection

### Phase 2: Core Implementation
1. Create AsyncSQLTraceManager to replace DuckDBTraceManager
2. Convert all database operations to async
3. Migrate schema from DuckDB to SQLite syntax
4. Port storage abstraction layer

### Phase 3: Migration & Testing
1. Create data migration script (DuckDB → Turso)
2. Update all tests for async operations
3. Add concurrent write tests
4. Update examples and documentation

## SQLAlchemy Schema Design

### Benefits of Using SQLAlchemy ORM
1. **Type Safety**: Column types are validated at runtime
2. **Relationships**: Automatic foreign key handling and lazy loading
3. **Query Builder**: Type-safe query construction
4. **Migration Support**: Alembic for schema versioning
5. **Database Agnostic**: Easy to switch between SQLite/PostgreSQL later

### Example Schema Conversion

```python
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Index, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()

class SessionTrace(Base):
    __tablename__ = 'session_traces'
    
    session_id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    num_timesteps = Column(Integer, default=0)
    num_events = Column(Integer, default=0)
    num_messages = Column(Integer, default=0)
    metadata = Column(JSON)  # SQLite stores as TEXT
    experiment_id = Column(String, ForeignKey('experiments.experiment_id'))
    
    # Relationships
    timesteps = relationship("SessionTimestep", back_populates="session", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="session", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    experiment = relationship("Experiment", back_populates="sessions")
    
    __table_args__ = (
        Index('idx_session_created', 'created_at'),
        Index('idx_session_experiment', 'experiment_id'),
    )

class SessionTimestep(Base):
    __tablename__ = 'session_timesteps'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('session_traces.session_id'), nullable=False)
    step_id = Column(String, nullable=False)
    turn_number = Column(Integer, nullable=False)
    started_at = Column(DateTime, default=func.current_timestamp())
    completed_at = Column(DateTime)
    
    # Relationships
    session = relationship("SessionTrace", back_populates="timesteps")
    
    __table_args__ = (
        Index('idx_timestep_session_step', 'session_id', 'step_id', unique=True),
        Index('idx_timestep_turn', 'turn_number'),
    )

class Event(Base):
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey('session_traces.session_id'), nullable=False)
    step_id = Column(String, nullable=False)
    event_type = Column(String, nullable=False)  # Using String instead of Enum for flexibility
    event_data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # For LM CAIS events
    model_name = Column(String)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    total_tokens = Column(Integer)
    cost_usd = Column(Integer)  # Store as cents
    latency_ms = Column(Integer)
    span_id = Column(String)
    trace_id = Column(String)
    
    # Relationships
    session = relationship("SessionTrace", back_populates="events")
    
    __table_args__ = (
        Index('idx_event_session_step', 'session_id', 'step_id'),
        Index('idx_event_type', 'event_type'),
        Index('idx_event_created', 'created_at'),
    )
```

### Query Examples with SQLAlchemy

```python
# Instead of raw SQL:
# "SELECT * FROM session_traces WHERE created_at > ? ORDER BY created_at DESC"

# Use type-safe queries:
recent_sessions = await session.execute(
    select(SessionTrace)
    .where(SessionTrace.created_at > cutoff_date)
    .order_by(SessionTrace.created_at.desc())
)

# Complex joins become readable:
session_with_events = await session.execute(
    select(SessionTrace)
    .options(selectinload(SessionTrace.events))
    .where(SessionTrace.session_id == session_id)
)
```

## Detailed Migration Checklist

### Schema Migration
- [ ] Replace `CREATE SEQUENCE` → `INTEGER PRIMARY KEY AUTOINCREMENT`
- [ ] Keep `JSON` column type (treated as TEXT in SQLite)
- [ ] Keep `TIMESTAMP` (maps to NUMERIC)
- [ ] Convert materialized views to regular views
- [ ] Ensure all indexes are SQLite-compatible
- [ ] Update foreign key constraints for SQLite syntax

### Code Changes
- [ ] Replace `DuckDBTraceManager` with `AsyncSQLTraceManager`
- [ ] Convert all database methods to async
- [ ] Replace threading locks with async session management
- [ ] Update bulk operations to use SQLAlchemy's insert().values()
- [ ] Convert DataFrame operations to use mappings().all()
- [ ] Update query_traces to return DataFrames from dict results
- [ ] Port feature extraction utilities (ft_utils.py)
- [ ] Create SQLAlchemy models from Pydantic types
- [ ] Use SQLAlchemy query builder instead of raw SQL
- [ ] Implement relationship loading strategies

### Dependencies
```python
# New requirements for tracing_v3
libsql-client
sqlalchemy-libsql
sqlalchemy>=2.0
aiosqlite
alembic  # For schema migrations
pandas  # Keep for DataFrame compatibility
tqdm    # Keep for progress bars
```

### Schema Migrations with Alembic
Using SQLAlchemy enables Alembic for version-controlled schema migrations:

```bash
# Initialize Alembic
alembic init synth_ai/tracing_v3/alembic

# Create initial migration
alembic revision --autogenerate -m "Initial tracing v3 schema"

# Apply migrations
alembic upgrade head
```

This provides:
- Version-controlled schema changes
- Rollback capabilities
- Multi-environment support (dev/staging/prod)
- Automatic migration generation from model changes

### Migration Script Components
1. **Export from DuckDB**:
   ```python
   COPY session_traces TO 'session_traces.parquet' (FORMAT 'parquet');
   COPY session_timesteps TO 'session_timesteps.parquet' (FORMAT 'parquet');
   COPY events TO 'events.parquet' (FORMAT 'parquet');
   COPY messages TO 'messages.parquet' (FORMAT 'parquet');
   ```

2. **Import to Turso**:
   - Read Parquet files with pandas
   - Bulk insert using async SQLAlchemy
   - Verify row counts and data integrity

3. **Update sequences**:
   - Find max IDs for each table
   - Set SQLite's sqlite_sequence appropriately

### Testing Updates
- [ ] Add pytest-asyncio to test dependencies
- [ ] Convert all database tests to async
- [ ] Create test fixture for sqld instance
- [ ] Add concurrent write stress tests
- [ ] Verify transaction isolation

### API Compatibility
Maintain backward compatibility where possible:
- Keep same method signatures (now async)
- Return same data structures (DataFrames, dicts)
- Preserve all existing functionality
- Add deprecation warnings for sync usage

## Benefits of Migration

### 1. Better Concurrency
- No more primary key clashes
- True multi-writer support
- Reduced lock contention

### 2. Network Access
- HTTP/gRPC API for distributed access
- Can run tracer in separate process/container
- REST API for debugging/inspection

### 3. Simpler Deployment
- Single daemon vs embedded complexity
- No process coordination needed
- Easier container deployment

### 4. Future-Proof
- Built-in replication support
- Cloud sync capabilities (Turso)
- Vector type support for embeddings

### 5. Familiar SQL
- Pure SQLite 3 syntax
- Better tooling support
- Easier debugging with standard SQLite tools

### 6. Modern ORM Benefits
- Type-safe SQLAlchemy models
- Automatic relationship management
- Query optimization
- Schema versioning with Alembic
- IDE autocomplete and type hints

## Directory Structure

```
tracing_v3/
├── __init__.py                 # Module exports
├── plan.md                     # This file
├── turso.md                    # Turso documentation
├── abstractions.py             # Core data structures (from v2)
├── config.py                   # Configuration with Turso settings
├── decorators.py               # Async-aware decorators
├── hooks.py                    # Hook system (from v2)
├── session_tracer.py           # Async SessionTracer
├── utils.py                    # Utility functions
├── turso/                      # Turso/sqld implementation
│   ├── __init__.py
│   ├── daemon.py              # sqld daemon management
│   ├── ft_utils.py            # Feature extraction (async)
│   ├── manager.py             # AsyncSQLTraceManager
│   ├── models.py              # SQLAlchemy declarative models
│   └── schema.py              # Schema creation and migration helpers
├── storage/                    # Storage abstraction (async)
│   ├── __init__.py
│   ├── base.py                # Async abstract base
│   ├── config.py              # Turso configuration
│   ├── exceptions.py          # Custom exceptions
│   ├── factory.py             # Factory with Turso support
│   ├── types.py               # Type definitions (unchanged)
│   └── utils.py               # Async utilities
├── migration/                  # Migration utilities
│   ├── __init__.py
│   ├── export.py              # DuckDB export logic
│   ├── import.py              # Turso import logic
│   └── verify.py              # Data verification
└── examples/                   # Updated examples
    ├── basic_async_usage.py
    ├── concurrent_writes.py
    └── migration_example.py
```

## Timeline Estimate

- **Week 1**: Infrastructure setup, AsyncSQLTraceManager skeleton
- **Week 2**: Schema migration, core async methods
- **Week 3**: Storage layer port, feature extraction
- **Week 4**: Migration scripts, testing, documentation

## Risk Mitigation

1. **Data Loss**: Create full backups before migration
2. **Performance**: Benchmark critical queries before/after
3. **Compatibility**: Run v2 and v3 in parallel during transition
4. **Rollback**: Keep v2 code intact for emergency rollback

## Success Criteria

- [ ] All tests pass with async operations
- [ ] Concurrent writes work without conflicts
- [ ] Performance meets or exceeds v2 for typical workloads
- [ ] Zero data loss during migration
- [ ] Clean API for users to migrate their code
- [ ] SQLAlchemy models provide full type safety
- [ ] Alembic migrations work correctly
- [ ] Relationships load efficiently with proper strategies
- [ ] Query builder produces optimal SQL


```python
# abstractions.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

@dataclass
class TimeRecord:
    event_time: float
    message_time: Optional[int] = None

@dataclass
class SessionEventMessage:
    content: str
    message_type: str
    time_record: TimeRecord

@dataclass
class BaseEvent:
    system_instance_id: str
    time_record: TimeRecord
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_metadata: Optional[List[Any]] = None

@dataclass
class RuntimeEvent(BaseEvent):
    actions: List[int]

@dataclass
class EnvironmentEvent(BaseEvent):
    reward: float
    terminated: bool

@dataclass
class SessionTimeStep:
    step_id: str
    step_index: int
    timestamp: datetime
    events: List[BaseEvent] = field(default_factory=list)
    step_messages: List[SessionEventMessage] = field(default_factory=list)
    step_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SessionTrace:
    session_id: str
    created_at: datetime
    session_time_steps: List[SessionTimeStep]
    event_history: List[BaseEvent]
    message_history: List[SessionEventMessage]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```

```python
# config.py
from dataclasses import dataclass
import os

@dataclass
class TursoConfig:
    db_url: str = os.getenv("TURSO_DATABASE_URL", "libsql://localhost:8080")
    auth_token: str = os.getenv("TURSO_AUTH_TOKEN", "")
    pool_size: int = int(os.getenv("TURSO_POOL_SIZE", "8"))
    foreign_keys: bool = True

CONFIG = TursoConfig()
```

```python
# decorators.py
import contextvars
from functools import wraps
from typing import Callable, Any

_session_id_ctx = contextvars.ContextVar("session_id", default=None)
_turn_number_ctx = contextvars.ContextVar("turn_number", default=None)

def set_session_id(session_id: str):
    _session_id_ctx.set(session_id)

def set_turn_number(turn: int):
    _turn_number_ctx.set(turn)

def with_session(fn: Callable[..., Any]):
    @wraps(fn)
    async def _inner(*args, **kwargs):
        if _session_id_ctx.get() is None:
            raise RuntimeError("session_id not set")
        return await fn(*args, **kwargs)
    return _inner
```

```python
# hooks.py
# Placeholder for domain‑specific hooks, copied straight from v2 where available
CRAFT_HOOKS = []  # populate from v2 when migrating
```

```python
# utils.py
import json
from datetime import datetime
from typing import Any

def iso_now() -> str:
    return datetime.utcnow().isoformat()

def json_dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, separators=(",", ":"))
```

```python
# session_tracer.py
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from .abstractions import (SessionTrace, SessionTimeStep, BaseEvent,
                           SessionEventMessage)
from .decorators import set_session_id, set_turn_number
from .turso.manager import AsyncSQLTraceManager
from .config import CONFIG

class SessionTracer:
    def __init__(self, hooks=None, db_url: Optional[str] = None):
        self.hooks = hooks or []
        self._current_trace: Optional[SessionTrace] = None
        self._lock = asyncio.Lock()
        self.db_url = db_url or CONFIG.db_url
        self.db: Optional[AsyncSQLTraceManager] = None

    @property
    def current_session(self):
        return self._current_trace

    async def start_session(self, session_id: str):
        async with self._lock:
            set_session_id(session_id)
            self._current_trace = SessionTrace(
                session_id=session_id,
                created_at=datetime.utcnow(),
                session_time_steps=[],
                event_history=[],
                message_history=[],
            )
            self.db = AsyncSQLTraceManager(self.db_url)

    async def start_timestep(self, step_id: str):
        assert self._current_trace
        step = SessionTimeStep(
            step_id=step_id,
            step_index=len(self._current_trace.session_time_steps),
            timestamp=datetime.utcnow(),
        )
        self._current_trace.session_time_steps.append(step)

    async def record_event(self, event: BaseEvent):
        assert self._current_trace
        self._current_trace.event_history.append(event)
        self._current_trace.session_time_steps[-1].events.append(event)

    async def record_message(self, msg: SessionEventMessage):
        assert self._current_trace
        self._current_trace.message_history.append(msg)
        self._current_trace.session_time_steps[-1].step_messages.append(msg)

    async def end_session(self, save: bool = True):
        async with self._lock:
            if save and self.db:
                await self.db.insert_session_trace(self._current_trace)
            trace = self._current_trace
            self._current_trace = None
            return trace
```

```python
# turso/__init__.py
from .manager import AsyncSQLTraceManager  # re‑export for convenience
```

```python
# turso/daemon.py
import subprocess
import pathlib
import shutil
import sys
from typing import Optional

_BINARY = shutil.which("sqld") or shutil.which("libsql-server")

def start_sqld(db_path: str = "traces.db", port: int = 8080) -> subprocess.Popen:
    if not _BINARY:
        raise RuntimeError("sqld binary not found in PATH")
    db_file = pathlib.Path(db_path).resolve()
    args = [
        _BINARY,
        "--db", str(db_file),
        "--http-listen-addr", f"127.0.0.1:{port}",
        "--idle-shutdown-timeout", "0",
    ]
    return subprocess.Popen(args, stdout=sys.stdout, stderr=sys.stderr)

def stop_sqld(proc: subprocess.Popen):
    proc.terminate()
    proc.wait()
```

```python
# turso/ft_utils.py
from typing import Sequence, Dict, Any
from itertools import chain
import numpy as np

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def one_hot(indices: Sequence[int], size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.int8)
    vec[list(indices)] = 1
    return vec
```

```python
# turso/schema.py
from sqlalchemy import (
    MetaData, Table, Column, Integer, String, Text, DateTime, ForeignKey, JSON
)

metadata = MetaData()

session_traces = Table(
    "session_traces",
    metadata,
    Column("session_id", String, primary_key=True),
    Column("created_at", DateTime, nullable=False),
    Column("num_timesteps", Integer, nullable=False, default=0),
    Column("num_events", Integer, nullable=False, default=0),
    Column("num_messages", Integer, nullable=False, default=0),
    Column("metadata", JSON)
)

session_timesteps = Table(
    "session_timesteps",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String, ForeignKey("session_traces.session_id")),
    Column("step_id", String),
    Column("step_index", Integer),
    Column("timestamp", DateTime),
    Column("num_events", Integer),
    Column("num_messages", Integer),
    Column("step_metadata", JSON),
)

events = Table(
    "events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String, ForeignKey("session_traces.session_id")),
    Column("timestep_id", Integer, ForeignKey("session_timesteps.id")),
    Column("event_type", String),
    Column("system_instance_id", String),
    Column("event_time", DateTime),
    Column("message_time", Integer),
    Column("model_name", String),
    Column("provider", String),
    Column("prompt_tokens", Integer),
    Column("completion_tokens", Integer),
    Column("total_tokens", Integer),
    Column("cost", Text),
    Column("latency_ms", Integer),
    Column("reward", Text),
    Column("terminated", Text),
    Column("system_state_before", Text),
    Column("system_state_after", Text),
    Column("metadata", JSON),
    Column("event_metadata", JSON),
)

messages = Table(
    "messages",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String, ForeignKey("session_traces.session_id")),
    Column("timestep_id", Integer, ForeignKey("session_timesteps.id")),
    Column("message_type", String),
    Column("content", Text),
    Column("timestamp", DateTime),
    Column("event_time", DateTime),
    Column("message_time", Integer),
)
```

```python
# turso/manager.py
import asyncio
from typing import Any, Dict, List, Sequence
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
from sqlalchemy import insert, select, text
from sqlalchemy.orm import sessionmaker
from ..config import CONFIG
from ..abstractions import SessionTrace
from .schema import metadata, session_traces, session_timesteps, events, messages
from ..utils import json_dumps

class AsyncSQLTraceManager:
    def __init__(self, db_url: str = CONFIG.db_url):
        self.engine: AsyncEngine = create_async_engine(
            db_url, pool_size=CONFIG.pool_size, future=True
        )
        self.Session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)
        self._schema_lock = asyncio.Lock()
        self._schema_ready = False

    async def _init_schema(self):
        async with self._schema_lock:
            if self._schema_ready:
                return
            async with self.engine.begin() as conn:
                await conn.run_sync(metadata.create_all)
                if CONFIG.foreign_keys:
                    await conn.execute(text("PRAGMA foreign_keys=ON"))
            self._schema_ready = True

    async def insert_session_trace(self, trace: SessionTrace):
        await self._init_schema()
        async with self.Session() as ses:
            await ses.execute(
                insert(session_traces).values(
                    session_id=trace.session_id,
                    created_at=trace.created_at,
                    num_timesteps=len(trace.session_time_steps),
                    num_events=len(trace.event_history),
                    num_messages=len(trace.message_history),
                    metadata=json_dumps(trace.metadata),
                ).prefix_with("OR IGNORE")
            )
            step_id_map: Dict[str, int] = {}
            # timesteps
            for ts in trace.session_time_steps:
                res = await ses.execute(
                    insert(session_timesteps).values(
                        session_id=trace.session_id,
                        step_id=ts.step_id,
                        step_index=ts.step_index,
                        timestamp=ts.timestamp,
                        num_events=len(ts.events),
                        num_messages=len(ts.step_messages),
                        step_metadata=json_dumps(ts.step_metadata),
                    ).returning(session_timesteps.c.id)
                )
                step_id_map[ts.step_id] = res.scalar()
            # events
            ev_rows: List[Dict[str, Any]] = []
            for ev in trace.event_history:
                ev_rows.append(
                    dict(
                        session_id=trace.session_id,
                        timestep_id=step_id_map.get(ev.metadata.get("step_id")),
                        event_type=ev.__class__.__name__.lower(),
                        system_instance_id=ev.system_instance_id,
                        event_time=ev.time_record.event_time,
                        message_time=ev.time_record.message_time,
                        metadata=json_dumps(ev.metadata),
                        event_metadata=json_dumps(ev.event_metadata),
                    )
                )
            if ev_rows:
                await ses.execute(insert(events), ev_rows)
            # messages
            msg_rows: List[Dict[str, Any]] = []
            for msg in trace.message_history:
                msg_rows.append(
                    dict(
                        session_id=trace.session_id,
                        message_type=msg.message_type,
                        content=msg.content,
                        timestamp=msg.time_record.event_time,
                        event_time=msg.time_record.event_time,
                        message_time=msg.time_record.message_time,
                    )
                )
            if msg_rows:
                await ses.execute(insert(messages), msg_rows)
            await ses.commit()

    async def query(self, stmt) -> List[Dict[str, Any]]:
        await self._init_schema()
        async with self.Session() as ses:
            res = await ses.execute(stmt)
            rows = res.mappings().all()
            return [dict(r) for r in rows]
```

All files above compile into a minimal yet functional async tracing stack using Turso/sqld; fill any remaining v2‑specific hooks or event subclasses as you migrate.
