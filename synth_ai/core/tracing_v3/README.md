# Tracing V3 - Turso/sqld Implementation

This is the next generation of the tracing system, migrated from DuckDB to Turso/sqld for better concurrent write support and modern architecture.

## Key Features

- **Multi-writer MVCC**: No more "database is locked" errors
- **Async-first**: Built on SQLAlchemy 2.0 async with full type safety
- **Vector Support**: Native vector type support for embeddings (future use)
- **SQLAlchemy ORM**: Type-safe models with automatic relationship management
- **Lightweight**: ~10MB sqld daemon vs heavy database installations
- **HTTP/gRPC Access**: Network-based access for distributed systems

## Architecture Changes from V2

1. **Database**: DuckDB → Turso/sqld (SQLite-compatible)
2. **Concurrency**: Single-writer with locks → Multi-writer MVCC
3. **API**: Synchronous → Fully async
4. **Schema**: Raw SQL → SQLAlchemy declarative models
5. **Connections**: Embedded → HTTP client to local daemon

## Installation

```bash
# Install sqld daemon (macOS)
brew install turso-tech/tools/sqld

# Or using Docker
docker pull ghcr.io/tursodatabase/libsql-server:latest

# Install Python dependencies
pip install libsql-client sqlalchemy-libsql sqlalchemy>=2.0 aiosqlite alembic
```

## Quick Start

```python
import asyncio
from synth_ai.tracing_v3 import SessionTracer
from synth_ai.tracing_v3.abstractions import LMCAISEvent, TimeRecord

async def main():
    # Create tracer instance
    tracer = SessionTracer()
    
    # Start a session
    async with tracer.session(metadata={"user": "test"}) as session_id:
        print(f"Started session: {session_id}")
        
        # Start a timestep
        async with tracer.timestep("step1", turn_number=1):
            # Record an LLM event
            event = LMCAISEvent(
                system_instance_id="gpt-4",
                time_record=TimeRecord(event_time=time.time()),
                model_name="gpt-4",
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cost_usd=0.0045,
                latency_ms=1200,
            )
            await tracer.record_event(event)
            
            # Record a message
            await tracer.record_message(
                content="Hello, world!",
                message_type="assistant"
            )
    
    # Session is automatically saved

if __name__ == "__main__":
    asyncio.run(main())
```

## Starting the sqld Daemon

### Option 1: Manual Start
```bash
sqld --db traces.db --http-listen-addr 127.0.0.1:8080
```

### Option 2: Using Python Helper
```python
from synth_ai.tracing_v3.turso.daemon import SqldDaemon

# Start daemon
daemon = SqldDaemon()
daemon.start()

# Your tracing code here...

# Stop daemon
daemon.stop()
```

### Option 3: Context Manager
```python
from synth_ai.tracing_v3.turso.daemon import SqldDaemon

async def main():
    with SqldDaemon() as daemon:
        # Daemon is running
        tracer = SessionTracer()
        # ... use tracer ...
```

## Configuration

Environment variables:
- `TURSO_DATABASE_URL`: Database URL (default: `sqlite+libsql://http://127.0.0.1:8080`)
- `TURSO_AUTH_TOKEN`: Authentication token (for cloud Turso)
- `TURSO_POOL_SIZE`: Connection pool size (default: 8)
- `TURSO_ECHO_SQL`: Echo SQL statements (default: false)
- `SQLD_HTTP_PORT`: HTTP port for sqld (default: 8080)

## SQLAlchemy Models

The system uses declarative SQLAlchemy models with full type safety:

```python
from synth_ai.tracing_v3.turso.models import SessionTrace, Event, Message

# Models include:
- SessionTrace: Main session records
- SessionTimestep: Logical turns within sessions
- Event: All event types (CAIS, Environment, Runtime)
- Message: Communication messages
- Experiment: Experiment tracking
- System/SystemVersion: System versioning
```

## Vector Support

The models include vector fields for future embedding storage:

```python
class Event(Base):
    # ... other fields ...
    embedding = Column(Vector)  # For event embeddings
    
class Message(Base):
    # ... other fields ...
    embedding = Column(Vector)  # For message embeddings
```

## Migration from V2

1. Export data from DuckDB:
```python
# In v2 environment
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
db = DuckDBTraceManager("traces.duckdb")
db.export_to_parquet("exports/")
```

2. Import to v3:
```python
# In v3 environment
from synth_ai.tracing_v3.migration import import_from_parquet
await import_from_parquet("exports/")
```

## Hooks System

The v3 system includes a powerful hooks system for extending functionality:

```python
from synth_ai.tracing_v3.hooks import HookManager

# Create custom hook
async def log_expensive_calls(event):
    if hasattr(event, 'cost_usd') and event.cost_usd > 0.10:
        print(f"Expensive call: ${event.cost_usd}")

# Register hook
tracer.hooks.register("event_recorded", log_expensive_calls)
```

## Performance Tips

1. **Batch Operations**: Use `batch_insert_sessions` for bulk inserts
2. **Connection Pooling**: Adjust `TURSO_POOL_SIZE` based on workload
3. **Local Daemon**: Keep sqld running locally for best performance
4. **Indexes**: The schema includes optimized indexes for common queries

## Differences from V2

| Feature | V2 (DuckDB) | V3 (Turso) |
|---------|-------------|------------|
| Concurrency | Single-writer | Multi-writer MVCC |
| API | Synchronous | Async/await |
| Schema | Raw SQL | SQLAlchemy ORM |
| Vectors | Not supported | Native support |
| Connection | Embedded | HTTP/gRPC |
| File size | ~50MB | ~10MB daemon |

## Troubleshooting

### "Connection refused" error
- Ensure sqld daemon is running: `ps aux | grep sqld`
- Check port availability: `lsof -i :8080`

### "No such table" error
- The schema is created automatically on first use
- Check database file permissions

### Performance issues
- Enable WAL mode (automatic with sqld)
- Increase connection pool size
- Use batch operations for bulk inserts

## Future Enhancements

- [ ] PostgreSQL backend support
- [ ] Cloud Turso integration
- [ ] Vector similarity search
- [ ] Real-time streaming
- [ ] GraphQL API
## Migration from DuckDB (V2) to Turso/sqld (V3)

### Overview

Migrated from DuckDB's embedded analytical engine to Turso's sqld daemon for:
- **Multi-writer MVCC**: No more "database is locked" errors
- **Better concurrency**: Multiple processes can write simultaneously
- **Network access**: HTTP/gRPC access for distributed systems
- **Lightweight**: ~10MB sqld daemon vs heavy database installations

### Why sqld?

`sqld` (the "server‑mode" binary that comes with libSQL/Turso) is the sweet spot between DuckDB's in‑process model and a full Postgres install. It provides:
- SQLite compatibility (same SQL dialect)
- Multi-writer MVCC support
- Lightweight footprint
- Easy local development setup

### Architecture Changes

1. **Database**: DuckDB → Turso/sqld (SQLite-compatible)
2. **Concurrency**: Single-writer with locks → Multi-writer MVCC
3. **API**: Synchronous → Fully async
4. **Schema**: Raw SQL → SQLAlchemy declarative models
5. **Connections**: Embedded → HTTP client to local daemon

See `plan.md` and `turso.md` in `old/` for detailed migration notes.
