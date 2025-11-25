"""SQLAlchemy database utilities for the experiment queue."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import load_config


class Base(DeclarativeBase):
    """Declarative base for experiment queue ORM models."""


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Get (or lazily create) the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        config = load_config()
        # Enable WAL mode for SQLite to allow concurrent reads/writes
        # This is critical for Celery broker + our ORM to work together
        connect_args = {
            "check_same_thread": False,
            "timeout": 5.0,  # 5 second timeout - fast enough but allows WAL mode to work
        }
        # Enable WAL mode via PRAGMA after connection
        def _enable_wal(dbapi_conn, connection_record):
            # dbapi_conn is the raw sqlite3 connection, not SQLAlchemy
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA synchronous=NORMAL")
        
        _engine = create_engine(
            config.sqlalchemy_url,
            echo=False,
            future=True,
            connect_args=connect_args,
            pool_pre_ping=True,
            poolclass=None,  # Use default pool
        )
        # Register event listener to enable WAL mode on each connection
        from sqlalchemy import event
        event.listen(_engine, "connect", _enable_wal)
    return _engine


def get_session() -> Session:
    """Return a new Session bound to the shared engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            class_=Session,
            expire_on_commit=False,
            autoflush=False,
            future=True,
        )
    return _session_factory()


def init_db() -> None:
    """Create tables if they do not exist and ensure WAL mode is enabled.
    
    This initializes the application database (SQLite) for experiment queue data.
    Celery broker uses Redis, so no SQLite locking conflicts.
    """
    engine = get_engine()
    
    # Enable WAL mode on the database file before creating tables
    # This must happen before Celery's broker connection tries to use it
    # Use multiple connections to ensure WAL mode is persistent
    with engine.connect() as conn:
        # Enable WAL mode (allows concurrent reads/writes)
        result = conn.execute(text("PRAGMA journal_mode=WAL"))
        wal_result = result.scalar()
        if wal_result != "wal":
            # Database might be locked or in use - try again
            import time
            time.sleep(0.1)
            result = conn.execute(text("PRAGMA journal_mode=WAL"))
            wal_result = result.scalar()
        conn.execute(text("PRAGMA synchronous=NORMAL"))
        conn.execute(text("PRAGMA busy_timeout=5000"))  # 5 seconds - allows concurrent access
        conn.commit()
        
        # Verify WAL mode is enabled
        if wal_result != "wal":
            import warnings
            warnings.warn(
                f"WAL mode not enabled! Got: {wal_result}. "
                f"This may cause SQLite locking errors with Celery.",
                RuntimeWarning,
                stacklevel=2,
            )
    
    # Create our application tables
    Base.metadata.create_all(engine)
    
    # Migrate schema: add status_json column if it doesn't exist
    with engine.begin() as conn:  # Use begin() to ensure transaction is committed
        # Check if table exists first
        table_exists = conn.execute(text(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='experiment_jobs'"
        )).scalar() > 0
        
        if table_exists:
            # Check if status_json column exists
            result = conn.execute(text(
                "SELECT COUNT(*) FROM pragma_table_info('experiment_jobs') WHERE name='status_json'"
            ))
            column_exists = result.scalar() > 0
            
            if not column_exists:
                # Add status_json column
                conn.execute(text(
                    "ALTER TABLE experiment_jobs ADD COLUMN status_json TEXT"
                ))
    
    # Migrate schema: create job_execution_logs table if it doesn't exist
    with engine.connect() as conn:
        # Check if job_execution_logs table exists
        result = conn.execute(text(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='job_execution_logs'"
        ))
        table_exists = result.scalar() > 0
        
        if not table_exists:
            # Create job_execution_logs table
            conn.execute(text("""
                CREATE TABLE job_execution_logs (
                    log_id VARCHAR(64) PRIMARY KEY,
                    job_id VARCHAR(64) NOT NULL,
                    command TEXT NOT NULL,
                    working_directory TEXT NOT NULL,
                    returncode INTEGER NOT NULL,
                    stdout TEXT NOT NULL DEFAULT '',
                    stderr TEXT NOT NULL DEFAULT '',
                    python_executable VARCHAR(255),
                    environment_keys TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (job_id) REFERENCES experiment_jobs(job_id) ON DELETE CASCADE
                )
            """))
            # Create indexes
            conn.execute(text("CREATE INDEX idx_job_execution_logs_job ON job_execution_logs(job_id)"))
            conn.execute(text("CREATE INDEX idx_job_execution_logs_returncode ON job_execution_logs(returncode)"))
            conn.execute(text("CREATE INDEX idx_job_execution_logs_created ON job_execution_logs(created_at)"))
            conn.commit()
    
    # Force WAL mode one more time after table creation
    # This ensures it's set even if table creation changed something
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        conn.commit()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide a transactional scope for DB operations."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
