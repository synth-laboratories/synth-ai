"""SQLAlchemy declarative models for tracing v3."""

from __future__ import annotations

import json

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.types import UserDefinedType

Base = declarative_base()


class Vector(UserDefinedType):
    """Custom type for vector storage in SQLite.

    Turso/libSQL supports native VECTOR type for embeddings.
    This will store vectors as JSON arrays in SQLite but can be
    upgraded to native VECTOR type when using Turso's extensions.
    """

    cache_ok = True

    def get_col_spec(self, **kw):
        # Use VECTOR type if available, otherwise TEXT
        return "VECTOR"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            if isinstance(value, list):
                return json.dumps(value)
            return value

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            if isinstance(value, str):
                return json.loads(value)
            return value

        return process


class JSONText(TypeDecorator):
    """Stores JSON as TEXT for SQLite compatibility."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value, default=str, separators=(",", ":"))
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


class SessionTrace(Base):
    """Database model for session traces.

    Stores high-level information about tracing sessions including
    metadata, statistics, and relationships to timesteps and events.
    """

    __tablename__ = "session_traces"

    session_id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=func.current_timestamp(), nullable=False)
    num_timesteps = Column(Integer, default=0, nullable=False)
    num_events = Column(Integer, default=0, nullable=False)
    num_messages = Column(Integer, default=0, nullable=False)
    session_metadata = Column("metadata", JSONText)
    experiment_id = Column(String, ForeignKey("experiments.experiment_id"))

    # Vector field for future use (e.g., session embeddings)
    embedding = Column(Vector)

    # Relationships
    timesteps = relationship(
        "SessionTimestep", back_populates="session", cascade="all, delete-orphan"
    )
    events = relationship("Event", back_populates="session", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    experiment = relationship("Experiment", back_populates="sessions")

    __table_args__ = (
        Index("idx_session_created", "created_at"),
        Index("idx_session_experiment", "experiment_id"),
    )


class SessionTimestep(Base):
    """Database model for session timesteps.

    Represents individual steps within a tracing session, with timing
    information and relationships to events and messages.
    """

    __tablename__ = "session_timesteps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("session_traces.session_id"), nullable=False)
    step_id = Column(String, nullable=False)
    step_index = Column(Integer, nullable=False)
    turn_number = Column(Integer)
    started_at = Column(DateTime, default=func.current_timestamp())
    completed_at = Column(DateTime)
    num_events = Column(Integer, default=0)
    num_messages = Column(Integer, default=0)
    step_metadata = Column("step_metadata", JSONText)

    # Relationships
    session = relationship("SessionTrace", back_populates="timesteps")
    events = relationship("Event", back_populates="timestep", cascade="all, delete-orphan")
    messages = relationship("Message", back_populates="timestep", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("session_id", "step_id", name="uq_session_step"),
        Index("idx_timestep_session_step", "session_id", "step_id"),
        Index("idx_timestep_turn", "turn_number"),
    )


class Event(Base):
    """Database model for events.

    Stores all types of events (LM CAIS, environment, runtime) with
    type-specific fields and common metadata. Supports vector embeddings
    for similarity search.
    """

    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("session_traces.session_id"), nullable=False)
    timestep_id = Column(Integer, ForeignKey("session_timesteps.id"))
    event_type = Column(String, nullable=False)  # 'cais', 'environment', 'runtime'
    system_instance_id = Column(String)
    event_time = Column(Float)  # Unix timestamp
    message_time = Column(Integer)
    created_at = Column(DateTime, default=func.current_timestamp())

    # LM CAIS event fields
    model_name = Column(String)
    provider = Column(String)
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    total_tokens = Column(Integer)
    cost_usd = Column(Integer)  # Store as cents to avoid float precision issues
    latency_ms = Column(Integer)
    span_id = Column(String)
    trace_id = Column(String)
    call_records = Column(JSONText)  # Store serialized LLMCallRecord list

    # Environment event fields
    reward = Column(Float)
    terminated = Column(Boolean)
    truncated = Column(Boolean)

    # Runtime event fields (actions stored in metadata)

    # System state tracking
    system_state_before = Column(JSONText)
    system_state_after = Column(JSONText)

    # Metadata fields
    event_metadata_json = Column("metadata", JSONText)
    event_extra_metadata = Column("event_metadata", JSONText)

    # Vector field for event embeddings (e.g., for similarity search)
    embedding = Column(Vector)

    # Relationships
    session = relationship("SessionTrace", back_populates="events")
    timestep = relationship("SessionTimestep", back_populates="events")

    __table_args__ = (
        Index("idx_event_session_step", "session_id", "timestep_id"),
        Index("idx_event_type", "event_type"),
        Index("idx_event_created", "created_at"),
        Index("idx_event_model", "model_name"),
        Index("idx_event_trace", "trace_id"),
        CheckConstraint(
            "event_type IN ('cais', 'environment', 'runtime')", name="check_event_type"
        ),
    )


class Message(Base):
    """Database model for messages.

    Stores conversational messages between users, assistants, and systems
    with support for embeddings and rich metadata.
    """

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("session_traces.session_id"), nullable=False)
    timestep_id = Column(Integer, ForeignKey("session_timesteps.id"))
    message_type = Column(
        String, nullable=False
    )  # 'user', 'assistant', 'system', 'tool_use', 'tool_result'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=func.current_timestamp())
    event_time = Column(Float)
    message_time = Column(Integer)
    message_metadata = Column("metadata", JSONText)

    # Vector field for message embeddings
    embedding = Column(Vector)

    # Relationships
    session = relationship("SessionTrace", back_populates="messages")
    timestep = relationship("SessionTimestep", back_populates="messages")

    __table_args__ = (
        Index("idx_message_session_step", "session_id", "timestep_id"),
        Index("idx_message_type", "message_type"),
        Index("idx_message_timestamp", "timestamp"),
        CheckConstraint(
            "message_type IN ('user', 'assistant', 'system', 'tool_use', 'tool_result')",
            name="check_message_type",
        ),
    )


class Experiment(Base):
    """Database model for experiments.

    Groups related sessions and systems for experimental evaluation
    and comparison. Supports rich configuration and metadata.
    """

    __tablename__ = "experiments"

    experiment_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(
        DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()
    )
    configuration = Column(JSONText)
    experiment_metadata = Column("metadata", JSONText)

    # Relationships
    sessions = relationship("SessionTrace", back_populates="experiment")
    systems = relationship(
        "ExperimentalSystem", back_populates="experiment", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_experiment_created", "created_at"),
        Index("idx_experiment_name", "name"),
    )


class System(Base):
    """Database model for systems.

    Represents agents, environments, or runtime systems that participate
    in tracing sessions. Supports versioning and type classification.
    """

    __tablename__ = "systems"

    system_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    system_type = Column(String)  # 'agent', 'environment', 'runtime'
    description = Column(Text)
    created_at = Column(DateTime, default=func.current_timestamp())
    system_metadata = Column("metadata", JSONText)

    # Relationships
    versions = relationship("SystemVersion", back_populates="system", cascade="all, delete-orphan")
    experiments = relationship("ExperimentalSystem", back_populates="system")

    __table_args__ = (
        Index("idx_system_name", "name"),
        Index("idx_system_type", "system_type"),
    )


class SystemVersion(Base):
    """Database model for system versions.

    Tracks different versions of systems with commit hashes,
    configuration changes, and relationships to experiments.
    """

    __tablename__ = "system_versions"

    version_id = Column(String, primary_key=True)
    system_id = Column(String, ForeignKey("systems.system_id"), nullable=False)
    version_number = Column(String, nullable=False)
    commit_hash = Column(String)
    created_at = Column(DateTime, default=func.current_timestamp())
    configuration = Column(JSONText)
    version_metadata = Column("metadata", JSONText)

    # Relationships
    system = relationship("System", back_populates="versions")
    experiments = relationship("ExperimentalSystem", back_populates="version")

    __table_args__ = (
        UniqueConstraint("system_id", "version_number", name="uq_system_version"),
        Index("idx_version_system", "system_id"),
        Index("idx_version_created", "created_at"),
    )


class ExperimentalSystem(Base):
    """Database model for experiment-system relationships.

    Junction table linking experiments with specific system versions,
    allowing tracking of which systems participated in which experiments.
    """

    __tablename__ = "experimental_systems"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, ForeignKey("experiments.experiment_id"), nullable=False)
    system_id = Column(String, ForeignKey("systems.system_id"), nullable=False)
    version_id = Column(String, ForeignKey("system_versions.version_id"), nullable=False)

    # Relationships
    experiment = relationship("Experiment", back_populates="systems")
    system = relationship("System", back_populates="experiments")
    version = relationship("SystemVersion", back_populates="experiments")

    __table_args__ = (
        UniqueConstraint("experiment_id", "system_id", name="uq_experiment_system"),
        Index("idx_experimental_system", "experiment_id", "system_id"),
    )


# Analytics Views (to be created as views in the database)
analytics_views = {
    "model_usage_stats": """
        CREATE VIEW IF NOT EXISTS model_usage_stats AS
        SELECT 
            model_name,
            provider,
            COUNT(*) as usage_count,
            SUM(input_tokens) as total_input_tokens,
            SUM(output_tokens) as total_output_tokens,
            SUM(total_tokens) as total_tokens,
            SUM(cost_usd) / 100.0 as total_cost_usd,
            AVG(latency_ms) as avg_latency_ms,
            MIN(created_at) as first_used,
            MAX(created_at) as last_used
        FROM events
        WHERE event_type = 'cais' AND model_name IS NOT NULL
        GROUP BY model_name, provider
    """,
    "session_summary": """
        CREATE VIEW IF NOT EXISTS session_summary AS
        SELECT 
            s.session_id,
            s.created_at,
            s.num_timesteps,
            s.num_events,
            s.num_messages,
            e.experiment_id,
            e.name as experiment_name,
            COUNT(DISTINCT ev.model_name) as unique_models_used,
            SUM(CASE WHEN ev.event_type = 'cais' THEN ev.cost_usd ELSE 0 END) / 100.0 as total_cost_usd
        FROM session_traces s
        LEFT JOIN experiments e ON s.experiment_id = e.experiment_id
        LEFT JOIN events ev ON s.session_id = ev.session_id
        GROUP BY s.session_id
    """,
    "experiment_overview": """
        CREATE VIEW IF NOT EXISTS experiment_overview AS
        SELECT 
            e.experiment_id,
            e.name,
            e.description,
            e.created_at,
            COUNT(DISTINCT s.session_id) as session_count,
            SUM(s.num_events) as total_events,
            SUM(s.num_messages) as total_messages,
            AVG(s.num_timesteps) as avg_timesteps_per_session
        FROM experiments e
        LEFT JOIN session_traces s ON e.experiment_id = s.experiment_id
        GROUP BY e.experiment_id
    """,
}


# Reward persistence tables


class OutcomeReward(Base):
    """Episode-level rewards/outcomes per session.

    Stores per-episode summary including total_reward (e.g., unique achievements),
    achievements_count, and total_steps. Used for filtering episodes by outcome.
    """

    __tablename__ = "outcome_rewards"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("session_traces.session_id"), nullable=False)
    total_reward = Column(Float, nullable=False)
    achievements_count = Column(Integer, nullable=False, default=0)
    total_steps = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=func.current_timestamp(), nullable=False)
    # Store additional structured metadata about the outcome (e.g., achievements list)
    reward_metadata = Column(JSONText)
    annotation = Column(JSONText)

    __table_args__ = (
        Index("idx_outcome_rewards_session", "session_id"),
        Index("idx_outcome_rewards_total", "total_reward"),
    )


class EventReward(Base):
    """First-class event-level rewards with annotations.

    Links to an event and session. `message_id` is optional.
    """

    __tablename__ = "event_rewards"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=False)
    session_id = Column(String, ForeignKey("session_traces.session_id"), nullable=False)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=True)
    turn_number = Column(Integer, nullable=True)
    reward_value = Column(Float, nullable=False, default=0.0)
    reward_type = Column(
        String, nullable=True
    )  # shaped | sparse | achievement | penalty | evaluator | human
    key = Column(String, nullable=True)  # e.g., achievement name
    annotation = Column(JSONText)  # free-form JSON
    source = Column(String, nullable=True)  # environment | runner | evaluator | human
    created_at = Column(DateTime, default=func.current_timestamp(), nullable=False)

    __table_args__ = (
        Index("idx_event_rewards_session", "session_id"),
        Index("idx_event_rewards_event", "event_id"),
        Index("idx_event_rewards_type", "reward_type"),
        Index("idx_event_rewards_key", "key"),
    )
