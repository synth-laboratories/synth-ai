"""SQLAlchemy ORM models for the experiment queue."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class ExperimentStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class ExperimentJobType(str, enum.Enum):
    GEPA = "gepa"
    MIPRO = "mipro"


class ExperimentJobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TrialStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Experiment(Base):
    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text())
    status: Mapped[ExperimentStatus] = mapped_column(
        SAEnum(ExperimentStatus, name="experiment_status"),
        default=ExperimentStatus.QUEUED,
        nullable=False,
    )
    parallelism_limit: Mapped[int] = mapped_column(Integer, default=3, nullable=False)
    config_toml: Mapped[str] = mapped_column(Text(), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )
    error: Mapped[str | None] = mapped_column(Text())

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    jobs: Mapped[list[ExperimentJob]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    trials: Mapped[list[Trial]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_experiments_status", "status"),
        Index("idx_experiments_created", "created_at"),
    )

class ExperimentJob(Base):
    __tablename__ = "experiment_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("experiments.experiment_id", ondelete="CASCADE"), nullable=False
    )
    job_type: Mapped[ExperimentJobType] = mapped_column(
        SAEnum(ExperimentJobType, name="experiment_job_type"),
        nullable=False,
    )
    config_path: Mapped[str] = mapped_column(Text(), nullable=False)
    config_overrides: Mapped[dict[str, Any]] = mapped_column(
        MutableDict.as_mutable(JSON), default=dict, nullable=False
    )
    status: Mapped[ExperimentJobStatus] = mapped_column(
        SAEnum(ExperimentJobStatus, name="experiment_job_status"),
        default=ExperimentJobStatus.QUEUED,
        nullable=False,
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(128))
    backend_job_id: Mapped[str | None] = mapped_column(String(128))
    result: Mapped[dict[str, Any] | None] = mapped_column(MutableDict.as_mutable(JSON))
    status_json: Mapped[dict[str, Any] | None] = mapped_column(MutableDict.as_mutable(JSON))
    error: Mapped[str | None] = mapped_column(Text())

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    experiment: Mapped[Experiment] = relationship(back_populates="jobs")
    trials: Mapped[list[Trial]] = relationship(back_populates="job")
    execution_logs: Mapped[list[JobExecutionLog]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_experiment_jobs_status", "status"),
        Index("idx_experiment_jobs_experiment", "experiment_id"),
        Index("idx_experiment_jobs_celery_task", "celery_task_id"),
    )


class Trial(Base):
    __tablename__ = "trials"

    trial_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("experiments.experiment_id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("experiment_jobs.job_id", ondelete="SET NULL")
    )
    trial_number: Mapped[int | None] = mapped_column(Integer)
    system_name: Mapped[str | None] = mapped_column(String(128))
    prompt_id: Mapped[str | None] = mapped_column(String(128))
    session_id: Mapped[str | None] = mapped_column(String(128))
    trace_stored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    reward: Mapped[float | None] = mapped_column(Float)
    aggregate_score: Mapped[float | None] = mapped_column(Float)
    rewards: Mapped[dict[str, Any] | None] = mapped_column(MutableDict.as_mutable(JSON))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        MutableDict.as_mutable(JSON),
        default=dict,
        nullable=False,
    )
    status: Mapped[TrialStatus] = mapped_column(
        SAEnum(TrialStatus, name="experiment_trial_status"),
        default=TrialStatus.PENDING,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    experiment: Mapped[Experiment] = relationship(back_populates="trials")
    job: Mapped[ExperimentJob | None] = relationship(back_populates="trials")

    __table_args__ = (
        Index("idx_trials_experiment", "experiment_id"),
        Index("idx_trials_job", "job_id"),
        Index("idx_trials_status", "status"),
        Index("idx_trials_reward", "reward"),
        Index("idx_trials_aggregate_score", "aggregate_score"),
        Index("idx_trials_experiment_trial_number", "experiment_id", "trial_number"),
        UniqueConstraint("experiment_id", "trial_number", name="uq_trial_number_per_experiment"),
    )


class JobExecutionLog(Base):
    """Detailed execution logs for job subprocess runs.
    
    Stores full stdout, stderr, command, and environment info for ALL jobs (successful and failed).
    This allows querying failures directly from the database without guessing.
    """
    __tablename__ = "job_execution_logs"

    log_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("experiment_jobs.job_id", ondelete="CASCADE"), nullable=False
    )
    
    # Execution details
    command: Mapped[str] = mapped_column(Text(), nullable=False)  # Full command executed
    working_directory: Mapped[str] = mapped_column(Text(), nullable=False)
    returncode: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Output (stored as Text to handle large outputs)
    stdout: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    stderr: Mapped[str] = mapped_column(Text(), nullable=False, default="")
    
    # Environment info (for debugging)
    python_executable: Mapped[str | None] = mapped_column(String(255))
    environment_keys: Mapped[list[str] | None] = mapped_column(JSON)  # List of env var keys present
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    
    # Relationship
    job: Mapped[ExperimentJob] = relationship(back_populates="execution_logs")

    __table_args__ = (
        Index("idx_job_execution_logs_job", "job_id"),
        Index("idx_job_execution_logs_returncode", "returncode"),
        Index("idx_job_execution_logs_created", "created_at"),
    )
