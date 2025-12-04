"""Pydantic schemas shared between CLI, worker, and API surfaces."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from .models import (
    Experiment,
    ExperimentJob,
    ExperimentJobStatus,
    ExperimentJobType,
    ExperimentStatus,
    Trial,
    TrialStatus,
)


class ExperimentJobSpec(BaseModel):
    """Job specification supplied when submitting an experiment."""

    job_type: ExperimentJobType = Field(
        default=ExperimentJobType.GEPA,
        validation_alias=AliasChoices("job_type", "type"),
        serialization_alias="job_type",
    )
    config_path: str
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class ExperimentSubmitRequest(BaseModel):
    """Submission payload for a new experiment."""

    name: str
    description: str | None = None
    parallelism: int = Field(default=3, ge=1, le=8)
    jobs: list[ExperimentJobSpec]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("jobs")
    @classmethod
    def _ensure_jobs(cls, value: list[ExperimentJobSpec]) -> list[ExperimentJobSpec]:
        if not value:
            raise ValueError("At least one job specification is required.")
        return value


class ExperimentJobSummary(BaseModel):
    """Serializable summary of a job."""

    job_id: str
    job_type: ExperimentJobType
    status: ExperimentJobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    backend_job_id: str | None = None
    error: str | None = None

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    @classmethod
    def from_job(cls, job: ExperimentJob) -> ExperimentJobSummary:
        return cls(
            job_id=job.job_id,
            job_type=ExperimentJobType(job.job_type),
            status=ExperimentJobStatus(job.status),
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            backend_job_id=job.backend_job_id,
            error=job.error,
        )


class TrialSummary(BaseModel):
    """Serializable trial summary."""

    trial_id: str
    trial_number: int | None
    system_name: str | None
    reward: float | None
    aggregate_score: float | None
    status: TrialStatus

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    @classmethod
    def from_trial(cls, trial: Trial) -> TrialSummary:
        return cls(
            trial_id=trial.trial_id,
            trial_number=trial.trial_number,
            system_name=trial.system_name,
            reward=trial.reward,
            aggregate_score=trial.aggregate_score,
            status=TrialStatus(trial.status),
        )


class ExperimentSummary(BaseModel):
    """Serializable experiment summary for CLI dashboards."""

    experiment_id: str
    name: str
    status: ExperimentStatus
    description: str | None = None
    parallelism_limit: int
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    job_count: int = 0
    trial_count: int = 0

    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

    @classmethod
    def from_experiment(cls, experiment: Experiment) -> ExperimentSummary:
        return cls(
            experiment_id=experiment.experiment_id,
            name=experiment.name,
            status=ExperimentStatus(experiment.status),
            description=experiment.description,
            parallelism_limit=experiment.parallelism_limit,
            created_at=experiment.created_at,
            started_at=experiment.started_at,
            completed_at=experiment.completed_at,
            job_count=len(experiment.jobs),
            trial_count=len(experiment.trials),
        )
