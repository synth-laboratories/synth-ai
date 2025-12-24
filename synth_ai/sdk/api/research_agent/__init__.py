"""Research Agent SDK models and job helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import os
import tomllib
from typing import Any


class OptimizationTool(str, Enum):
    MIPRO = "mipro"
    GEPA = "gepa"


class ModelProvider(str, Enum):
    OPENAI = "openai"
    GROQ = "groq"


@dataclass
class PermittedModel:
    model: str
    provider: ModelProvider

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "provider": self.provider.value}


@dataclass
class PermittedModelsConfig:
    models: list[PermittedModel] = field(default_factory=list)
    default_temperature: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {"models": [model.to_dict() for model in self.models]}
        if self.default_temperature is not None:
            data["default_temperature"] = self.default_temperature
        return data


@dataclass
class DatasetSource:
    source_type: str
    hf_repo_id: str | None = None
    hf_split: str | None = None
    description: str | None = None
    file_ids: list[str] | None = None
    inline_data: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"source_type": self.source_type}
        if self.hf_repo_id:
            data["hf_repo_id"] = self.hf_repo_id
        if self.hf_split:
            data["hf_split"] = self.hf_split
        if self.description:
            data["description"] = self.description
        if self.file_ids is not None:
            data["file_ids"] = self.file_ids
        if self.inline_data is not None:
            data["inline_data"] = self.inline_data
        return data


@dataclass
class MIPROConfig:
    meta_model: str = "llama-3.3-70b-versatile"
    meta_provider: ModelProvider = ModelProvider.GROQ
    num_trials: int = 10
    proposer_effort: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "meta_model": self.meta_model,
            "meta_provider": self.meta_provider.value,
            "num_trials": self.num_trials,
        }
        if self.proposer_effort is not None:
            data["proposer_effort"] = self.proposer_effort
        return data


@dataclass
class GEPAConfig:
    mutation_model: str = "openai/gpt-oss-120b"
    population_size: int = 20
    proposer_type: str = "dspy"
    spec_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "mutation_model": self.mutation_model,
            "population_size": self.population_size,
            "proposer_type": self.proposer_type,
        }
        if self.spec_path is not None:
            data["spec_path"] = self.spec_path
        return data


@dataclass
class ResearchConfig:
    task_description: str
    tools: list[OptimizationTool] = field(default_factory=list)
    datasets: list[DatasetSource] = field(default_factory=list)
    primary_metric: str = "accuracy"
    num_iterations: int = 10
    mipro_config: MIPROConfig | None = None
    gepa_config: GEPAConfig | None = None
    permitted_models: PermittedModelsConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "task_description": self.task_description,
            "tools": [tool.value for tool in self.tools],
            "primary_metric": self.primary_metric,
            "num_iterations": self.num_iterations,
        }
        if self.datasets:
            data["datasets"] = [ds.to_dict() for ds in self.datasets]
        if self.mipro_config is not None:
            data["mipro_config"] = self.mipro_config.to_dict()
        if self.gepa_config is not None:
            data["gepa_config"] = self.gepa_config.to_dict()
        if self.permitted_models is not None:
            data["permitted_models"] = self.permitted_models.to_dict()
        return data


@dataclass
class ResearchAgentJobConfig:
    research: ResearchConfig
    repo_url: str = ""
    repo_branch: str | None = None
    inline_files: dict[str, str] | None = None
    backend_url: str = ""
    api_key: str = ""
    allow_missing_api_key: bool = False
    backend: str | None = None
    model: str | None = None
    max_agent_spend_usd: float | None = None
    max_synth_spend_usd: float | None = None
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if not self.repo_url and not self.inline_files:
            raise ValueError("Either repo_url or inline_files must be provided")
        if not self.api_key:
            self.api_key = os.getenv("SYNTH_API_KEY", "").strip()
        if not self.api_key and not self.allow_missing_api_key:
            raise ValueError("api_key is required")
        if not self.backend_url:
            self.backend_url = "https://api.usesynth.ai"

    @classmethod
    def from_toml(cls, path: str | Path) -> "ResearchAgentJobConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        if "research_agent" not in data:
            raise ValueError("Config must have [research_agent] section")
        section = data["research_agent"]
        research_section = section.get("research")
        if research_section is None:
            raise ValueError("research_agent.research config is required")

        tools = [OptimizationTool(tool) for tool in research_section.get("tools", [])]
        datasets = [
            DatasetSource(
                source_type=ds.get("source_type", ""),
                hf_repo_id=ds.get("hf_repo_id"),
                hf_split=ds.get("hf_split"),
                description=ds.get("description"),
                file_ids=ds.get("file_ids"),
                inline_data=ds.get("inline_data"),
            )
            for ds in research_section.get("datasets", [])
        ]
        mipro_cfg = None
        if research_section.get("mipro_config"):
            cfg = research_section["mipro_config"]
            mipro_cfg = MIPROConfig(
                meta_model=cfg.get("meta_model", MIPROConfig.meta_model),
                meta_provider=ModelProvider(cfg.get("meta_provider", ModelProvider.GROQ.value)),
                num_trials=cfg.get("num_trials", MIPROConfig.num_trials),
                proposer_effort=cfg.get("proposer_effort"),
            )

        research = ResearchConfig(
            task_description=research_section.get("task_description", ""),
            tools=tools,
            datasets=datasets,
            primary_metric=research_section.get("primary_metric", "accuracy"),
            num_iterations=research_section.get("num_iterations", 10),
            mipro_config=mipro_cfg,
        )

        return cls(
            research=research,
            repo_url=section.get("repo_url", "") or "",
            repo_branch=section.get("repo_branch"),
            backend=section.get("backend"),
            model=section.get("model"),
            max_agent_spend_usd=section.get("max_agent_spend_usd"),
            max_synth_spend_usd=section.get("max_synth_spend_usd"),
            reasoning_effort=section.get("reasoning_effort"),
            backend_url=section.get("backend_url", ""),
            api_key=section.get("api_key", ""),
            allow_missing_api_key=True,
        )


class ResearchAgentJob:
    def __init__(self, *, config: ResearchAgentJobConfig) -> None:
        self.config = config
        self._job_id: str | None = None

    @property
    def job_id(self) -> str | None:
        return self._job_id

    @classmethod
    def from_research_config(
        cls,
        *,
        research: ResearchConfig,
        repo_url: str,
        backend_url: str,
        api_key: str,
        model: str | None = None,
        max_agent_spend_usd: float | None = None,
    ) -> "ResearchAgentJob":
        config = ResearchAgentJobConfig(
            research=research,
            repo_url=repo_url,
            backend_url=backend_url,
            api_key=api_key,
            model=model,
            max_agent_spend_usd=max_agent_spend_usd,
        )
        return cls(config=config)

    @classmethod
    def from_id(
        cls,
        *,
        job_id: str,
        backend_url: str,
        api_key: str,
    ) -> "ResearchAgentJob":
        research = ResearchConfig(task_description="Existing research job")
        config = ResearchAgentJobConfig(
            research=research,
            repo_url="existing",
            backend_url=backend_url,
            api_key=api_key,
        )
        job = cls(config=config)
        job._job_id = job_id
        return job

    def submit(self) -> str:
        if self._job_id is not None:
            raise RuntimeError("Job already submitted")
        if OptimizationTool.GEPA in self.config.research.tools:
            raise NotImplementedError("GEPA optimization is not yet fully supported")
        self._job_id = "ra_pending"
        return self._job_id

    def poll_until_complete(self) -> dict[str, Any]:
        if self._job_id is None:
            raise RuntimeError("Job not submitted yet")
        return {"job_id": self._job_id, "status": "submitted"}

    def get_status(self) -> dict[str, Any]:
        if self._job_id is None:
            raise RuntimeError("Job not submitted yet")
        return {"job_id": self._job_id, "status": "submitted"}


__all__ = [
    "ResearchAgentJob",
    "ResearchAgentJobConfig",
    "ResearchConfig",
    "DatasetSource",
    "OptimizationTool",
    "MIPROConfig",
    "GEPAConfig",
    "PermittedModelsConfig",
    "PermittedModel",
    "ModelProvider",
]
