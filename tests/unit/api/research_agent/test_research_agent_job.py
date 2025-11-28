"""Unit tests for Research Agent SDK."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from synth_ai.sdk.api.research_agent import (
    ResearchAgentJob,
    ResearchAgentJobConfig,
    ResearchConfig,
    DatasetSource,
    OptimizationTool,
    MIPROConfig,
    GEPAConfig,
    PermittedModelsConfig,
    PermittedModel,
    ModelProvider,
)


class TestResearchConfig:
    """Tests for ResearchConfig."""

    def test_basic_config(self) -> None:
        """Test creating a basic ResearchConfig."""
        config = ResearchConfig(
            task_description="Test task",
            tools=[OptimizationTool.MIPRO],
        )
        assert config.task_description == "Test task"
        assert config.tools == [OptimizationTool.MIPRO]
        assert config.primary_metric == "accuracy"
        assert config.num_iterations == 10

    def test_config_with_datasets(self) -> None:
        """Test config with datasets."""
        config = ResearchConfig(
            task_description="Test task",
            datasets=[
                DatasetSource(
                    source_type="huggingface",
                    hf_repo_id="PolyAI/banking77",
                    hf_split="train",
                ),
            ],
        )
        assert len(config.datasets) == 1
        assert config.datasets[0].source_type == "huggingface"
        assert config.datasets[0].hf_repo_id == "PolyAI/banking77"

    def test_config_to_dict(self) -> None:
        """Test serialization to dict."""
        config = ResearchConfig(
            task_description="Optimize for accuracy",
            tools=[OptimizationTool.MIPRO, OptimizationTool.GEPA],
            datasets=[
                DatasetSource(
                    source_type="huggingface",
                    hf_repo_id="PolyAI/banking77",
                )
            ],
            num_iterations=15,
            mipro_config=MIPROConfig(
                meta_model="gpt-4o",
                num_trials=20,
            ),
        )
        d = config.to_dict()
        assert d["task_description"] == "Optimize for accuracy"
        assert d["tools"] == ["mipro", "gepa"]
        assert len(d["datasets"]) == 1
        assert d["datasets"][0]["hf_repo_id"] == "PolyAI/banking77"
        assert d["num_iterations"] == 15
        assert d["mipro_config"]["meta_model"] == "gpt-4o"
        assert d["mipro_config"]["num_trials"] == 20


class TestDatasetSource:
    """Tests for DatasetSource."""

    def test_huggingface_source(self) -> None:
        """Test HuggingFace dataset source."""
        ds = DatasetSource(
            source_type="huggingface",
            hf_repo_id="PolyAI/banking77",
            hf_split="train",
            description="Banking intents",
        )
        d = ds.to_dict()
        assert d["source_type"] == "huggingface"
        assert d["hf_repo_id"] == "PolyAI/banking77"
        assert d["hf_split"] == "train"
        assert d["description"] == "Banking intents"

    def test_upload_source(self) -> None:
        """Test uploaded file source."""
        ds = DatasetSource(
            source_type="upload",
            file_ids=["file_123", "file_456"],
        )
        d = ds.to_dict()
        assert d["source_type"] == "upload"
        assert d["file_ids"] == ["file_123", "file_456"]

    def test_inline_source(self) -> None:
        """Test inline data source."""
        ds = DatasetSource(
            source_type="inline",
            inline_data={"data.jsonl": '{"input": "test"}'},
        )
        d = ds.to_dict()
        assert d["source_type"] == "inline"
        assert d["inline_data"]["data.jsonl"] == '{"input": "test"}'


class TestMIPROConfig:
    """Tests for MIPROConfig."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = MIPROConfig()
        assert config.meta_model == "llama-3.3-70b-versatile"
        assert config.meta_provider == ModelProvider.GROQ
        assert config.num_trials == 10

    def test_to_dict(self) -> None:
        """Test serialization."""
        config = MIPROConfig(
            meta_model="gpt-4o",
            meta_provider=ModelProvider.OPENAI,
            num_trials=20,
            proposer_effort="HIGH",
        )
        d = config.to_dict()
        assert d["meta_model"] == "gpt-4o"
        assert d["meta_provider"] == "openai"
        assert d["num_trials"] == 20
        assert d["proposer_effort"] == "HIGH"


class TestGEPAConfig:
    """Tests for GEPAConfig."""

    def test_defaults(self) -> None:
        """Test default values."""
        config = GEPAConfig()
        assert config.mutation_model == "openai/gpt-oss-120b"
        assert config.population_size == 20
        assert config.proposer_type == "dspy"

    def test_to_dict_with_spec(self) -> None:
        """Test serialization with spec file."""
        config = GEPAConfig(
            proposer_type="spec",
            spec_path="gepa_spec.json",
        )
        d = config.to_dict()
        assert d["proposer_type"] == "spec"
        assert d["spec_path"] == "gepa_spec.json"


class TestPermittedModelsConfig:
    """Tests for PermittedModelsConfig."""

    def test_with_models(self) -> None:
        """Test config with models."""
        config = PermittedModelsConfig(
            models=[
                PermittedModel(model="gpt-4o-mini", provider=ModelProvider.OPENAI),
                PermittedModel(model="llama-3.3-70b-versatile", provider=ModelProvider.GROQ),
            ],
            default_temperature=0.5,
        )
        d = config.to_dict()
        assert len(d["models"]) == 2
        assert d["models"][0]["model"] == "gpt-4o-mini"
        assert d["models"][0]["provider"] == "openai"
        assert d["default_temperature"] == 0.5


class TestResearchAgentJobConfig:
    """Tests for ResearchAgentJobConfig."""

    def test_config_validation_missing_repo_and_inline_files(self) -> None:
        """Test that config requires either repo_url or inline_files."""
        research = ResearchConfig(task_description="Test")
        with pytest.raises(ValueError, match="Either repo_url or inline_files"):
            ResearchAgentJobConfig(
                research=research,
                repo_url="",
                inline_files=None,
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )

    def test_config_validation_missing_api_key(self) -> None:
        """Test that missing api_key raises ValueError."""
        research = ResearchConfig(task_description="Test")
        with patch.dict(os.environ, {}, clear=True), pytest.raises(
            ValueError, match="api_key is required"
        ):
            ResearchAgentJobConfig(
                research=research,
                repo_url="https://github.com/test/repo",
                backend_url="https://api.usesynth.ai",
                api_key="",
            )

    def test_config_auto_resolve_api_key(self) -> None:
        """Test that api_key is resolved from environment."""
        research = ResearchConfig(task_description="Test")
        with patch.dict(os.environ, {"SYNTH_API_KEY": "env-key"}):
            config = ResearchAgentJobConfig(
                research=research,
                repo_url="https://github.com/test/repo",
                backend_url="https://api.usesynth.ai",
                api_key="",
            )
            assert config.api_key == "env-key"

    def test_config_auto_resolve_backend_url(self) -> None:
        """Test that backend_url defaults to production."""
        research = ResearchConfig(task_description="Test")
        with patch.dict(os.environ, {"SYNTH_API_KEY": "test-key"}):
            config = ResearchAgentJobConfig(
                research=research,
                repo_url="https://github.com/test/repo",
                backend_url="",
                api_key="test-key",
            )
            assert config.backend_url == "https://api.usesynth.ai"

    def test_config_with_inline_files(self) -> None:
        """Test config with inline_files instead of repo_url."""
        research = ResearchConfig(task_description="Test")
        config = ResearchAgentJobConfig(
            research=research,
            inline_files={"pipeline.py": "def main(): pass"},
            backend_url="https://api.usesynth.ai",
            api_key="test-key",
        )
        assert config.inline_files == {"pipeline.py": "def main(): pass"}
        assert config.repo_url == ""

    def test_config_with_spend_limits(self) -> None:
        """Test config with spend limits."""
        research = ResearchConfig(task_description="Test")
        config = ResearchAgentJobConfig(
            research=research,
            repo_url="https://github.com/test/repo",
            max_agent_spend_usd=25.0,
            max_synth_spend_usd=150.0,
            backend_url="https://api.usesynth.ai",
            api_key="test-key",
        )
        assert config.max_agent_spend_usd == 25.0
        assert config.max_synth_spend_usd == 150.0

    def test_config_with_reasoning_effort(self) -> None:
        """Test config with reasoning effort."""
        research = ResearchConfig(task_description="Test")
        config = ResearchAgentJobConfig(
            research=research,
            repo_url="https://github.com/test/repo",
            reasoning_effort="medium",
            backend_url="https://api.usesynth.ai",
            api_key="test-key",
        )
        assert config.reasoning_effort == "medium"

    def test_from_toml_missing_file(self, tmp_path: Path) -> None:
        """Test that from_toml raises FileNotFoundError for missing file."""
        fake_path = tmp_path / "nonexistent.toml"
        with pytest.raises(FileNotFoundError):
            ResearchAgentJobConfig.from_toml(fake_path)

    def test_from_toml_missing_section(self, tmp_path: Path) -> None:
        """Test that from_toml raises ValueError if [research_agent] section missing."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[other_section]\nkey = 'value'\n")

        with pytest.raises(ValueError, match="must have \\[research_agent\\] section"):
            ResearchAgentJobConfig.from_toml(config_file)

    def test_from_toml_missing_research(self, tmp_path: Path) -> None:
        """Test that from_toml raises ValueError if research config missing."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[research_agent]\nrepo_url = 'https://github.com/test/repo'\n")

        with pytest.raises(ValueError, match="research_agent.research config is required"):
            ResearchAgentJobConfig.from_toml(config_file)

    def test_from_toml_valid_config(self, tmp_path: Path) -> None:
        """Test loading valid TOML config."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""[research_agent]
repo_url = "https://github.com/test/repo"
repo_branch = "main"
backend = "daytona"
model = "gpt-5.1-codex-mini"
max_agent_spend_usd = 25.0
max_synth_spend_usd = 150.0
reasoning_effort = "medium"

[research_agent.research]
task_description = "Optimize for accuracy"
tools = ["mipro"]
primary_metric = "accuracy"
num_iterations = 15

[[research_agent.research.datasets]]
source_type = "huggingface"
hf_repo_id = "PolyAI/banking77"

[research_agent.research.mipro_config]
meta_model = "llama-3.3-70b-versatile"
num_trials = 20
""")

        config = ResearchAgentJobConfig.from_toml(config_file)
        assert config.repo_url == "https://github.com/test/repo"
        assert config.repo_branch == "main"
        assert config.backend == "daytona"
        assert config.model == "gpt-5.1-codex-mini"
        assert config.max_agent_spend_usd == 25.0
        assert config.max_synth_spend_usd == 150.0
        assert config.reasoning_effort == "medium"
        assert config.research.task_description == "Optimize for accuracy"
        assert config.research.tools == [OptimizationTool.MIPRO]
        assert config.research.num_iterations == 15
        assert len(config.research.datasets) == 1
        assert config.research.datasets[0].hf_repo_id == "PolyAI/banking77"
        assert config.research.mipro_config is not None
        assert config.research.mipro_config.meta_model == "llama-3.3-70b-versatile"
        assert config.research.mipro_config.num_trials == 20


class TestResearchAgentJob:
    """Tests for ResearchAgentJob."""

    def test_from_research_config(self) -> None:
        """Test creating job from ResearchConfig."""
        research = ResearchConfig(
            task_description="Optimize prompt",
            tools=[OptimizationTool.MIPRO],
        )
        job = ResearchAgentJob.from_research_config(
            research=research,
            repo_url="https://github.com/test/repo",
            model="gpt-5.1-codex-mini",
            max_agent_spend_usd=25.0,
            backend_url="https://api.usesynth.ai",
            api_key="test-key",
        )
        assert job.config.research.task_description == "Optimize prompt"
        assert job.config.model == "gpt-5.1-codex-mini"
        assert job.config.max_agent_spend_usd == 25.0

    def test_from_id(self) -> None:
        """Test creating job from existing job ID."""
        with patch.dict(
            os.environ,
            {
                "SYNTH_API_KEY": "test-key",
            },
            clear=True,
        ):
            job = ResearchAgentJob.from_id(
                job_id="ra_1234567890",
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
            assert job.job_id == "ra_1234567890"
            assert job.config.backend_url == "https://api.usesynth.ai"

    def test_submit_already_submitted(self) -> None:
        """Test that submit() raises error if already submitted."""
        research = ResearchConfig(task_description="Test")
        config = ResearchAgentJobConfig(
            research=research,
            repo_url="https://github.com/test/repo",
            backend_url="https://api.usesynth.ai",
            api_key="test-key",
        )
        job = ResearchAgentJob(config=config)
        job._job_id = "ra_existing"

        with pytest.raises(RuntimeError, match="already submitted"):
            job.submit()

    def test_poll_until_complete_requires_submission(self) -> None:
        """Test that poll_until_complete() requires job to be submitted."""
        research = ResearchConfig(task_description="Test")
        config = ResearchAgentJobConfig(
            research=research,
            repo_url="https://github.com/test/repo",
            backend_url="https://api.usesynth.ai",
            api_key="test-key",
        )
        job = ResearchAgentJob(config=config)
        with pytest.raises(RuntimeError, match="not submitted yet"):
            job.poll_until_complete()

    def test_get_status_requires_submission(self) -> None:
        """Test that get_status() requires job to be submitted."""
        research = ResearchConfig(task_description="Test")
        config = ResearchAgentJobConfig(
            research=research,
            repo_url="https://github.com/test/repo",
            backend_url="https://api.usesynth.ai",
            api_key="test-key",
        )
        job = ResearchAgentJob(config=config)
        with pytest.raises(RuntimeError, match="not submitted yet"):
            job.get_status()

    def test_submit_gepa_not_implemented(self) -> None:
        """Test that GEPA raises NotImplementedError on submit."""
        research = ResearchConfig(
            task_description="Test",
            tools=[OptimizationTool.GEPA],
        )
        config = ResearchAgentJobConfig(
            research=research,
            repo_url="https://github.com/test/repo",
            backend_url="https://api.usesynth.ai",
            api_key="test-key",
        )
        job = ResearchAgentJob(config=config)
        with pytest.raises(NotImplementedError, match="GEPA optimization is not yet fully supported"):
            job.submit()

    def test_submit_gepa_with_mipro_not_implemented(self) -> None:
        """Test that using both MIPRO and GEPA raises NotImplementedError."""
        research = ResearchConfig(
            task_description="Test",
            tools=[OptimizationTool.MIPRO, OptimizationTool.GEPA],
        )
        config = ResearchAgentJobConfig(
            research=research,
            repo_url="https://github.com/test/repo",
            backend_url="https://api.usesynth.ai",
            api_key="test-key",
        )
        job = ResearchAgentJob(config=config)
        with pytest.raises(NotImplementedError, match="GEPA optimization is not yet fully supported"):
            job.submit()
