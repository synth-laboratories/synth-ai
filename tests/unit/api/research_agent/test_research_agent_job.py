"""Unit tests for Research Agent SDK."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from synth_ai.sdk.api.research_agent.job import (
    ResearchAgentJob,
    ResearchAgentJobConfig,
)


class TestResearchAgentJobConfig:
    """Tests for ResearchAgentJobConfig."""
    
    def test_config_validation_missing_repo_and_inline_files(self) -> None:
        """Test that config requires either repo_url or inline_files."""
        with pytest.raises(ValueError, match="Either repo_url or inline_files"):
            ResearchAgentJobConfig(
                algorithm="scaffold_tuning",
                repo_url="",
                inline_files=None,
                backend_url="https://api.usesynth.ai",
                api_key="test-key",
            )
    
    def test_config_validation_missing_api_key(self) -> None:
        """Test that missing api_key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True), pytest.raises(
            ValueError, match="api_key is required"
        ):
            ResearchAgentJobConfig(
                algorithm="scaffold_tuning",
                repo_url="https://github.com/test/repo",
                backend_url="https://api.usesynth.ai",
                api_key="",
            )
    
    def test_config_auto_resolve_api_key(self) -> None:
        """Test that api_key is resolved from environment."""
        with patch.dict(os.environ, {"SYNTH_API_KEY": "env-key"}):
            config = ResearchAgentJobConfig(
                algorithm="scaffold_tuning",
                repo_url="https://github.com/test/repo",
                backend_url="https://api.usesynth.ai",
                api_key="",
            )
            assert config.api_key == "env-key"
    
    def test_config_auto_resolve_backend_url(self) -> None:
        """Test that backend_url defaults to production."""
        with patch.dict(os.environ, {"SYNTH_API_KEY": "test-key"}):
            config = ResearchAgentJobConfig(
                algorithm="scaffold_tuning",
                repo_url="https://github.com/test/repo",
                backend_url="",
                api_key="test-key",
            )
            assert config.backend_url == "https://api.usesynth.ai"
    
    def test_config_with_inline_files(self) -> None:
        """Test config with inline_files instead of repo_url."""
        config = ResearchAgentJobConfig(
            algorithm="scaffold_tuning",
            inline_files={"pipeline.py": "def main(): pass"},
            backend_url="https://api.usesynth.ai",
            api_key="test-key",
        )
        assert config.inline_files == {"pipeline.py": "def main(): pass"}
        assert config.repo_url == ""
    
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
    
    def test_from_toml_missing_algorithm(self, tmp_path: Path) -> None:
        """Test that from_toml raises ValueError if algorithm missing."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[research_agent]\nrepo_url = 'https://github.com/test/repo'\n")
        
        with pytest.raises(ValueError, match="research_agent.algorithm is required"):
            ResearchAgentJobConfig.from_toml(config_file)
    
    def test_from_toml_valid_config(self, tmp_path: Path) -> None:
        """Test loading valid TOML config."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""[research_agent]
algorithm = "scaffold_tuning"
repo_url = "https://github.com/test/repo"
repo_branch = "main"
backend = "daytona"
model = "gpt-4o"

[research_agent.scaffold_tuning]
objective.metric_name = "accuracy"
objective.max_iterations = 5
target_files = ["prompts/*.txt"]
""")
        
        config = ResearchAgentJobConfig.from_toml(config_file)
        assert config.algorithm == "scaffold_tuning"
        assert config.repo_url == "https://github.com/test/repo"
        assert config.repo_branch == "main"
        assert config.backend == "daytona"
        assert config.model == "gpt-4o"
        assert config.algorithm_config["objective"]["metric_name"] == "accuracy"
        assert config.algorithm_config["objective"]["max_iterations"] == 5


class TestResearchAgentJob:
    """Tests for ResearchAgentJob."""
    
    def test_from_config_missing_api_key(self, tmp_path: Path) -> None:
        """Test that from_config raises ValueError if API key missing."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""[research_agent]
algorithm = "scaffold_tuning"
repo_url = "https://github.com/test/repo"
""")
        
        with patch.dict(os.environ, {}, clear=True), pytest.raises(
            ValueError, match="api_key is required"
        ):
            ResearchAgentJob.from_config(
                config_path=config_file,
            )
    
    def test_from_config_resolves_env(self, tmp_path: Path) -> None:
        """Test that from_config resolves backend and API key from env."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""[research_agent]
algorithm = "scaffold_tuning"
repo_url = "https://github.com/test/repo"
""")
        
        with patch.dict(
            os.environ,
            {
                "SYNTH_API_KEY": "test-key",
            },
            clear=True,
        ):
            job = ResearchAgentJob.from_config(config_path=config_file)
            assert job.config.api_key == "test-key"
    
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
    
    def test_submit_requires_config(self) -> None:
        """Test that submit() requires config for new jobs."""
        # Create job from job_id (no config)
        job = ResearchAgentJob.from_id(
            job_id="ra_1234567890",
            backend_url="https://api.usesynth.ai",
            api_key="test-key",
        )
        # Should fail because we don't have proper config
        with pytest.raises((ValueError, RuntimeError), match="repo_url|inline_files|Cannot build"):
            job.submit()
    
    def test_submit_already_submitted(self, tmp_path: Path) -> None:
        """Test that submit() raises error if already submitted."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""[research_agent]
algorithm = "scaffold_tuning"
repo_url = "https://github.com/test/repo"
""")
        
        with patch.dict(os.environ, {"SYNTH_API_KEY": "test-key"}):
            job = ResearchAgentJob.from_config(config_path=config_file)
            job._job_id = "ra_existing"
            
            with pytest.raises(RuntimeError, match="already submitted"):
                job.submit()
    
    def test_poll_until_complete_requires_submission(self, tmp_path: Path) -> None:
        """Test that poll_until_complete() requires job to be submitted."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""[research_agent]
algorithm = "scaffold_tuning"
repo_url = "https://github.com/test/repo"
""")
        
        with patch.dict(os.environ, {"SYNTH_API_KEY": "test-key"}):
            job = ResearchAgentJob.from_config(config_path=config_file)
            with pytest.raises(RuntimeError, match="not yet submitted"):
                job.poll_until_complete()


