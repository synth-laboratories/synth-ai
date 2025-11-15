"""Unit tests for artifacts list command."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from synth_ai.cli.commands.artifacts.list import list_command


@pytest.fixture
def runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_artifacts_data() -> dict:
    """Mock artifacts list response."""
    return {
        "summary": {"total": 3, "models": 1, "prompts": 2},
        "models": [
            {
                "id": "ft:Qwen/Qwen3-0.6B:job_12345",
                "type": "ft",
                "base_model": "Qwen/Qwen3-0.6B",
                "job_id": "job_12345",
                "status": "succeeded",
                "created_at": "2025-01-01T00:00:00Z",
            }
        ],
        "prompts": [
            {
                "job_id": "pl_71c12c4c7c474c34",
                "algorithm": "gepa",
                "status": "succeeded",
                "best_score": 0.25,
                "created_at": "2025-01-01T00:00:00Z",
            },
            {
                "job_id": "pl_abc123",
                "algorithm": "mipro",
                "status": "succeeded",
                "best_score": 0.30,
                "created_at": "2025-01-02T00:00:00Z",
            },
        ],
    }


class TestListCommand:
    """Test list command CLI behavior."""

    @patch("synth_ai.cli.commands.artifacts.list.ArtifactsClient")
    def test_list_all_artifacts(
        self, mock_client_class: MagicMock, runner: CliRunner, mock_artifacts_data: dict
    ) -> None:
        """Test list command for all artifacts."""
        mock_client = AsyncMock()
        mock_client.list_artifacts = AsyncMock(return_value=mock_artifacts_data)
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            list_command,
            ["--api-key", "test_key"],
        )

        assert result.exit_code == 0
        # Check for either models or prompts section (depending on what's returned)
        output_lower = result.output.lower()
        assert ("models" in output_lower or "prompts" in output_lower or "optimized prompts" in output_lower)

    @patch("synth_ai.cli.commands.artifacts.list.ArtifactsClient")
    def test_list_models_only(
        self, mock_client_class: MagicMock, runner: CliRunner, mock_artifacts_data: dict
    ) -> None:
        """Test list command for models only."""
        mock_client = AsyncMock()
        models_data = {
            "summary": {"total": 1, "models": 1, "prompts": 0},
            "models": mock_artifacts_data["models"],
            "prompts": [],
        }
        mock_client.list_artifacts = AsyncMock(return_value=models_data)
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            list_command,
            ["--type", "models", "--api-key", "test_key"],
        )

        assert result.exit_code == 0
        mock_client.list_artifacts.assert_called_once_with(
            artifact_type="models", status="succeeded", limit=50
        )

    @patch("synth_ai.cli.commands.artifacts.list.ArtifactsClient")
    def test_list_prompts_only(
        self, mock_client_class: MagicMock, runner: CliRunner, mock_artifacts_data: dict
    ) -> None:
        """Test list command for prompts only."""
        mock_client = AsyncMock()
        prompts_data = {
            "summary": {"total": 2, "models": 0, "prompts": 2},
            "models": [],
            "prompts": mock_artifacts_data["prompts"],
        }
        mock_client.list_artifacts = AsyncMock(return_value=prompts_data)
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            list_command,
            ["--type", "prompts", "--api-key", "test_key"],
        )

        assert result.exit_code == 0
        mock_client.list_artifacts.assert_called_once_with(
            artifact_type="prompts", status="succeeded", limit=50
        )

    @patch("synth_ai.cli.commands.artifacts.list.ArtifactsClient")
    def test_list_json_format(
        self, mock_client_class: MagicMock, runner: CliRunner, mock_artifacts_data: dict
    ) -> None:
        """Test list command in JSON format."""
        mock_client = AsyncMock()
        mock_client.list_artifacts = AsyncMock(return_value=mock_artifacts_data)
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            list_command,
            ["--format", "json", "--api-key", "test_key"],
        )

        assert result.exit_code == 0
        import json

        output_data = json.loads(result.output)
        assert "summary" in output_data
        assert "models" in output_data
        assert "prompts" in output_data

    @patch("synth_ai.cli.commands.artifacts.list.ArtifactsClient")
    def test_list_with_limit(
        self, mock_client_class: MagicMock, runner: CliRunner, mock_artifacts_data: dict
    ) -> None:
        """Test list command with limit."""
        mock_client = AsyncMock()
        mock_client.list_artifacts = AsyncMock(return_value=mock_artifacts_data)
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            list_command,
            ["--limit", "10", "--api-key", "test_key"],
        )

        assert result.exit_code == 0
        mock_client.list_artifacts.assert_called_once_with(
            artifact_type=None, status="succeeded", limit=10
        )

    @patch("synth_ai.cli.commands.artifacts.list.ArtifactsClient")
    def test_list_with_status_filter(
        self, mock_client_class: MagicMock, runner: CliRunner, mock_artifacts_data: dict
    ) -> None:
        """Test list command with status filter."""
        mock_client = AsyncMock()
        mock_client.list_artifacts = AsyncMock(return_value=mock_artifacts_data)
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            list_command,
            ["--status", "running", "--api-key", "test_key"],
        )

        assert result.exit_code == 0
        mock_client.list_artifacts.assert_called_once_with(
            artifact_type=None, status="running", limit=50
        )

    @patch("synth_ai.cli.commands.artifacts.list.ArtifactsClient")
    def test_list_empty_results(
        self, mock_client_class: MagicMock, runner: CliRunner
    ) -> None:
        """Test list command with empty results."""
        mock_client = AsyncMock()
        empty_data = {
            "summary": {"total": 0, "models": 0, "prompts": 0},
            "models": [],
            "prompts": [],
        }
        mock_client.list_artifacts = AsyncMock(return_value=empty_data)
        mock_client_class.return_value = mock_client

        result = runner.invoke(
            list_command,
            ["--api-key", "test_key"],
        )

        assert result.exit_code == 0
        assert "No artifacts found" in result.output or "0" in result.output

