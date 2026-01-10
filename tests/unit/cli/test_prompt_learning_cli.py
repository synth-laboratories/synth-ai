"""Unit tests for prompt learning CLI functionality."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from synth_ai.cli.train import (
    _DEFAULT_PROMPT_LEARNING_HIDDEN_EVENTS,
    _save_prompt_learning_results_locally,
    handle_prompt_learning,
)
from synth_ai.sdk.streaming import CLIHandler, StreamConfig, StreamEndpoints, StreamType

pytestmark = pytest.mark.unit


class TestPromptLearningCLIConfiguration:
    """Test CLI configuration for prompt learning."""

    def test_hidden_events_constant(self) -> None:
        """Test that hidden events constant is defined correctly."""
        assert "prompt.learning.policy.tokens" in _DEFAULT_PROMPT_LEARNING_HIDDEN_EVENTS

    def test_cli_handler_with_hidden_events(self) -> None:
        """Test CLIHandler configuration with prompt learning hidden events."""
        handler = CLIHandler(
            hidden_event_types=_DEFAULT_PROMPT_LEARNING_HIDDEN_EVENTS,
            hidden_event_substrings={"modal", "hatchet"},
        )

        assert "prompt.learning.policy.tokens" in handler._hidden_event_types
        assert "modal" in handler._hidden_event_substrings
        assert "hatchet" in handler._hidden_event_substrings

    def test_stream_config_for_prompt_learning_cli(self) -> None:
        """Test StreamConfig for prompt learning in CLI mode."""
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            metric_names={"gepa.transformation.mean_score"},
        )

        assert StreamType.METRICS in config.enabled_streams
        assert "gepa.transformation.mean_score" in config.metric_names

    def test_stream_config_for_prompt_learning_chart(self) -> None:
        """Test StreamConfig for prompt learning in chart mode."""
        config = StreamConfig(
            enabled_streams={StreamType.STATUS, StreamType.EVENTS, StreamType.METRICS},
            event_types={
                "prompt.learning.progress",
                "prompt.learning.gepa.start",
                "prompt.learning.gepa.complete",
            },
            metric_names={"gepa.transformation.mean_score"},
        )

        assert StreamType.METRICS in config.enabled_streams
        assert "prompt.learning.progress" in config.event_types
        assert "gepa.transformation.mean_score" in config.metric_names


class TestSavePromptLearningResults:
    """Test results file saving functionality."""

    def test_save_results_creates_file(self, tmp_path: Path) -> None:
        """Test that results file is created and saved."""
        job_id = "pl_test123"
        config_path = tmp_path / "test.toml"
        config_path.write_text('[prompt_learning]\nalgorithm = "gepa"')

        # Mock HTTP client
        events_response = {
            "events": [
                {
                    "seq": 1,
                    "type": "prompt.learning.gepa.start",
                    "message": "Starting GEPA",
                    "created_at": "2024-11-04T20:30:00Z",
                },
                {
                    "seq": 2,
                    "type": "prompt.learning.best.prompt",
                    "message": "Best prompt",
                    "data": {
                        "best_score": 0.5714,
                        "best_prompt": {
                            "sections": [
                                {"role": "system", "content": "You are helpful"},
                                {"role": "user", "content": "{query}"},
                            ]
                        },
                    },
                    "created_at": "2024-11-04T20:30:00Z",
                },
                {
                    "seq": 3,
                    "type": "prompt.learning.final.results",
                    "message": "Final results",
                    "data": {
                        "attempted_candidates": [
                            {"accuracy": 0.4, "prompt_length": 100},
                        ],
                        "optimized_candidates": [
                            {"score": {"accuracy": 0.5714, "prompt_length": 105}},
                        ],
                    },
                    "created_at": "2024-11-04T20:30:00Z",
                },
            ]
        }

        with patch("synth_ai.cli.train.http_get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = events_response

            # Call the function
            _save_prompt_learning_results_locally(
                backend_base="http://localhost:8000/api",
                api_key="sk-test",
                job_id=job_id,
                config_path=config_path,
                results_folder=tmp_path / "results",
            )

        # Check that results directory was created
        results_dir = config_path.parent / "results"
        assert results_dir.exists()

        # Check that results file was created
        result_files = list(results_dir.glob(f"*{job_id}*.txt"))
        assert len(result_files) > 0

    def test_save_results_handles_missing_events(self, tmp_path: Path) -> None:
        """Test that save_results handles missing events gracefully."""
        job_id = "pl_test456"
        config_path = tmp_path / "test.toml"
        config_path.write_text('[prompt_learning]\nalgorithm = "gepa"')

        with patch("synth_ai.cli.train.http_get") as mock_get:
            mock_get.return_value.status_code = 404

            with patch("synth_ai.cli.train.Path") as mock_path:
                mock_config_dir = MagicMock()
                mock_config_dir.__truediv__ = lambda self, other: tmp_path / other
                mock_config_dir.__str__ = lambda self: str(tmp_path)
                mock_path.return_value = mock_config_dir
                mock_path.cwd.return_value = tmp_path

                # Should not raise exception
                _save_prompt_learning_results_locally(
                    backend_base="http://localhost:8000/api",
                    api_key="sk-test",
                    job_id=job_id,
                    config_path=config_path,
                    results_folder=tmp_path / "results",
                )

    def test_save_results_extracts_baseline_score(self, tmp_path: Path) -> None:
        """Test that baseline score is extracted from events."""
        job_id = "pl_test789"
        config_path = tmp_path / "test.toml"
        config_path.write_text('[prompt_learning]\nalgorithm = "gepa"')

        events_response = {
            "events": [
                {
                    "seq": 1,
                    "type": "prompt.learning.validation.scored",
                    "message": "baseline val_accuracy=0.400 (N=50)",
                    "data": {"accuracy": 0.4, "n": 50},
                    "created_at": "2024-11-04T20:30:00Z",
                },
                {
                    "seq": 2,
                    "type": "prompt.learning.best.prompt",
                    "message": "Best prompt",
                    "data": {"score": 0.5714},
                    "created_at": "2024-11-04T20:30:00Z",
                },
            ]
        }

        with patch("synth_ai.cli.train.http_get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = events_response

            _save_prompt_learning_results_locally(
                backend_base="http://localhost:8000/api",
                api_key="sk-test",
                job_id=job_id,
                config_path=config_path,
                results_folder=tmp_path / "results",
            )

        # Check that results file contains baseline score
        results_dir = config_path.parent / "results"
        if results_dir.exists():
            result_files = list(results_dir.glob(f"*{job_id}*.txt"))
            if result_files:
                content = result_files[0].read_text()
                assert "baseline" in content.lower() or "0.4" in content


class TestHandlePromptLearning:
    """Test handle_prompt_learning function."""

    def test_handle_prompt_learning_creates_job(self) -> None:
        """Test that handle_prompt_learning creates a job correctly."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
task_app_api_key = "test-key"
results_folder = "results"

[prompt_learning.gepa]
num_generations = 5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            config_path = Path(f.name)

        try:
            with (
                patch("synth_ai.cli.train.build_prompt_learning_payload") as mock_build,
                patch("synth_ai.cli.train.check_local_api_health") as mock_health,
                patch("synth_ai.cli.train.http_post") as mock_post,
                patch("synth_ai.cli.train.JobStreamer") as mock_streamer,
                patch.dict("os.environ", {"ENVIRONMENT_API_KEY": "env-key"}),
            ):
                mock_build.return_value.task_url = "http://localhost:8001"
                mock_build.return_value.payload = {
                    "algorithm": "gepa",
                    "config_body": {"prompt_learning": {"algorithm": "gepa"}},
                }
                mock_health.return_value.ok = True
                mock_post.return_value.status_code = 201
                mock_post.return_value.json.return_value = {"job_id": "pl_test123"}

                async def mock_stream_until_terminal():
                    return {"status": "succeeded"}

                mock_streamer_instance = MagicMock()
                mock_streamer_instance.stream_until_terminal = mock_stream_until_terminal
                mock_streamer.return_value = mock_streamer_instance

                handle_prompt_learning(
                    cfg_path=config_path,
                    backend_base="http://localhost:8000/api",
                    synth_key="sk-test",
                    task_url_override=None,
                    allow_experimental=None,
                    dry_run=False,
                    poll=True,
                    poll_timeout=60.0,
                    poll_interval=1.0,
                    stream_format="cli",
                )

                # Verify job was created
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                assert "/prompt-learning/online/jobs" in call_args[0][0]

                # Verify streamer was configured correctly
                mock_streamer.assert_called_once()
                streamer_call = mock_streamer.call_args
                assert streamer_call.kwargs["endpoints"] == StreamEndpoints.prompt_learning(
                    "pl_test123"
                )
                assert StreamType.METRICS in streamer_call.kwargs["config"].enabled_streams

        finally:
            config_path.unlink()

    def test_handle_prompt_learning_no_poll(self) -> None:
        """Test handle_prompt_learning with polling disabled."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
task_app_api_key = "test-key"
results_folder = "results"

[prompt_learning.gepa]
num_generations = 5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            config_path = Path(f.name)

        try:
            with (
                patch("synth_ai.cli.train.build_prompt_learning_payload") as mock_build,
                patch("synth_ai.cli.train.check_local_api_health") as mock_health,
                patch("synth_ai.cli.train.http_post") as mock_post,
                patch("synth_ai.cli.train.JobStreamer") as mock_streamer,
                patch.dict("os.environ", {"ENVIRONMENT_API_KEY": "env-key"}),
            ):
                mock_build.return_value.task_url = "http://localhost:8001"
                mock_build.return_value.payload = {"algorithm": "gepa"}
                mock_health.return_value.ok = True
                mock_post.return_value.status_code = 201
                mock_post.return_value.json.return_value = {"job_id": "pl_test123"}

                handle_prompt_learning(
                    cfg_path=config_path,
                    backend_base="http://localhost:8000/api",
                    synth_key="sk-test",
                    task_url_override=None,
                    allow_experimental=None,
                    dry_run=False,
                    poll=False,
                    poll_timeout=60.0,
                    poll_interval=1.0,
                    stream_format="cli",
                )

                # Verify streamer was not called
                mock_streamer.assert_not_called()

        finally:
            config_path.unlink()

    def test_handle_prompt_learning_health_check_failure(self) -> None:
        """Test that health check failure raises exception."""
        toml_content = """
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://localhost:8001"
task_app_api_key = "test-key"
results_folder = "results"

[prompt_learning.gepa]
num_generations = 5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            config_path = Path(f.name)

        try:
            with (
                patch("synth_ai.cli.train.build_prompt_learning_payload") as mock_build,
                patch("synth_ai.cli.train.check_local_api_health") as mock_health,
                patch.dict("os.environ", {"ENVIRONMENT_API_KEY": "env-key"}),
            ):
                mock_build.return_value.task_url = "http://localhost:8001"
                mock_health.return_value.ok = False
                mock_health.return_value.detail = "Connection refused"

                from click import ClickException

                with pytest.raises(ClickException, match="health check"):
                    handle_prompt_learning(
                        cfg_path=config_path,
                        backend_base="http://localhost:8000/api",
                        synth_key="sk-test",
                        task_url_override=None,
                        allow_experimental=None,
                        dry_run=False,
                        poll=False,
                        poll_timeout=60.0,
                        poll_interval=1.0,
                        stream_format="cli",
                    )

        finally:
            config_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
