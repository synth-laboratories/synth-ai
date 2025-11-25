"""Integration tests for SFT (Supervised Fine-Tuning) SDK.

These tests validate the SDK components for SFT jobs:
1. Configuration validation
2. Dataset validation
3. Client functionality with mocks

Tests are marked with pytest markers for selective execution:
- @pytest.mark.integration: All integration tests
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.integration]


class TestSFTConfigValidation:
    """Test SFT configuration validation."""

    def test_validate_sft_config_requires_algorithm(self) -> None:
        """SFT config requires algorithm section."""
        from synth_ai.cli.commands.train.validation import validate_sft_config
        from synth_ai.cli.commands.train.errors import MissingAlgorithmError

        with pytest.raises(MissingAlgorithmError):
            validate_sft_config({})

    def test_validate_sft_config_requires_job_section(self) -> None:
        """SFT config requires job section."""
        from synth_ai.cli.commands.train.validation import validate_sft_config
        from synth_ai.cli.commands.train.errors import InvalidSFTConfigError

        with pytest.raises(InvalidSFTConfigError):
            validate_sft_config({"algorithm": {"variety": "fft"}})

    def test_validate_sft_config_requires_model(self) -> None:
        """SFT config requires model in job section."""
        from synth_ai.cli.commands.train.validation import validate_sft_config
        from synth_ai.cli.commands.train.errors import MissingModelError

        with pytest.raises(MissingModelError):
            validate_sft_config({
                "algorithm": {"variety": "fft"},
                "job": {"data": "/path/to/train.jsonl"},  # non-empty but no model
            })

    def test_validate_sft_config_requires_dataset(self) -> None:
        """SFT config requires dataset path."""
        from synth_ai.cli.commands.train.validation import validate_sft_config
        from synth_ai.cli.commands.train.errors import MissingDatasetError

        with pytest.raises(MissingDatasetError):
            validate_sft_config({
                "algorithm": {"variety": "fft"},
                "job": {"model": "Qwen/Qwen3-0.6B"},
            })

    def test_validate_sft_config_requires_compute(self) -> None:
        """SFT config requires compute section."""
        from synth_ai.cli.commands.train.validation import validate_sft_config
        from synth_ai.cli.commands.train.errors import MissingComputeError

        with pytest.raises(MissingComputeError):
            validate_sft_config({
                "algorithm": {"variety": "fft"},
                "job": {"model": "Qwen/Qwen3-0.6B", "data": "/path/to/data.jsonl"},
            })


class TestSFTDataValidation:
    """Test SFT training data validation."""

    @pytest.fixture
    def valid_sft_jsonl(self, tmp_path: Path) -> Path:
        """Create a valid SFT JSONL file."""
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        path = tmp_path / "valid.jsonl"
        path.write_text(json.dumps(data) + "\n")
        return path

    @pytest.fixture
    def empty_messages_jsonl(self, tmp_path: Path) -> Path:
        """Create a JSONL file with empty messages."""
        data = {"messages": []}
        path = tmp_path / "empty.jsonl"
        path.write_text(json.dumps(data) + "\n")
        return path

    @pytest.fixture
    def missing_messages_jsonl(self, tmp_path: Path) -> Path:
        """Create a JSONL file missing messages key."""
        data = {"other": "data"}
        path = tmp_path / "missing.jsonl"
        path.write_text(json.dumps(data) + "\n")
        return path

    def test_validate_jsonl_valid(self, valid_sft_jsonl) -> None:
        """validate_jsonl_or_raise should pass for valid data."""
        from synth_ai.sdk.learning.sft.data import validate_jsonl_or_raise

        # Should not raise
        validate_jsonl_or_raise(valid_sft_jsonl, min_messages=2)

    def test_validate_jsonl_empty_messages(self, empty_messages_jsonl) -> None:
        """validate_jsonl_or_raise should reject empty messages."""
        from synth_ai.sdk.learning.sft.data import validate_jsonl_or_raise

        with pytest.raises(ValueError, match="at least"):
            validate_jsonl_or_raise(empty_messages_jsonl, min_messages=2)

    def test_validate_jsonl_missing_messages(self, missing_messages_jsonl) -> None:
        """validate_jsonl_or_raise should reject missing messages key."""
        from synth_ai.sdk.learning.sft.data import validate_jsonl_or_raise

        with pytest.raises(ValueError):
            validate_jsonl_or_raise(missing_messages_jsonl, min_messages=2)


class TestFtClientUpload:
    """Test FtClient file upload functionality."""

    @pytest.fixture
    def valid_jsonl_dataset(self, tmp_path: Path) -> Path:
        """Create a valid JSONL training dataset."""
        dataset_path = tmp_path / "train.jsonl"
        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        dataset_path.write_text(json.dumps(data) + "\n")
        return dataset_path

    @pytest.fixture
    def invalid_jsonl_dataset(self, tmp_path: Path) -> Path:
        """Create an invalid JSONL dataset (empty messages)."""
        dataset_path = tmp_path / "invalid.jsonl"
        data = {"messages": []}
        dataset_path.write_text(json.dumps(data) + "\n")
        return dataset_path

    @pytest.mark.asyncio
    async def test_upload_training_file_success(self, valid_jsonl_dataset, monkeypatch) -> None:
        """upload_training_file should upload and return file ID."""
        from synth_ai.sdk.learning.sft.client import FtClient

        # Create mock HTTP client that returns file ID
        class MockHTTPClient:
            def __init__(self, *args, **kwargs):
                self.calls = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_multipart(self, url, data, files):
                self.calls.append(("post_multipart", url, data, files))
                return {"id": "file-upload-123"}

        mock_clients = []

        def make_mock(*args, **kwargs):
            client = MockHTTPClient(*args, **kwargs)
            mock_clients.append(client)
            return client

        monkeypatch.setattr("synth_ai.sdk.learning.sft.client.AsyncHttpClient", make_mock)

        client = FtClient(base_url="https://api.usesynth.ai", api_key="test-key")
        file_id = await client.upload_training_file(valid_jsonl_dataset)

        assert file_id == "file-upload-123"
        assert len(mock_clients) == 1
        assert len(mock_clients[0].calls) == 1

        call_method, call_url, call_data, call_files = mock_clients[0].calls[0]
        assert call_method == "post_multipart"
        assert call_url == "/api/learning/files"
        assert call_data["purpose"] == "fine-tune"
        assert "file" in call_files

    @pytest.mark.asyncio
    async def test_upload_training_file_validates_jsonl(self, invalid_jsonl_dataset, monkeypatch) -> None:
        """upload_training_file should reject invalid JSONL."""
        from synth_ai.sdk.learning.sft.client import FtClient

        # Mock should not be called - validation happens before upload
        def make_mock(*args, **kwargs):
            raise AssertionError("HTTP client should not be constructed for invalid data")

        monkeypatch.setattr("synth_ai.sdk.learning.sft.client.AsyncHttpClient", make_mock)

        client = FtClient(base_url="https://api.usesynth.ai", api_key="test-key")

        with pytest.raises(ValueError):
            await client.upload_training_file(invalid_jsonl_dataset)


class TestFtClientCreateJob:
    """Test FtClient job creation functionality."""

    @pytest.mark.asyncio
    async def test_create_sft_job_rejects_unknown_model(self) -> None:
        """create_sft_job should reject unknown model identifiers."""
        from synth_ai.sdk.learning.sft.client import FtClient

        client = FtClient(base_url="https://api.usesynth.ai", api_key="test-key")

        with pytest.raises(ValueError):
            await client.create_sft_job(
                model="Unknown/FakeModel",
                training_file_id="file-123",
                hyperparameters={},
            )

    @pytest.mark.asyncio
    async def test_create_sft_job_valid_model(self, monkeypatch) -> None:
        """create_sft_job should create job for valid model."""
        from synth_ai.sdk.learning.sft.client import FtClient

        class MockHTTPClient:
            def __init__(self, *args, **kwargs):
                self.json_calls = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post_json(self, url, json):
                self.json_calls.append((url, json))
                return {"id": "job-sft-123", "status": "created"}

        mock_clients = []

        def make_mock(*args, **kwargs):
            client = MockHTTPClient(*args, **kwargs)
            mock_clients.append(client)
            return client

        monkeypatch.setattr("synth_ai.sdk.learning.sft.client.AsyncHttpClient", make_mock)

        client = FtClient(base_url="https://api.usesynth.ai", api_key="test-key")
        response = await client.create_sft_job(
            model="Qwen/Qwen3-0.6B",
            training_file_id="file-123",
            hyperparameters={"n_epochs": 3, "learning_rate": 1e-5},
            metadata={"tags": ["test"]},
        )

        assert response["id"] == "job-sft-123"
        assert len(mock_clients) == 1
        assert len(mock_clients[0].json_calls) == 1

        call_url, call_payload = mock_clients[0].json_calls[0]
        assert call_url == "/api/learning/jobs"
        assert call_payload["model"] == "Qwen/Qwen3-0.6B"
        assert call_payload["hyperparameters"]["n_epochs"] == 3
