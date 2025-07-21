"""Tests for synth_ai.tracing.upload module."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import requests
from requests.exceptions import HTTPError
from synth_ai.tracing.abstractions import (
    AgentComputeStep,
    ArbitraryInputs,
    ArbitraryOutputs,
    Event,
    Dataset,
    EnvironmentComputeStep,
    EventPartitionElement,
    MessageInputs,
    MessageOutputs,
    RewardSignal,
    SystemTrace,
    TrainingQuestion,
)
from synth_ai.tracing.upload import (
    UploadValidator,
    createPayload,
    format_upload_output,
    upload,
    validate_json,
    validate_upload,
)

import time
from datetime import datetime
from synth_ai.tracing.events.store import event_store


@pytest.fixture(autouse=True)
def clear_event_store():
    """Clear event store before each test."""
    with event_store._lock:
        event_store._traces.clear()
    yield
    with event_store._lock:
        event_store._traces.clear()


def create_test_event(**kwargs):
    """Helper function to create a test Event with required fields."""
    # Default values
    base_time = kwargs.pop("opened", time.time())
    defaults = {
        "system_instance_id": "test_system",
        "event_type": "test",
        "opened": base_time,
        "closed": kwargs.pop("closed", base_time + 1),
        "partition_index": 0,
        "agent_compute_step": AgentComputeStep(
            event_order=1,
            compute_began=datetime.fromtimestamp(base_time),
            compute_ended=datetime.fromtimestamp(base_time + 0.5),
            compute_input=[MessageInputs(messages=[{"role": "user", "content": "test"}])],
            compute_output=[
                MessageOutputs(messages=[{"role": "assistant", "content": "response"}])
            ],
        ),
        "environment_compute_steps": [
            EnvironmentComputeStep(
                event_order=2,
                compute_began=datetime.fromtimestamp(base_time + 0.5),
                compute_ended=datetime.fromtimestamp(base_time + 1),
                compute_input=[ArbitraryInputs(inputs={"key": "value"})],
                compute_output=[ArbitraryOutputs(outputs={"result": "success"})],
            )
        ],
    }
    # Update with provided kwargs
    defaults.update(kwargs)
    return Event(**defaults)


class TestValidateJson:
    """Test JSON validation functionality."""

    def test_valid_json(self):
        """Test that valid JSON passes validation."""
        data = {
            "string": "test",
            "number": 123,
            "float": 45.67,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        }
        # Should not raise
        validate_json(data)

    def test_invalid_json_with_function(self):
        """Test that non-serializable objects raise ValueError."""
        data = {
            "function": lambda x: x,  # Functions are not JSON serializable
        }
        with pytest.raises(ValueError, match="non-JSON-serializable"):
            validate_json(data)

    def test_invalid_json_with_object(self):
        """Test that custom objects raise ValueError."""

        class CustomObject:
            pass

        data = {"object": CustomObject()}
        with pytest.raises(ValueError, match="non-JSON-serializable"):
            validate_json(data)


class TestCreatePayload:
    """Test payload creation."""

    def test_create_payload_structure(self):
        """Test that createPayload produces correct structure."""
        # Create test data
        questions = [TrainingQuestion(intent="test", criteria="test", id="q1")]
        signals = [RewardSignal(question_id="q1", system_instance_id="sys1", reward=1.0)]
        dataset = Dataset(questions=questions, reward_signals=signals)

        event = create_test_event(event_type="test", opened=1.0, closed=2.0, partition_index=0)
        partition = EventPartitionElement(partition_index=0, events=[event])
        trace = SystemTrace(
            system_name="test",
            system_id="id1",
            system_instance_id="inst1",
            partition=[partition],
        )

        payload = createPayload(dataset, [trace])

        assert "traces" in payload
        assert "dataset" in payload
        assert len(payload["traces"]) == 1
        assert payload["traces"][0]["system_name"] == "test"
        assert len(payload["dataset"]["questions"]) == 1
        assert len(payload["dataset"]["reward_signals"]) == 1


class TestUploadValidator:
    """Test the UploadValidator class."""

    def test_valid_upload_data(self):
        """Test validation of valid upload data."""
        traces = [
            {
                "system_instance_id": "inst1",
                "partition": [
                    {
                        "partition_index": 0,
                        "events": [
                            {
                                "event_type": "test",
                                "opened": 1.0,
                                "closed": 2.0,
                                "partition_index": 0,
                            }
                        ],
                    }
                ],
            }
        ]

        dataset = {
            "questions": [{"intent": "test", "criteria": "test"}],
            "reward_signals": [],
        }

        # Should not raise
        validator = UploadValidator(traces=traces, dataset=dataset)
        assert validator.traces == traces
        assert validator.dataset == dataset

    def test_invalid_traces_missing_field(self):
        """Test validation fails with missing required trace fields."""
        traces = [
            {
                # Missing system_instance_id
                "partition": [
                    {
                        "partition_index": 0,
                        "events": [],
                    }
                ]
            }
        ]

        dataset = {"questions": [], "reward_signals": []}

        with pytest.raises(ValueError, match="system_instance_id"):
            UploadValidator(traces=traces, dataset=dataset)

    def test_invalid_event_structure(self):
        """Test validation fails with invalid event structure."""
        traces = [
            {
                "system_instance_id": "inst1",
                "partition": [
                    {
                        "partition_index": 0,
                        "events": [
                            {
                                # Missing required event fields
                                "event_type": "test",
                            }
                        ],
                    }
                ],
            }
        ]

        dataset = {"questions": [], "reward_signals": []}

        with pytest.raises(ValueError, match="missing required fields"):
            UploadValidator(traces=traces, dataset=dataset)

    def test_empty_traces_list(self):
        """Test validation fails with empty traces list."""
        traces = []
        dataset = {"questions": [], "reward_signals": []}

        with pytest.raises(ValueError, match="Traces list cannot be empty"):
            UploadValidator(traces=traces, dataset=dataset)


class TestFormatUploadOutput:
    """Test the format_upload_output function."""

    def test_format_upload_output_complete(self):
        """Test formatting complete upload output."""
        # Create test data
        questions = [
            TrainingQuestion(intent="intent1", criteria="criteria1", id="q1"),
            TrainingQuestion(intent="intent2", criteria="criteria2", id="q2"),
        ]
        signals = [
            RewardSignal(
                question_id="q1",
                system_instance_id="sys1",
                reward=0.8,
                annotation="good",
            ),
        ]
        dataset = Dataset(questions=questions, reward_signals=signals)

        event = create_test_event(event_type="test", opened=1.0, closed=2.0, partition_index=0)
        partition = EventPartitionElement(partition_index=0, events=[event])
        trace = SystemTrace(
            system_name="test",
            system_id="id1",
            system_instance_id="inst1",
            partition=[partition],
            metadata={"version": "1.0"},
        )

        questions_json, signals_json, traces_json = format_upload_output(dataset, [trace])

        # Check questions
        assert len(questions_json) == 2
        assert questions_json[0]["intent"] == "intent1"
        assert questions_json[0]["criteria"] == "criteria1"
        assert questions_json[0]["id"] == "q1"

        # Check signals
        assert len(signals_json) == 1
        assert signals_json[0]["reward"] == 0.8
        assert signals_json[0]["annotation"] == "good"

        # Check traces
        assert len(traces_json) == 1
        assert traces_json[0]["system_instance_id"] == "inst1"
        assert traces_json[0]["metadata"] == {"version": "1.0"}


class TestUploadFunction:
    """Test the main upload function."""

    @patch.dict(os.environ, {"SYNTH_API_KEY": "test-api-key"})
    @patch("synth_ai.tracing.upload.send_system_traces_s3")
    @patch("synth_ai.tracing.events.store.event_store")
    def test_upload_success(self, mock_event_store, mock_send_s3):
        """Test successful upload."""
        # Setup mocks
        mock_send_s3.return_value = ("upload-123", "https://signed-url")
        mock_event_store.get_system_traces.return_value = []
        mock_event_store._traces = {}
        mock_event_store._lock = MagicMock()
        mock_event_store.add_event = MagicMock()

        # Create test data
        questions = [TrainingQuestion(id="q1", intent="test", criteria="test")]
        signals = [RewardSignal(question_id="q1", system_instance_id="sys1", reward=1.0)]
        dataset = Dataset(questions=questions, reward_signals=signals)

        event = create_test_event(event_type="test", opened=1.0, closed=2.0, partition_index=0)
        partition = EventPartitionElement(partition_index=0, events=[event])
        trace = SystemTrace(
            system_name="test",
            system_id="id1",
            system_instance_id="inst1",
            partition=[partition],
        )

        # Call upload
        response, q_json, s_json, t_json = upload(dataset, [trace])

        # Verify response
        assert response["status"] == "success"
        assert response["upload_id"] == "upload-123"
        assert response["signed_url"] == "https://signed-url"

        # Verify formatted data
        assert len(q_json) == 1
        assert len(s_json) == 1
        assert len(t_json) == 1

    @patch.dict(os.environ, {}, clear=True)
    def test_upload_no_api_key(self):
        """Test upload fails without API key."""
        dataset = Dataset(questions=[], reward_signals=[])

        with pytest.raises(ValueError, match="SYNTH_API_KEY environment variable not set"):
            upload(dataset)

    @patch.dict(os.environ, {"SYNTH_API_KEY": "test-key"})
    @patch("synth_ai.tracing.upload.event_store")
    def test_upload_no_traces(self, mock_event_store):
        """Test upload fails with no traces."""
        # Mock event store to return no traces
        mock_event_store.get_system_traces.return_value = []
        mock_event_store._traces = {}
        mock_event_store._lock = MagicMock()
        mock_event_store.add_event = MagicMock()

        dataset = Dataset(questions=[], reward_signals=[])

        with pytest.raises(ValueError, match="No system traces found"):
            upload(dataset, [])

    @patch.dict(os.environ, {"SYNTH_API_KEY": "test-key"})
    @patch("synth_ai.tracing.upload.send_system_traces_s3")
    @patch("synth_ai.tracing.events.store.event_store")
    def test_upload_http_error(self, mock_event_store, mock_send_s3):
        """Test upload handles HTTP errors."""
        # Setup mocks
        mock_event_store.get_system_traces.return_value = []
        mock_send_s3.side_effect = HTTPError("404 Not Found")

        # Create test data
        dataset = Dataset(questions=[], reward_signals=[])
        event = create_test_event(event_type="test", opened=1.0, closed=2.0, partition_index=0)
        partition = EventPartitionElement(partition_index=0, events=[event])
        trace = SystemTrace(
            system_name="test",
            system_id="id1",
            system_instance_id="inst1",
            partition=[partition],
        )

        with pytest.raises(HTTPError):
            upload(dataset, [trace])


class TestValidateUpload:
    """Test the validate_upload function."""

    def test_validate_upload_valid(self):
        """Test validation passes for valid data."""
        traces = [
            {
                "system_instance_id": "inst1",
                "partition": [
                    {
                        "partition_index": 0,
                        "events": [
                            {
                                "event_type": "test",
                                "opened": 1.0,
                                "closed": 2.0,
                                "partition_index": 0,
                            }
                        ],
                    }
                ],
            }
        ]

        dataset = {
            "questions": [{"intent": "test", "criteria": "test"}],
            "reward_signals": [],
        }

        # Should return True and not raise
        result = validate_upload(traces, dataset)
        assert result is True

    def test_validate_upload_invalid(self):
        """Test validation fails for invalid data."""
        traces = []  # Empty traces
        dataset = {"questions": [], "reward_signals": []}

        with pytest.raises(ValueError, match="Upload validation failed"):
            validate_upload(traces, dataset)
