"""Tests for synth_ai.tracing.abstractions module."""

import time
import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pytest
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


def create_test_event(**kwargs):
    """Helper function to create a test Event with required fields."""
    # Default values
    base_time = time.time()
    defaults = {
        "system_instance_id": "test_system",
        "event_type": "test",
        "opened": base_time,
        "closed": base_time + 1,
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


class TestEvent:
    """Test the Event class."""

    def test_event_creation(self):
        """Test creating an Event instance."""
        event = create_test_event(event_type="test_event")
        assert event.event_type == "test_event"
        assert event.closed > event.opened
        assert event.partition_index == 0
        assert event.system_instance_id == "test_system"

    def test_event_to_dict(self):
        """Test converting Event to dictionary."""
        event = create_test_event(event_type="test_event", partition_index=1)
        result = event.to_dict()
        assert result["event_type"] == "test_event"
        assert result["partition_index"] == 1
        assert result["agent_compute_step"] is not None
        assert result["environment_compute_steps"] is not None
        assert len(result["environment_compute_steps"]) == 1


class TestAgentComputeStep:
    """Test the AgentComputeStep class."""

    def test_agent_compute_step_creation(self):
        """Test creating an AgentComputeStep instance."""
        current_time = time.time()
        step = AgentComputeStep(
            event_order=1,
            compute_began=datetime.fromtimestamp(current_time),
            compute_ended=datetime.fromtimestamp(current_time + 1),
            compute_input=[MessageInputs(messages=[{"role": "user", "content": "test"}])],
            compute_output=[
                MessageOutputs(messages=[{"role": "assistant", "content": "response"}])
            ],
            model_name="test-model",
        )
        assert step.event_order == 1
        assert step.compute_ended > step.compute_began
        assert step.model_name == "test-model"

    def test_agent_compute_step_optional_fields(self):
        """Test AgentComputeStep with optional fields."""
        current_time = time.time()
        step = AgentComputeStep(
            event_order=1,
            compute_began=datetime.fromtimestamp(current_time),
            compute_ended=datetime.fromtimestamp(current_time + 1),
            compute_input=[],
            compute_output=[],
        )
        assert step.model_name is None


class TestEnvironmentComputeStep:
    """Test the EnvironmentComputeStep class."""

    def test_environment_compute_step_creation(self):
        """Test creating an EnvironmentComputeStep instance."""
        current_time = time.time()
        step = EnvironmentComputeStep(
            event_order=1,
            compute_began=datetime.fromtimestamp(current_time),
            compute_ended=datetime.fromtimestamp(current_time + 1),
            compute_input=[ArbitraryInputs(inputs={"action": "test"})],
            compute_output=[ArbitraryOutputs(outputs={"result": "success"})],
        )
        assert step.event_order == 1


class TestEventPartitionElement:
    """Test the EventPartitionElement class."""

    def test_event_partition_element_creation(self):
        """Test creating an EventPartitionElement instance."""
        event = create_test_event(event_type="test", partition_index=0)
        partition = EventPartitionElement(partition_index=0, events=[event])
        assert partition.partition_index == 0
        assert len(partition.events) == 1
        assert partition.events[0].event_type == "test"


class TestTrainingQuestion:
    """Test the TrainingQuestion class."""

    def test_training_question_creation(self):
        """Test creating a TrainingQuestion instance."""
        question = TrainingQuestion(
            intent="Test intent",
            criteria="Test criteria",
            id="test-id",
        )
        assert question.intent == "Test intent"
        assert question.criteria == "Test criteria"
        assert question.id == "test-id"

    def test_training_question_with_uuid(self):
        """Test TrainingQuestion with UUID."""
        test_id = str(uuid.uuid4())
        question = TrainingQuestion(
            id=test_id,
            intent="Test intent",
            criteria="Test criteria",
        )
        assert question.intent == "Test intent"
        assert question.criteria == "Test criteria"
        assert question.id == test_id
        # Check that id is a valid UUID
        uuid.UUID(question.id)  # This will raise if not valid


class TestRewardSignal:
    """Test the RewardSignal class."""

    def test_reward_signal_creation(self):
        """Test creating a RewardSignal instance."""
        signal = RewardSignal(
            question_id="q1",
            system_instance_id="sys1",
            reward=0.8,
            annotation="Good job",
        )
        assert signal.question_id == "q1"
        assert signal.system_instance_id == "sys1"
        assert signal.reward == 0.8
        assert signal.annotation == "Good job"

    def test_reward_signal_optional_annotation(self):
        """Test RewardSignal without annotation."""
        signal = RewardSignal(
            question_id="q1",
            system_instance_id="sys1",
            reward=0.5,
        )
        assert signal.annotation is None


class TestDataset:
    """Test the Dataset class."""

    def test_dataset_creation(self):
        """Test creating a Dataset instance."""
        questions = [
            TrainingQuestion(id="q1", intent="Intent 1", criteria="Criteria 1"),
            TrainingQuestion(id="q2", intent="Intent 2", criteria="Criteria 2"),
        ]
        signals = [
            RewardSignal(question_id="q1", system_instance_id="sys1", reward=1.0),
            RewardSignal(question_id="q2", system_instance_id="sys1", reward=0.5),
        ]
        dataset = Dataset(questions=questions, reward_signals=signals)
        assert len(dataset.questions) == 2
        assert len(dataset.reward_signals) == 2

    def test_dataset_to_dict(self):
        """Test converting Dataset to dictionary."""
        questions = [TrainingQuestion(intent="Test", criteria="Test", id="q1")]
        signals = [RewardSignal(question_id="q1", system_instance_id="sys1", reward=1.0)]
        dataset = Dataset(questions=questions, reward_signals=signals)

        result = dataset.to_dict()
        assert "questions" in result
        assert "reward_signals" in result
        assert len(result["questions"]) == 1
        assert len(result["reward_signals"]) == 1


class TestSystemTrace:
    """Test the SystemTrace class."""

    def test_system_trace_creation(self):
        """Test creating a SystemTrace instance."""
        event = create_test_event(event_type="test", partition_index=0)
        partition = EventPartitionElement(partition_index=0, events=[event])

        trace = SystemTrace(
            system_name="test_system",
            system_id="sys-123",
            system_instance_id="instance-456",
            partition=[partition],
            current_partition_index=0,
            metadata={"version": "1.0"},
        )

        assert trace.system_name == "test_system"
        assert trace.system_id == "sys-123"
        assert trace.system_instance_id == "instance-456"
        assert len(trace.partition) == 1
        assert trace.metadata == {"version": "1.0"}

    def test_system_trace_to_dict(self):
        """Test converting SystemTrace to dictionary."""
        event = create_test_event(event_type="test", partition_index=0)
        partition = EventPartitionElement(partition_index=0, events=[event])

        trace = SystemTrace(
            system_name="test_system",
            system_id="sys-123",
            system_instance_id="instance-456",
            partition=[partition],
        )

        result = trace.to_dict()
        assert result["system_name"] == "test_system"
        assert result["system_id"] == "sys-123"
        assert result["system_instance_id"] == "instance-456"
        assert "partition" in result
        assert len(result["partition"]) == 1
