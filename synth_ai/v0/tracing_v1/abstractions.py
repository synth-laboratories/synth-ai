import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class MessageInputs:
    messages: List[Dict[str, str]]  # {"role": "", "content": ""}


@dataclass
class ArbitraryInputs:
    inputs: Dict[str, Any]


@dataclass
class MessageOutputs:
    messages: List[Dict[str, str]]


@dataclass
class ArbitraryOutputs:
    outputs: Dict[str, Any]


@dataclass
class ComputeStep:
    event_order: int
    compute_ended: datetime  # time step
    compute_began: datetime  # time step
    compute_input: Dict[str, Any]  # {variable_name: value}
    compute_output: Dict[str, Any]  # {variable_name: value}

    def to_dict(self):
        # Serialize compute_input
        serializable_input = [
            input_item.__dict__
            for input_item in self.compute_input
            if isinstance(input_item, (MessageInputs, ArbitraryInputs))
        ]

        # Serialize compute_output
        serializable_output = [
            output_item.__dict__
            for output_item in self.compute_output
            if isinstance(output_item, (MessageOutputs, ArbitraryOutputs))
        ]

        # Warn about non-serializable inputs/outputs
        for item in self.compute_input:
            if not isinstance(item, (MessageInputs, ArbitraryInputs)):
                logger.warning(f"Skipping non-serializable input: {item}")
        for item in self.compute_output:
            if not isinstance(item, (MessageOutputs, ArbitraryOutputs)):
                logger.warning(f"Skipping non-serializable output: {item}")

        return {
            "event_order": self.event_order,
            "compute_ended": self.compute_ended.isoformat()
            if isinstance(self.compute_ended, datetime)
            else self.compute_ended,
            "compute_began": self.compute_began.isoformat()
            if isinstance(self.compute_began, datetime)
            else self.compute_began,
            "compute_input": serializable_input,
            "compute_output": serializable_output,
        }


@dataclass
class AgentComputeStep(ComputeStep):
    model_name: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = None
    should_learn: Optional[bool] = None
    compute_input: List[Union[MessageInputs, ArbitraryInputs]]
    compute_output: List[Union[MessageOutputs, ArbitraryOutputs]]

    def to_dict(self):
        base_dict = super().to_dict()  # Get the parent class serialization
        base_dict["model_name"] = self.model_name  # Add model_name
        return base_dict


@dataclass
class EnvironmentComputeStep(ComputeStep):
    compute_input: List[ArbitraryInputs]
    compute_output: List[ArbitraryOutputs]


@dataclass
class Event:
    system_instance_id: str
    event_type: str
    opened: float
    closed: Optional[float]
    partition_index: int
    agent_compute_step: AgentComputeStep
    environment_compute_steps: List["EnvironmentComputeStep"]
    system_name: Optional[str] = None
    system_id: Optional[str] = None
    event_metadata: Dict[str, Any] = None  # JSON-serializable metadata for this specific event

    def __post_init__(self):
        if self.event_metadata is None:
            self.event_metadata = {}

    def to_dict(self):
        return {
            "event_type": self.event_type,
            "opened": self.opened.isoformat() if isinstance(self.opened, datetime) else self.opened,
            "closed": self.closed.isoformat() if isinstance(self.closed, datetime) else self.closed,
            "partition_index": self.partition_index,
            "agent_compute_step": self.agent_compute_step.to_dict()
            if self.agent_compute_step
            else None,
            "environment_compute_steps": [
                step.to_dict() for step in self.environment_compute_steps
            ],
            "event_metadata": self.event_metadata,
        }

    # backwards compatibility
    @property
    def agent_compute_steps(self) -> List[AgentComputeStep]:
        """Backwards compatibility method that returns a list containing the agent_compute_step."""
        return [self.agent_compute_step] if self.agent_compute_step is not None else []


@dataclass
class EventPartitionElement:
    partition_index: int
    events: List[Event]

    def to_dict(self):
        return {
            "partition_index": self.partition_index,
            "events": [event.to_dict() for event in self.events],
        }


@dataclass
class SystemTrace:
    system_name: str
    system_id: str
    system_instance_id: str
    partition: List[EventPartitionElement]
    metadata: Optional[Dict[str, Any]] = None  # System-level metadata
    instance_metadata: Dict[str, Any] = (
        None  # JSON-serializable metadata for this specific instance
    )
    current_partition_index: int = 0  # Track current partition

    def __post_init__(self):
        if self.instance_metadata is None:
            self.instance_metadata = {}

    def to_dict(self):
        return {
            "system_name": self.system_name,
            "system_id": self.system_id,
            "system_instance_id": self.system_instance_id,
            "partition": [element.to_dict() for element in self.partition],
            "current_partition_index": self.current_partition_index,
            "metadata": self.metadata if self.metadata else None,
            "instance_metadata": self.instance_metadata,
        }


class TrainingQuestion(BaseModel):
    """
    A training question is a question that an agent (system_instance_id) is trying to answer.
    It contains an intent and criteria that the agent is trying to meet.
    """

    id: str
    intent: str
    criteria: str

    def to_dict(self):
        return {
            "id": self.id,
            "intent": self.intent,
            "criteria": self.criteria,
        }


class RewardSignal(BaseModel):
    """
    A reward signal tells us how well an agent (system_instance_id) is doing on a particular question (question_id).
    """

    question_id: str
    system_instance_id: str
    reward: Union[float, int, bool]
    annotation: Optional[str] = None

    def to_dict(self):
        return {
            "question_id": self.question_id,
            "system_instance_id": self.system_instance_id,
            "reward": self.reward,
            "annotation": self.annotation,
        }


class Dataset(BaseModel):
    """
    A dataset is a collection of training questions and reward signals.
    This better represents the data that is used to train a model, and gives us more information about the data.
    """

    questions: List[TrainingQuestion]
    reward_signals: List[RewardSignal]

    def to_dict(self):
        return {
            "questions": [question.to_dict() for question in self.questions],
            "reward_signals": [signal.to_dict() for signal in self.reward_signals],
        }
