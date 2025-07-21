from typing import Optional, Dict, List, Callable, Set, Any
from synth_ai.environments.v0_observability.history import SynthGlobalTrajectory
from uuid import UUID
from abc import abstractmethod
from dataclasses import dataclass, field


@dataclass
class Task:
    global_premises: str
    global_constraints: str
    global_objectives: str

    shared_env_params: Optional[Dict]


@dataclass
class TaskInstanceMetadata:
    pass


@dataclass
class Intent:
    rubric: Dict[str, Any]
    gold_trajectories: Optional[SynthGlobalTrajectory]
    gold_state_diff: Dict
    deterministic_eval_functions: List[Callable] = field(default_factory=list)


@dataclass
class Impetus:
    instructions: str

    # ?


@dataclass
class TaskInstance:
    id: UUID
    impetus: Impetus
    intent: Intent
    metadata: TaskInstanceMetadata
    is_reproducible: bool
    initial_engine_snapshot: Optional["StatefulEngineSnapshot"]

    @abstractmethod
    async def serialize(self) -> Dict:
        pass

    @abstractmethod
    async def deserialize(self) -> "TaskInstance":
        pass


@dataclass
class TaskInstanceMetadataFilter:
    @abstractmethod
    def __call__(
        self, instance: TaskInstance
    ) -> bool:  # Use Any temporarily for broader compatibility
        # Using Any avoids strict dependency on AgentStatefulTaskInstance here
        # Subclasses like MetadataFilter in helpers.py can specify the type.
        """Return True if the instance passes the filter."""


@dataclass
class SplitInfo:
    val_instance_ids: Set[str]
    test_instance_ids: Set[str]
    _is_split_defined: bool


@dataclass
class TaskInstanceSet:
    name: str
    description: str
    instances: List[TaskInstance]
    split_info: SplitInfo
