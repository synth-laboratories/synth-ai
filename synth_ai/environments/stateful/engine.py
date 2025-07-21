from synth_ai.environments.environment.shared_engine import Engine
from typing import TypeVar


SnapshotType = TypeVar("SnapshotType", bound="StatefulEngineSnapshot")


class StatefulEngineSnapshot:
    pass


class StatefulEngine(Engine):
    async def serialize(self):
        pass

    @classmethod
    async def deserialize(self, engine_snapshot: StatefulEngineSnapshot):
        pass

    async def _step_engine(self):
        pass
