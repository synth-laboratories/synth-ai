from abc import abstractmethod
from typing import Any


class InternalObservation:
    public_observation: Any
    private_observation: Any


class GetObservationCallable:
    @abstractmethod
    async def get_observation(self) -> InternalObservation:
        pass

    pass


class Engine:
    async def initialize(self):
        pass

    async def terminate(self):
        pass

    async def _step_engine(self):
        pass
