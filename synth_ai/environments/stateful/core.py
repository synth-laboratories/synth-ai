from abc import abstractmethod
from typing import List

from synth_ai.environments.environment.shared_engine import Engine, InternalObservation
from synth_ai.environments.environment.tools import EnvToolCall


class StatefulEnvironment(Engine):
    @abstractmethod
    async def initialize(self) -> InternalObservation:
        pass

    @abstractmethod
    async def terminate(self) -> InternalObservation:
        pass

    # main external api
    @abstractmethod
    def validate_tool_calls(self, tool_calls: EnvToolCall):
        pass

    @abstractmethod
    async def step(self, tool_calls: List[EnvToolCall]) -> InternalObservation:
        pass

    @abstractmethod
    async def checkpoint(self) -> InternalObservation:
        pass
