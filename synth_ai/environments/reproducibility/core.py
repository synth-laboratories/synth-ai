from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar


class IReproducibleEngine(ABC):
    """
    An abstract base class for engines that support serialization and deserialization,
    making them reproducible.
    """

    @abstractmethod
    async def _serialize_engine(
        self,
    ) -> Any:  # Replace Any with a more specific Snapshot type if common one emerges
        """Serializes the current state of the engine."""
        pass

    @classmethod
    @abstractmethod
    async def _deserialize_engine(cls, snapshot: Any) -> "IReproducibleEngine":  # Replace Any
        """Creates an engine instance from a serialized snapshot."""
        pass


# Type variable for the engine, ensuring it adheres to the IReproducibleEngine interface.
EngineType_co = TypeVar("EngineType_co", bound=IReproducibleEngine, covariant=True)


class ReproducibleEnvironment(Generic[EngineType_co]):
    """
    A mixin class for environments that support reproducibility through
    engine serialization and deserialization.

    It expects the environment to have an 'engine' attribute that conforms to
    the IReproducibleEngine interface. This contract is enforced via type hinting
    and the IReproducibleEngine ABC.
    """

    engine: EngineType_co
    # No explicit runtime checks like hasattr are performed here.
    # The presence and correctness of _serialize_engine and _deserialize_engine
    # methods on the engine are ensured by the IReproducibleEngine contract.
