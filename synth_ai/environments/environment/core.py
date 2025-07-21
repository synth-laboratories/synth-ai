from synth_ai.core.system import System

class Environment(System):
    """
    Base class for all environments, providing a name attribute.
    """

    _default_name: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Set a default name for the subclass based on its class name
        cls._default_name = cls.__name__

    @property
    def name(self) -> str:
        """Return the environment name, defaulting to the subclass name."""
        return getattr(self, "_name", self._default_name)

    @name.setter
    def name(self, value: str):
        self._name = value
