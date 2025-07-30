from synth_ai.core.system import System


class Environment(System):
    """
    Base class for all environments in the Synth AI framework.

    This class provides the fundamental structure for all environment types,
    including a name attribute system that supports both automatic naming
    based on the class name and manual name assignment.

    The Environment class serves as the foundation for more specialized
    environment types like StatefulEnvironment, providing common functionality
    and ensuring consistent interfaces across all environment implementations.

    Attributes:
        _default_name: Class-level default name derived from the class name
        _name: Instance-level name override (optional)

    Example:
        >>> class MyCustomEnv(Environment):
        ...     pass
        >>> env = MyCustomEnv()
        >>> print(env.name)  # "MyCustomEnv"
        >>> env.name = "custom_environment"
        >>> print(env.name)  # "custom_environment"
    """

    _default_name: str

    def __init_subclass__(cls, **kwargs):
        """
        Initialize subclass with automatic name assignment.

        This method is called when a new subclass of Environment is created.
        It automatically sets the _default_name attribute to the class name,
        providing a fallback name for environment instances.

        Args:
            **kwargs: Additional keyword arguments passed to parent classes
        """
        super().__init_subclass__(**kwargs)
        # Set a default name for the subclass based on its class name
        cls._default_name = cls.__name__

    @property
    def name(self) -> str:
        """
        Get the environment name.

        Returns the instance-specific name if set via the setter,
        otherwise returns the class-level default name.

        Returns:
            str: The environment name, either custom or default
        """
        return getattr(self, "_name", self._default_name)

    @name.setter
    def name(self, value: str):
        """
        Set a custom name for this environment instance.

        Args:
            value: The custom name to assign to this environment
        """
        self._name = value
