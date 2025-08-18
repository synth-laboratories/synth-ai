from abc import abstractmethod

from synth_ai.environments.environment.shared_engine import Engine, InternalObservation
from synth_ai.environments.environment.tools import EnvToolCall


class StatefulEnvironment(Engine):
    """
    Abstract base class for stateful environments in the Synth AI framework.

    This class defines the interface for environments that maintain state between
    interactions and support agent-environment interactions through tool calls.
    StatefulEnvironments are designed to work with AI agents that can observe
    the environment, take actions through tool calls, and receive feedback.

    The environment follows a standard lifecycle:
    1. Initialize - Set up initial state and return first observation
    2. Step - Process agent tool calls and return new observations
    3. Checkpoint - Save current state for potential restoration
    4. Terminate - Clean up and finalize the environment

    All methods are async to support non-blocking operations and integration
    with modern async/await patterns in AI applications.

    Example:
        >>> class MyGameEnv(StatefulEnvironment):
        ...     async def initialize(self):
        ...         # Set up game state
        ...         return initial_observation
        ...
        ...     async def step(self, tool_calls):
        ...         # Process player actions
        ...         return new_observation
        >>>
        >>> env = MyGameEnv(task)
        >>> obs = await env.initialize()
        >>> result = await env.step([tool_call])
    """

    @abstractmethod
    async def initialize(self) -> InternalObservation:
        """
        Initialize the environment and return the initial observation.

        This method sets up the environment's initial state, loads any
        necessary resources, and prepares the environment for agent interaction.
        It should be called once before any step() calls.

        Returns:
            InternalObservation: The initial state observation that the agent
                will use to understand the environment and plan its first action.

        Raises:
            EnvironmentError: If initialization fails due to invalid configuration
                or resource unavailability.

        Example:
            >>> env = MyEnvironment(task)
            >>> initial_obs = await env.initialize()
            >>> print(initial_obs.observation)  # Agent-visible state
        """
        pass

    @abstractmethod
    async def terminate(self) -> InternalObservation:
        """
        Terminate the environment and return the final observation.

        This method performs cleanup operations, saves any persistent state,
        and prepares the final observation that summarizes the environment's
        end state. It should be called when the episode or session is complete.

        Returns:
            InternalObservation: The final state observation, typically including
                summary information, final scores, or completion status.

        Example:
            >>> final_obs = await env.terminate()
            >>> print(final_obs.observation.get('final_score'))
        """
        pass

    # main external api
    @abstractmethod
    def validate_tool_calls(self, tool_calls: EnvToolCall):
        """
        Validate that tool calls are properly formatted and executable.

        This method checks tool calls before execution to ensure they:
        - Reference valid tools available in this environment
        - Provide required arguments with correct types
        - Follow any environment-specific constraints or rules

        Args:
            tool_calls: The tool call(s) to validate. Can be a single call
                or a list of calls depending on the environment's capabilities.

        Raises:
            ValidationError: If tool calls are invalid, malformed, or not
                supported by this environment.
            TypeError: If tool_calls is not the expected type.

        Example:
            >>> tool_call = EnvToolCall(tool="move", args={"direction": "north"})
            >>> env.validate_tool_calls(tool_call)  # Raises if invalid
        """
        pass

    @abstractmethod
    async def step(self, tool_calls: list[EnvToolCall]) -> InternalObservation:
        """
        Execute tool calls and return the resulting observation.

        This is the main interaction method where agents submit actions
        (as tool calls) and receive feedback from the environment. The method:
        1. Validates the tool calls (may call validate_tool_calls)
        2. Executes the actions in the environment
        3. Updates the environment state
        4. Returns the new observation for the agent

        Args:
            tool_calls: List of tool calls representing the agent's actions.
                Each tool call specifies a tool name and arguments.

        Returns:
            InternalObservation: The new state observation after executing
                the tool calls, including any changes, rewards, or feedback.

        Raises:
            ValidationError: If tool calls are invalid or cannot be executed.
            EnvironmentError: If execution fails due to environment state issues.

        Example:
            >>> tool_calls = [
            ...     EnvToolCall(tool="move", args={"direction": "north"}),
            ...     EnvToolCall(tool="pick_up", args={"item": "key"})
            ... ]
            >>> obs = await env.step(tool_calls)
            >>> print(obs.observation.get('player_location'))
        """
        pass

    @abstractmethod
    async def checkpoint(self) -> InternalObservation:
        """
        Create a checkpoint of the current environment state.

        This method saves the current state of the environment for potential
        restoration later. It's useful for:
        - Implementing save/load functionality
        - Creating branching scenarios for exploration
        - Debugging and development
        - Rollback mechanisms for error recovery

        Returns:
            InternalObservation: Current state observation, potentially including
                checkpoint metadata or state identifiers.

        Example:
            >>> checkpoint_obs = await env.checkpoint()
            >>> checkpoint_id = checkpoint_obs.metadata.get('checkpoint_id')
        """
        pass
