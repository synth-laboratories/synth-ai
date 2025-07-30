from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, Type


class EnvToolCall(BaseModel):
    """
    Represents an agent-requested call to an environment tool.

    This class encapsulates an action that an AI agent wants to perform
    in an environment. Tool calls consist of a tool name and a dictionary
    of arguments that will be passed to the tool for execution.

    The tool call system provides a standardized way for agents to interact
    with environments, making it easy to:
    - Validate agent actions before execution
    - Log and trace agent behavior
    - Implement complex multi-step actions
    - Handle errors and provide feedback

    Attributes:
        tool: The name of the tool to invoke (must be registered in environment)
        args: Arguments to pass to the tool, with argument names as keys

    Example:
        >>> # Simple movement action
        >>> move_call = EnvToolCall(tool="move", args={"direction": "north"})

        >>> # Complex action with multiple parameters
        >>> craft_call = EnvToolCall(
        ...     tool="craft_item",
        ...     args={"item": "sword", "materials": ["iron", "wood"], "quantity": 1}
        ... )

        >>> # Action with no arguments
        >>> look_call = EnvToolCall(tool="look", args={})
    """

    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """
    Represents the result of executing an environment tool.

    This class standardizes the response format for all tool executions,
    providing a consistent interface for success/failure status, return
    values, and error information. This makes it easy for agents and
    environment systems to handle tool execution results uniformly.

    Attributes:
        ok: Whether the tool execution was successful
        payload: The return value from the tool (None if failed or no return value)
        error: Error message if execution failed (None if successful)

    Example:
        >>> # Successful tool execution
        >>> success_result = ToolResult(
        ...     ok=True,
        ...     payload={"new_position": [5, 3], "items_collected": ["key"]},
        ...     error=None
        ... )

        >>> # Failed tool execution
        >>> error_result = ToolResult(
        ...     ok=False,
        ...     payload=None,
        ...     error="Cannot move north: wall blocking path"
        ... )

        >>> # Success with no return value
        >>> simple_success = ToolResult(ok=True)
    """

    ok: bool
    payload: Any | None = None
    error: str | None = None


class AbstractTool(ABC):
    """
    Abstract base class for all environment tools.

    Tools are the primary mechanism for agents to interact with environments.
    Each tool represents a specific action or capability that an agent can
    invoke, such as moving, picking up items, or examining objects.

    Tools define their own call and result schemas using Pydantic models,
    enabling automatic validation and documentation generation. This ensures
    that agents provide valid inputs and receive structured outputs.

    The tool system supports:
    - Type-safe argument validation
    - Automatic error handling and reporting
    - Consistent result formatting
    - Dynamic tool registration and discovery

    Attributes:
        name: Unique identifier for this tool (used in EnvToolCall.tool)
        call_schema: Pydantic model defining valid arguments for this tool
        result_schema: Pydantic model defining the structure of results

    Example:
        >>> class MoveTool(AbstractTool):
        ...     name = "move"
        ...     call_schema = MoveArgs  # Pydantic model with 'direction' field
        ...
        ...     async def __call__(self, call: EnvToolCall) -> ToolResult:
        ...         direction = call.args['direction']
        ...         # Perform movement logic
        ...         if valid_move:
        ...             return ToolResult(ok=True, payload=new_position)
        ...         else:
        ...             return ToolResult(ok=False, error="Invalid move")

        >>> # Register and use the tool
        >>> move_tool = MoveTool()
        >>> register_tool(move_tool)
        >>> call = EnvToolCall(tool="move", args={"direction": "north"})
        >>> result = await move_tool(call)
    """

    name: str
    call_schema: Type[BaseModel]
    result_schema: Type[BaseModel] = ToolResult

    @abstractmethod
    async def __call__(self, call: EnvToolCall) -> ToolResult:
        """
        Execute the tool with the given tool call.

        This method contains the core logic for the tool's functionality.
        It should:
        1. Validate the tool call arguments (using call_schema)
        2. Perform the requested action
        3. Return a ToolResult with appropriate success/failure status

        Args:
            call: The tool call containing the tool name and arguments

        Returns:
            ToolResult: Result of tool execution with success status,
                payload data, and any error messages

        Raises:
            ValidationError: If call arguments don't match call_schema
            EnvironmentError: If tool execution fails due to environment state

        Example:
            >>> call = EnvToolCall(tool="move", args={"direction": "east"})
            >>> result = await tool(call)
            >>> if result.ok:
            ...     print(f"Moved to: {result.payload['position']}")
            ... else:
            ...     print(f"Move failed: {result.error}")
        """
        ...


TOOL_REGISTRY: dict[str, AbstractTool] = {}


def register_tool(tool: AbstractTool) -> None:
    """
    Register a tool instance for use in environments.

    This function adds a tool to the global registry, making it available
    for environments to use when processing agent tool calls. Tools must
    be registered before they can be invoked by agents.

    The registry uses the tool's name attribute as the key, so tool names
    must be unique across all registered tools.

    Args:
        tool: The tool instance to register. Must have a unique name.

    Raises:
        ValueError: If a tool with the same name is already registered
        TypeError: If tool is not an instance of AbstractTool

    Example:
        >>> class LookTool(AbstractTool):
        ...     name = "look"
        ...     # ... implementation

        >>> look_tool = LookTool()
        >>> register_tool(look_tool)
        >>>
        >>> # Now agents can use: EnvToolCall(tool="look", args={})

    Note:
        Tools are typically registered during environment initialization
        or module import. Once registered, tools remain available for
        the duration of the application session.
    """
    TOOL_REGISTRY[tool.name] = tool
