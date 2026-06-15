"""Builder helpers for OpenAI Agents SDK phase-1 tool definitions.

These produce plain dicts that can be passed directly to the ``tools``
field of a ``/responses`` create request.  They do not validate against
the server — they are convenience constructors that enforce the correct
shape so callers do not have to remember field names.

Phase-1 tool families (per horizons-private chunk-1 freeze):
  - function  — function-calling tools with a JSON Schema parameters block
  - mcp       — MCP server tools (server_label + server_url + optional headers)

Phase-2+ tool families (shell, apply_patch, web_search, code_interpreter,
computer_use) are intentionally absent here.
"""

from __future__ import annotations

from typing import Any


def function_tool(
    name: str,
    *,
    description: str = "",
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a function-tool definition dict.

    Args:
        name: The function name the model will invoke.
        description: Human-readable description of what the function does.
        parameters: JSON Schema object describing the function's parameters.
            Defaults to an empty object schema if omitted.

    Returns:
        A dict with ``type="function"`` ready to pass in the ``tools`` list.

    Example::

        tool = function_tool(
            "get_weather",
            description="Return current weather for a city.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
    """
    return {
        "type": "function",
        "name": str(name).strip(),
        "description": str(description or "").strip(),
        "parameters": parameters
        if parameters is not None
        else {"type": "object", "properties": {}},
    }


def mcp_tool(
    server_label: str,
    server_url: str,
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return an MCP-server tool definition dict.

    Args:
        server_label: Short label identifying the MCP server (used as the
            tool's display name and routing key).
        server_url: Full URL of the MCP server endpoint.
        headers: Optional HTTP headers the runtime should forward when
            connecting to the MCP server (e.g. auth tokens).

    Returns:
        A dict with ``type="mcp"`` ready to pass in the ``tools`` list.

    Example::

        tool = mcp_tool(
            "my_tools",
            "https://mcp.example.com",
            headers={"Authorization": "Bearer tok_..."},
        )
    """
    result: dict[str, Any] = {
        "type": "mcp",
        "server_label": str(server_label).strip(),
        "server_url": str(server_url).strip(),
    }
    if headers:
        result["headers"] = dict(headers)
    return result


__all__ = ["function_tool", "mcp_tool"]
