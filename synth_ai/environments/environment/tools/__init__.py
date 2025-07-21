from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, Type


class EnvToolCall(BaseModel):
    """Agent-requested call into an environment tool."""

    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    ok: bool
    payload: Any | None = None
    error: str | None = None


class AbstractTool(ABC):
    name: str
    call_schema: Type[BaseModel]
    result_schema: Type[BaseModel] = ToolResult

    @abstractmethod
    async def __call__(self, call: EnvToolCall) -> ToolResult: ...


TOOL_REGISTRY: dict[str, AbstractTool] = {}


def register_tool(tool: AbstractTool) -> None:
    """Register a tool instance under its `name` for dispatch."""
    TOOL_REGISTRY[tool.name] = tool
