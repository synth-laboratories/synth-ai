"""Type definitions for storage layer."""

from enum import Enum


class EventType(str, Enum):
    """Enumeration of event types."""

    CAIS = "cais"
    ENVIRONMENT = "environment"
    RUNTIME = "runtime"


class MessageType(str, Enum):
    """Enumeration of message types."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class Provider(str, Enum):
    """Enumeration of LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    LOCAL = "local"
    UNKNOWN = "unknown"
