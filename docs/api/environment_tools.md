---
title: 'Environment Tools'
description: 'Tool system for environment interactions'
---

# Environment Tools

The tools module provides a system for defining and using tools within environments.

## Classes

### AbstractTool

Base class for all tools in the environment.

**Signature:** `AbstractTool(name: str, description: str)`

**Methods:**
- `execute(args)` - Execute the tool with given arguments
- `validate(args)` - Validate arguments before execution

### EnvToolCall

Represents a tool call within an environment.

**Signature:** `EnvToolCall(tool: str, args: dict)`

**Attributes:**
- `tool` - Name of the tool to call
- `args` - Arguments for the tool call

### ToolResult

Represents the result of a tool execution.

**Signature:** `ToolResult(content: Any, success: bool)`

**Attributes:**
- `content` - The result content
- `success` - Whether the tool execution was successful

