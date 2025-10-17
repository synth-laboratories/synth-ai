"""Tool schema for mini-SWE command execution."""

from __future__ import annotations

RUN_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": (
            "Execute a bash command inside the task workspace. Use this for all shell "
            "operations including editing files, running tests, and submitting results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute. Must be non-empty.",
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 600,
                    "description": (
                        "Optional timeout (seconds) for the command. Defaults to the environment "
                        "timeout if omitted."
                    ),
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_patch",
        "description": (
            "Finish the task and submit the final patch. Call this once you believe the "
            "fix is complete and tests pass."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "Optional submission command. Defaults to "
                        "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached` "
                        "if omitted."
                    ),
                }
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

TOOLS_SCHEMA = [RUN_COMMAND_TOOL, SUBMIT_TOOL]

# Compatibility: some OpenAI reasoning models (e.g., gpt-5) insist on calling
# a generic function (e.g., 'interact' or 'interact_many'). Provide stubs so
# vendor requests do not 400 on unknown function names; the policy will map
# these calls to concrete environment tools.
COMPAT_INTERACT_TOOL = {
    "type": "function",
    "function": {
        "name": "interact",
        "description": "Compatibility shim for models that call a generic 'interact' tool.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        },
    },
}

COMPAT_INTERACT_MANY_TOOL = {
    "type": "function",
    "function": {
        "name": "interact_many",
        "description": "Compatibility shim for models that call 'interact_many'.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": True,
        },
    },
}

# Append compatibility tools last so preferred tools remain first in the list
TOOLS_SCHEMA.extend([COMPAT_INTERACT_TOOL, COMPAT_INTERACT_MANY_TOOL])

