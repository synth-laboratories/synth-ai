from __future__ import annotations

import json
import textwrap
from typing import Any

DEFAULT_SYSTEM_TEMPLATE = textwrap.dedent(
    """\
    You are a helpful assistant that can interact with a software repository by issuing shell commands.
    Follow the workflow and formatting guidelines exactly. Every response MUST contain a THOUGHT section
    and exactly one bash command enclosed in a single ```bash``` block.
    """
)

DEFAULT_INSTANCE_TEMPLATE = textwrap.dedent(
    """\
    Please solve this task:

    {{problem_statement}}

    {{instructions}}

    Remember:
    - Explain your reasoning in a THOUGHT section before the command.
    - Provide exactly one bash command wrapped in ```bash``` fences.
    - Use non-interactive flags and prefer deterministic tooling.
    - To finish, run `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached`.
    """
)

DEFAULT_ACTION_TEMPLATE = textwrap.dedent(
    """\
    <returncode>{{ output.returncode }}</returncode>
    {% if output.stdout | length < 10000 %}
    <output>
    {{ output.stdout }}
    </output>
    {% else %}
    <warning>Output truncated ({{ output.stdout | length }} characters)</warning>
    <output_head>{{ output.stdout[:5000] }}</output_head>
    <output_tail>{{ output.stdout[-5000:] }}</output_tail>
    {% endif %}
    """
)


def summarise_history(history: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Return the most recent command history entries, truncated for transport."""

    def _truncate(text: str, max_len: int = 4000) -> str:
        if len(text) <= max_len:
            return text
        head = text[: max_len // 2]
        tail = text[-max_len // 2 :]
        return f"{head}\n... [truncated {len(text) - max_len} chars] ...\n{tail}"

    trimmed: list[dict[str, Any]] = []
    for item in history[-limit:]:
        trimmed.append(
            {
                "command": item.get("command"),
                "returncode": item.get("returncode"),
                "stdout": _truncate(item.get("stdout", "")),
                "duration": item.get("duration"),
            }
        )
    return trimmed


def format_observation(observation: dict[str, Any]) -> str:
    """Simple pretty-printer used by tracing/logging."""

    last = observation.get("last")
    task = observation.get("task", {})
    summary = {
        "instance_id": task.get("instance_id"),
        "step": observation.get("step_idx"),
        "submitted": bool(observation.get("submitted")),
        "last_command": (last or {}).get("command"),
        "returncode": (last or {}).get("returncode"),
    }
    return json.dumps(summary, indent=2, sort_keys=True)

