from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import click


def format_error_message(
    summary: str,
    *,
    context: str,
    problem: str,
    impact: str,
    solutions: Sequence[Tuple[str, str]],
    docs_url: Optional[str] = None,
    emoji: str = "❌",
) -> str:
    """Create a consistent, human-friendly error message body."""
    lines: List[str] = [
        f"{emoji} {summary}",
        "",
        f"Context: {context}",
        f"Problem: {problem}",
        f"Impact: {impact}",
        "",
        "Solutions:",
    ]
    for idx, (solution, explanation) in enumerate(solutions, start=1):
        lines.append(f"  {idx}. {solution}")
        if explanation:
            lines.append(f"     → {explanation}")

    if docs_url:
        lines.extend(("", f"Learn more: {docs_url}"))

    return "\n".join(lines)


def get_required_value(
    name: str,
    *,
    cli_value: Optional[str] = None,
    env_value: Optional[str] = None,
    config_value: Optional[str] = None,
    default: Optional[str] = None,
    docs_url: Optional[str] = None,
) -> str:
    """Resolve a required value or raise a UsageError with actionable guidance."""

    def _normalized(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped if stripped else None

    resolved = (
        _normalized(cli_value)
        or _normalized(env_value)
        or _normalized(config_value)
        or _normalized(default)
    )
    if resolved is not None:
        return resolved

    sources: List[str] = []
    if cli_value is None:
        sources.append(f"--{name.replace('_', '-')} flag")
    if env_value is None:
        sources.append(f"{name.upper()} environment variable")
    if config_value is None:
        sources.append("config file")

    message = format_error_message(
        summary=f"Missing required value: {name}",
        context="Command requires this value",
        problem="No value supplied by CLI flags, environment variables, or config",
        impact="Command cannot continue without the required input",
        solutions=[
            (f"Use --{name.replace('_', '-')}", "Provide the value explicitly via CLI"),
            (f"Set {name.upper()}=<value>", "Populate it via environment variables"),
            (f"Add {name} = \"<value>\" to config", "Persist it in the TOML config"),
        ],
        docs_url=docs_url,
    )
    raise click.UsageError(message)
