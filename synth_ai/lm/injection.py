from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any

Rule = dict[str, Any]

_rules_ctx: contextvars.ContextVar[list[Rule] | None] = contextvars.ContextVar(
    "injection_rules", default=None
)


def set_injection_rules(rules: list[Rule]):
    """Set prompt-injection rules for the current context and return a reset token.

    Each rule must be a dict with at least keys: "find" and "replace" (strings).
    Optional: "roles" as a list of role names to scope the replacement.
    """
    if not isinstance(rules, list) or not all(
        isinstance(r, dict) and "find" in r and "replace" in r for r in rules
    ):
        raise ValueError("Injection rules must be a list of dicts with 'find' and 'replace'")
    return _rules_ctx.set(rules)


def get_injection_rules() -> list[Rule] | None:
    """Get the current context's injection rules, if any."""
    return _rules_ctx.get()


def clear_injection_rules(token) -> None:
    """Reset the injection rules to the previous value using the provided token."""
    _rules_ctx.reset(token)


def apply_injection(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply ordered substring replacements to text parts of messages in place.

    - Only modifies `str` content or list parts where `part["type"] == "text"`.
    - Honors optional `roles` scoping in each rule.
    - Returns the input list for convenience.
    """
    rules = get_injection_rules()
    if not rules:
        return messages

    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str):
            new_content = content
            for r in rules:
                allowed_roles = r.get("roles")
                if allowed_roles is not None and role not in allowed_roles:
                    continue
                new_content = new_content.replace(str(r["find"]), str(r["replace"]))
            m["content"] = new_content
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text = part.get("text", "")
                    new_text = text
                    for r in rules:
                        allowed_roles = r.get("roles")
                        if allowed_roles is not None and role not in allowed_roles:
                            continue
                        new_text = new_text.replace(str(r["find"]), str(r["replace"]))
                    part["text"] = new_text
    return messages


@contextmanager
def injection_rules_ctx(rules: list[Rule]):
    """Context manager to temporarily apply injection rules within the block."""
    tok = set_injection_rules(rules)
    try:
        yield
    finally:
        clear_injection_rules(tok)
