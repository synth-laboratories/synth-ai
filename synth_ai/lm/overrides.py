from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any

from synth_ai.lm.injection import (
    apply_injection as _apply_injection,
)
from synth_ai.lm.injection import (
    clear_injection_rules,
    set_injection_rules,
)

# Context to hold a list of override specs to evaluate per-call
# Each spec shape (minimal v1):
# {
#   "match": {"contains": "atm", "role": "user" | "system" | None},
#   "injection_rules": [{"find": str, "replace": str, "roles": Optional[List[str]]}],
#   "params": { ... api params to override ... },
#   "tools": { ... optional tools overrides ... },
# }
_override_specs_ctx: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("override_specs", default=None)
)

# ContextVars actually applied for the specific call once matched
_param_overrides_ctx: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "param_overrides", default=None
)
_tool_overrides_ctx: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "tool_overrides", default=None
)
_current_override_label_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "override_label", default=None
)


def set_override_specs(specs: list[dict[str, Any]]):
    if not isinstance(specs, list):
        raise ValueError("override specs must be a list of dicts")
    return _override_specs_ctx.set(specs)


def get_override_specs() -> list[dict[str, Any]] | None:
    return _override_specs_ctx.get()


def clear_override_specs(token) -> None:
    _override_specs_ctx.reset(token)


def _matches(spec: dict[str, Any], messages: list[dict[str, Any]]) -> bool:
    match = spec.get("match") or {}
    contains = match.get("contains")
    role = match.get("role")  # optional
    if not contains:
        # no match criteria means always apply
        return True
    contains_l = str(contains).lower()
    for m in messages:
        if role and m.get("role") != role:
            continue
        c = m.get("content")
        if isinstance(c, str) and contains_l in c.lower():
            return True
        if isinstance(c, list):
            for part in c:
                if part.get("type") == "text" and contains_l in str(part.get("text", "")).lower():
                    return True
    return False


def resolve_override_for_messages(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    specs = get_override_specs() or []
    for spec in specs:
        try:
            if _matches(spec, messages):
                return spec
        except Exception:
            # On matcher errors, skip spec
            continue
    return None


def apply_injection(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Delegate to injection.apply_injection
    return _apply_injection(messages)


def apply_param_overrides(api_params: dict[str, Any]) -> dict[str, Any]:
    ov = _param_overrides_ctx.get()
    if not ov:
        return api_params
    # Shallow merge only known keys users provided
    for k, v in ov.items():
        api_params[k] = v
    return api_params


def apply_tool_overrides(api_params: dict[str, Any]) -> dict[str, Any]:
    """Apply tool overrides to OpenAI/Anthropic-like api_params in place.

    Supports keys under spec["tools"]:
      - set_tools: replace tools entirely
      - add_tools: append tools
      - remove_tools_by_name: remove by function name
      - tool_choice: set tool_choice param
    """
    ov = _tool_overrides_ctx.get()
    if not ov:
        return api_params
    tov = ov.get("tools") if isinstance(ov, dict) else None
    if tov:
        tools = api_params.get("tools")
        if "set_tools" in tov:
            tools = tov["set_tools"]
        if "add_tools" in tov:
            tools = (tools or []) + tov["add_tools"]
        if "remove_tools_by_name" in tov and tools:
            names = set(tov["remove_tools_by_name"])  # function names
            new_tools = []
            for t in tools:
                try:
                    # OpenAI dict style
                    fn = t.get("function", {}).get("name") if isinstance(t, dict) else None
                except Exception:
                    fn = None
                # If BaseTool objects slipped through
                if fn is None:
                    fn = getattr(t, "function_name", None)
                if fn is None or fn not in names:
                    new_tools.append(t)
            tools = new_tools
        if tools is not None:
            api_params["tools"] = tools
        if "tool_choice" in tov:
            api_params["tool_choice"] = tov["tool_choice"]
    return api_params


@contextmanager
def use_overrides_for_messages(messages: list[dict[str, Any]]):
    """Resolve an override spec against messages and apply its contexts within the scope.

    - Sets injection rules and param overrides if present on the matched spec.
    - Yields, then resets ContextVars to previous values.
    """
    spec = resolve_override_for_messages(messages) or {}
    inj_rules = spec.get("injection_rules")
    params = spec.get("params")
    inj_tok = None
    param_tok = None
    tool_tok = None
    label_tok = None
    try:
        if inj_rules:
            inj_tok = set_injection_rules(inj_rules)
        if params:
            param_tok = _param_overrides_ctx.set(params)
        tools = spec.get("tools")
        if tools:
            tool_tok = _tool_overrides_ctx.set({"tools": tools})
        lbl = spec.get("label")
        if lbl:
            label_tok = _current_override_label_ctx.set(str(lbl))
        yield
    finally:
        if inj_tok is not None:
            clear_injection_rules(inj_tok)
        if param_tok is not None:
            _param_overrides_ctx.reset(param_tok)
        if tool_tok is not None:
            _tool_overrides_ctx.reset(tool_tok)
        if label_tok is not None:
            _current_override_label_ctx.reset(label_tok)


def get_current_override_label() -> str | None:
    return _current_override_label_ctx.get()


class LMOverridesContext:
    """Context manager to register per-call override specs.

    Usage:
        with LMOverridesContext([
            {"match": {"contains": "atm", "role": "user"}, "injection_rules": [...], "params": {...}},
            {"match": {"contains": "refund"}, "params": {"temperature": 0.0}},
        ]):
            run_pipeline()
    """

    def __init__(self, override_specs: list[dict[str, Any]] | None | dict[str, Any] = None):
        if isinstance(override_specs, dict):
            override_specs = [override_specs]
        self._specs = override_specs or []
        self._tok = None

    def __enter__(self):
        self._tok = set_override_specs(self._specs)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._tok is not None:
            clear_override_specs(self._tok)
