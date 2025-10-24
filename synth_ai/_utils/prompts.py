from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = ["ensure_required_args"]

PromptFunc = Callable[[str], str]
CoerceFunc = Callable[[str], Any]


def ensure_required_args(
    args: argparse.Namespace,
    required: Mapping[str, str],
    *,
    input_func: PromptFunc | None = None,
    coerce: Mapping[str, CoerceFunc] | None = None,
    allow_empty: Sequence[str] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> argparse.Namespace:
    """Prompt for missing namespace attributes defined in *required*."""

    prompt = input_func or (lambda message: input(message))
    coerce_map = dict(coerce or {})
    allow_empty_set = set(allow_empty or ())
    default_map = dict(defaults or {})
    _sentinel = object()

    for attr, message in required.items():
        value = getattr(args, attr, None)
        if value not in (None, ""):
            continue

        default = default_map.get(attr, _sentinel)
        if default is _sentinel:
            prompt_text = message if message.endswith(": ") else f"{message}: "
        else:
            label = message.rstrip(": ")
            prompt_text = f"{label} [{default}]: "

        while True:
            try:
                raw = prompt(prompt_text)
            except EOFError:
                raw = ""
            except KeyboardInterrupt:
                raise SystemExit(1) from None
            raw = raw.strip()

            if not raw:
                if default is not _sentinel:
                    value = default
                    setattr(args, attr, value)
                    break
                if attr not in allow_empty_set:
                    print("Value required; please try again.")
                    continue
                value = raw
                setattr(args, attr, value)
                break

            if attr in coerce_map and raw:
                try:
                    value = coerce_map[attr](raw)
                except Exception as exc:
                    print(f"Invalid value: {exc}")
                    continue
            else:
                value = raw

            setattr(args, attr, value)
            break

    return args
