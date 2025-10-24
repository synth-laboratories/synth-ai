from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import click

from .user_config import load_user_config, update_user_config

__all__ = [
    "ensure_key",
    "print_next_step",
    "key_preview",
    "ensure_required_args",
]


def ensure_key(
    key_name: str,
    *,
    prompt_text: str | None = None,
    hidden: bool = True,
    required: bool = True,
    prompt: bool = True,
    success_message: str | None = None,
) -> str:
    """Ensure a secret/API key is available in the environment and user config."""

    prompt_text = prompt_text or f"Enter value for {key_name}"
    current = (os.environ.get(key_name) or load_user_config().get(key_name) or "").strip()

    if current:
        os.environ[key_name] = current
        return current

    if not prompt:
        return ""

    while True:
        try:
            entered = click.prompt(
                prompt_text,
                hide_input=hidden,
                default="",
                show_default=False,
            )
        except (EOFError, KeyboardInterrupt) as exc:
            if required:
                raise click.ClickException(f"{key_name} is required.") from exc
            return ""

        current = entered.strip()
        if current or not required:
            break
        click.echo(f"{key_name.replace('_', ' ')} cannot be empty. Please try again.")

    if not current:
        return ""

    os.environ[key_name] = current
    update_user_config({key_name: current})
    if success_message:
        click.echo(success_message)
    return current


def print_next_step(message: str, lines: Sequence[str]) -> None:
    """Emit a short 'next steps' banner with bullet items."""

    click.echo(f"\n➡️  Next, {message}:")
    for line in lines:
        click.echo(f"   {line}")
    click.echo("")


def key_preview(value: str, label: str) -> str:
    """Return a short descriptor for a secret without leaking the full value."""

    try:
        text = value or ""
        length = len(text)
        prefix = text[:6] if length >= 6 else text
        suffix = text[-5:] if length >= 5 else text
        return f"{label} len={length} prefix={prefix} last5={suffix}"
    except Exception:
        return f"{label} len=0"


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
