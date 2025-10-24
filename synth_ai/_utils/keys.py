from __future__ import annotations

import os

import click

from .user_config import load_user_config, update_user_config

__all__ = ["ensure_key"]


def ensure_key(
    key_name: str,
    *,
    prompt_text: str | None = None,
    hidden: bool = True,
    required: bool = True,
    prompt: bool = True,
    success_message: str | None = None,
) -> str:
    """Ensure a secret/API key is available in the environment and user config.

    Args:
        key_name: Environment/config key to ensure.
        prompt_text: Message displayed when prompting for the key.
        hidden: Hide input while prompting (default True).
        required: When True, empty responses are rejected and missing keys raise.
        prompt: When False, the user is not prompted and missing keys return "".
        success_message: Optional message printed when a key is saved.

    Returns:
        The resolved key value (may be empty when not required).
    """

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
