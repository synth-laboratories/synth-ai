from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import click


class PromptedChoiceType(click.Choice):
    """`click.Choice` variant that reprompts with an interactive menu on failure.

    Example
    -------
    ```python
    import click

    from synth_ai.utils.cli import PromptedChoiceType, PromptedChoiceOption


    @click.command()
    @click.option(
        "--mode",
        cls=PromptedChoiceOption,
        type=PromptedChoiceType(["sft", "rl"]),
        required=True,
    )
    def train(mode: str) -> None:
        click.echo(f"Selected mode: {mode}")
    ```
    """

    def convert(
        self,
        value: Any,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> str:
        """Validate *value*; prompt for a replacement when it is missing or invalid."""
        if param is None:
            raise RuntimeError("Invalid parameter")
        if ctx is None:
            raise RuntimeError("Invalid context")
        if value in (None, ""):
            return self._prompt_user(param, ctx)
        try:
            return super().convert(value, param, ctx)
        except click.BadParameter:
            cmd_name = self._get_cmd_name(ctx)
            if getattr(param, "opts", None):
                click.echo(f'\n[{cmd_name}] Invalid value "{value}" for {self._get_arg_name(param)}')
            else:
                click.echo(f'\n[{cmd_name}] Invalid value "{value}"')
            return self._prompt_user(param, ctx)

    def _prompt_user(
        self,
        param: click.Parameter,
        ctx: click.Context | None,
    ) -> str:
        """Render a numbered picker and return the user’s selection."""
        arg_name = self._get_arg_name(param)
        click.echo(f"\n[{self._get_cmd_name(ctx)}] Please choose a value for {arg_name}")
        for index, choice in enumerate(self.choices, 1):
            click.echo(f" [{index}] {choice}")
        while True:
            selection = click.prompt("> ", type=int)
            if 1 <= selection <= len(self.choices):
                return cast(str, self.choices[selection - 1])
            click.echo(f"Invalid selection for {arg_name}, please try again")

    def _get_cmd_name(self, ctx: click.Context | None) -> str:
        """Safely extract the current command name for diagnostic output."""
        cmd = getattr(ctx, "command", None) if ctx is not None else None
        if cmd is None:
            return "?"
        name = getattr(cmd, "name", None)
        return name or "?"

    def _get_arg_name(self, param: click.Parameter) -> str:
        """Return the most human-friendly identifier for the parameter."""
        opts = getattr(param, "opts", None)
        if opts:
            return opts[-1]
        name = getattr(param, "name", None)
        if name:
            return name
        human_name = getattr(param, "human_readable_name", None)
        if human_name:
            return human_name
        return "?"


class PromptedChoiceOption(click.Option):
    """`click.Option` subclass that triggers the interactive picker when missing.

    Example
    -------
    ```python
    import click

    from synth_ai.utils.cli import PromptedChoiceType, PromptedChoiceOption


    @click.command()
    @click.option(
        "--mode",
        cls=PromptedChoiceOption,
        type=PromptedChoiceType(["sft", "rl"]),
        required=True,
    )
    def train(mode: str) -> None:
        click.echo(f"Selected mode: {mode}")
    ```
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("prompt", True)
        kwargs.setdefault("prompt_required", True)
        super().__init__(*args, **kwargs)

    def prompt_for_value(self, ctx: click.Context) -> Any:
        """Invoke the choice picker when the option was omitted."""
        option_type = getattr(self, "type", None)
        if isinstance(option_type, PromptedChoiceType):
            return option_type._prompt_user(self, ctx)
        return super().prompt_for_value(ctx)
    

def print_next_step(message: str, lines: Sequence[str]) -> None:
    print(f"\n➡️  Next, {message}:")
    for line in lines:
        print(f"   {line}")
    print("")


__all__ = ["PromptedChoiceType", "PromptedChoiceOption", "print_next_step"]
