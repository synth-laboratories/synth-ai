from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, cast

import click


def prompt_choice(msg: str, choices: list[str]) -> str:
    print(msg)
    for i, label in enumerate(choices, start=1):
        print(f" [{i}] {label}")
    while True:
        try:
            choice = click.prompt(
                "Select an option",
                default=1,
                type=int,
                show_choices=False
            )
        except click.Abort:
            raise
        if 1 <= choice <= len(choices):
            return choices[choice - 1]
        print(f"Invalid selection. Enter a number between 1 and {len(choices)}")


class PromptedChoiceType(click.Choice):
    """`click.Choice` variant that reprompts with an interactive menu on failure.

    Example
    -------
    ```python
    import click

    from synth_ai.cli.lib.prompts import PromptedChoiceType, PromptedChoiceOption


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
            choice = click.prompt(
                "Select an option",
                default=1,
                type=int,
                show_choices=False
            )
            if 1 <= choice <= len(self.choices):
                print('')
                return cast(str, self.choices[choice - 1])
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

    from synth_ai.cli.lib.prompts import PromptedChoiceType, PromptedChoiceOption


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


def prompt_for_path(
    label: str,
    *,
    available_paths: Sequence[str | Path] | None = None,
    file_type: str | None = None,
    path_type: click.Path | None = None,
) -> Path:
    """Prompt for a filesystem path, optionally offering curated choices."""

    def _normalize_suffix(ext: str | None) -> str | None:
        if not ext:
            return None
        stripped = ext.strip()
        if not stripped:
            return None
        if not stripped.startswith("."):
            stripped = f".{stripped}"
        return stripped.lower()

    def _format_label(text: str) -> str:
        return text.strip() or "path"

    expected_suffix = _normalize_suffix(file_type)
    prompt_label = _format_label(label)

    path_type = path_type or click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        path_type=Path,
    )

    candidates: list[str] = []
    if available_paths:
        seen: set[str] = set()
        for entry in available_paths:
            candidate = str(Path(entry))
            suffix = Path(candidate).suffix.lower()
            if candidate in seen:
                continue
            if expected_suffix and suffix != expected_suffix:
                continue
            seen.add(candidate)
            candidates.append(candidate)

    ctx = click.get_current_context(silent=True)

    while True:
        if candidates:
            click.echo(f"\nPlease choose a {prompt_label}:")
            for index, option in enumerate(candidates, 1):
                click.echo(f" [{index}] {option}")
            custom_index = len(candidates) + 1
            click.echo(f" [{custom_index}] Enter a custom path")

            selection = click.prompt("> ", type=int)
            if 1 <= selection <= len(candidates):
                raw_value = candidates[selection - 1]
            elif selection == custom_index:
                raw_value = click.prompt(prompt_label, type=path_type)
            else:
                click.echo("Invalid selection, please try again")
                continue
        else:
            raw_value = click.prompt(prompt_label, type=path_type)

        try:
            converted = path_type.convert(str(raw_value), None, ctx)
        except click.BadParameter as exc:
            click.echo(str(exc))
            continue

        result = converted if isinstance(converted, Path) else Path(str(converted) if isinstance(converted, bytes) else converted)
        if expected_suffix and result.suffix.lower() != expected_suffix:
            click.echo(f"Expected a {expected_suffix} file. Received: {result}")
            continue
        
        return result


class PromptedPathOption(click.Option):
    """Option that prompts for a filesystem path when omitted."""

    def __init__(
        self,
        *args: Any,
        available_paths: Sequence[str | Path] | Callable[[], Sequence[str | Path]] | None = None,
        file_type: str | None = None,
        path_type: click.Path | None = None,
        prompt_guard: Callable[[click.Context], bool] | None = None,
        **kwargs: Any,
    ) -> None:
        self._available_paths = available_paths
        self._file_type = file_type
        self._path_type = path_type
        self._prompt_guard = prompt_guard
        kwargs.setdefault("prompt", True)
        kwargs.setdefault("prompt_required", True)
        super().__init__(*args, **kwargs)

    def _resolve_available_paths(self) -> Sequence[str | Path] | None:
        if callable(self._available_paths):
            try:
                return self._available_paths()
            except Exception:
                return None
        return self._available_paths

    def prompt_for_value(self, ctx: click.Context) -> Any:
        if not ctx:
            return super().prompt_for_value(ctx)
        if self._prompt_guard is not None:
            try:
                if not self._prompt_guard(ctx):
                    return None
            except Exception:
                return None
        label = self.help or self.name or "path"
        available_paths = self._resolve_available_paths()
        return prompt_for_path(
            label,
            available_paths=available_paths,
            file_type=self._file_type,
            path_type=self._path_type or getattr(self, "type", None),
        )


def ctx_print(msg: str, emit: bool = True) -> None:
    if not emit:
        return None
    print(msg)
    return None