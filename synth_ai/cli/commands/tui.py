"""TUI command group."""

from __future__ import annotations

import click

from synth_ai.tui import run_prompt_learning_tui


def register(cli: click.Group) -> None:
    """Attach TUI commands to the top-level CLI group."""

    @cli.group("tui", invoke_without_command=True)
    @click.pass_context
    @click.option("--job-id", default=None, help="Focus on a specific prompt learning job ID.")
    @click.option("--backend", "backend_base", default=None, help="Backend base URL.")
    @click.option(
        "--api-key",
        default=None,
        envvar="SYNTH_API_KEY",
        help="Synth API key (defaults to SYNTH_API_KEY).",
    )
    @click.option("--refresh-interval", default=5.0, show_default=True, help="Job list refresh.")
    @click.option("--event-interval", default=2.0, show_default=True, help="Event poll interval.")
    @click.option("--limit", default=50, show_default=True, help="Max jobs to show.")
    def tui_group(
        ctx: click.Context,
        job_id: str | None,
        backend_base: str | None,
        api_key: str | None,
        refresh_interval: float,
        event_interval: float,
        limit: int,
    ) -> None:
        """Launch terminal monitoring dashboards."""
        if ctx.invoked_subcommand:
            return
        run_prompt_learning_tui(
            job_id=job_id,
            backend_base=backend_base,
            api_key=api_key,
            refresh_interval=refresh_interval,
            event_interval=event_interval,
            limit=limit,
        )

    @tui_group.command("prompt-learning")
    @click.option("--job-id", default=None, help="Focus on a specific prompt learning job ID.")
    @click.option("--backend", "backend_base", default=None, help="Backend base URL.")
    @click.option(
        "--api-key",
        default=None,
        envvar="SYNTH_API_KEY",
        help="Synth API key (defaults to SYNTH_API_KEY).",
    )
    @click.option("--refresh-interval", default=5.0, show_default=True, help="Job list refresh.")
    @click.option("--event-interval", default=2.0, show_default=True, help="Event poll interval.")
    @click.option("--limit", default=50, show_default=True, help="Max jobs to show.")
    def prompt_learning_tui(
        job_id: str | None,
        backend_base: str | None,
        api_key: str | None,
        refresh_interval: float,
        event_interval: float,
        limit: int,
    ) -> None:
        """Launch the prompt learning monitoring TUI."""
        run_prompt_learning_tui(
            job_id=job_id,
            backend_base=backend_base,
            api_key=api_key,
            refresh_interval=refresh_interval,
            event_interval=event_interval,
            limit=limit,
        )
