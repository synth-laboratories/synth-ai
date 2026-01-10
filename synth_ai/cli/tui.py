"""TUI command."""

import click


def _run_tui(
    *,
    job_id: str | None,
    backend_base: str | None,
    api_key: str | None,
    refresh_interval: float,
    event_interval: float,
    limit: int,
) -> None:
    from synth_ai.tui import run_prompt_learning_tui

    run_prompt_learning_tui(
        job_id=job_id,
        backend_base=backend_base,
        api_key=api_key,
        refresh_interval=refresh_interval,
        event_interval=event_interval,
        limit=limit,
    )


@click.group(invoke_without_command=True)
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
def tui(
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
    _run_tui(
        job_id=job_id,
        backend_base=backend_base,
        api_key=api_key,
        refresh_interval=refresh_interval,
        event_interval=event_interval,
        limit=limit,
    )


@tui.command()
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
def prompt_learning(
    job_id: str | None,
    backend_base: str | None,
    api_key: str | None,
    refresh_interval: float,
    event_interval: float,
    limit: int,
) -> None:
    """Launch the prompt learning monitoring TUI."""
    _run_tui(
        job_id=job_id,
        backend_base=backend_base,
        api_key=api_key,
        refresh_interval=refresh_interval,
        event_interval=event_interval,
        limit=limit,
    )
