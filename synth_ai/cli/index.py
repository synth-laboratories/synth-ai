"""Private Synth Index research intake commands."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group()
def index() -> None:
    """Inspect and submit private research contributions."""


@index.group()
def research() -> None:
    """Intake a verified research export conversion."""


@research.command("preview")
@click.argument("conversion", type=click.Path(exists=True, file_okay=False, path_type=Path))
def preview(conversion: Path) -> None:
    """Check exact package bytes and show provenance, evidence and review state."""
    from synth_ai.sdk.index.research_intake import preview_conversion

    try:
        _, _, summary = preview_conversion(conversion)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(summary, indent=2, sort_keys=True))


@research.command("submit")
@click.argument("conversion", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--state-file",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Persistent retry identity; defaults beside the conversion package.",
)
@click.option("--backend-url", envvar="SYNTH_BACKEND_URL", help="Synth backend base URL.")
@click.option(
    "--api-key", envvar="SYNTH_API_KEY", help="Synth API key from an authorized local environment."
)
@click.option(
    "--finalize-only",
    is_flag=True,
    help="Upload and finalize a private draft while holding submission for a separate review decision.",
)
def submit(
    conversion: Path,
    state_file: Path | None,
    backend_url: str | None,
    api_key: str | None,
    finalize_only: bool,
) -> None:
    """Resume private intake; the server controls review eligibility."""
    from httpx import HTTPError

    from synth_ai import SynthClient
    from synth_ai.core.errors import SynthError
    from synth_ai.sdk.index.research_intake import submit_conversion

    if not api_key:
        raise click.ClickException("SYNTH_API_KEY or --api-key is required for research intake")
    state_path = state_file or conversion / ".research-intake-state.json"
    try:
        with SynthClient(api_key=api_key, base_url=backend_url) as client:
            result = submit_conversion(
                client.index, conversion, state_path, finalize_only=finalize_only
            )
    except (OSError, ValueError, KeyError, TypeError, SynthError, HTTPError) as error:
        raise click.ClickException(str(error)) from error
    click.echo(json.dumps(result, indent=2, sort_keys=True))
