#!/usr/bin/env python3
"""
CLI: Interactive TUI dashboard for Synth AI.
"""

import os

import click
from rich.console import Console


def register(cli):
    @cli.command()
    @click.option(
        "--url",
        "db_url",
        default="sqlite+libsql://http://127.0.0.1:8080",
        help="Database URL (default: sqlite+libsql://http://127.0.0.1:8080)",
    )
    @click.option("--debug", is_flag=True, help="Enable debug logging")
    def tui(db_url: str, debug: bool):
        """Launch interactive TUI dashboard showing experiments, balance, and active runs."""
        console = Console()

        # Import here to avoid circular imports and handle optional dependencies
        try:
            from synth_ai.tui.dashboard import main as tui_main
        except (ImportError, ModuleNotFoundError) as e:
            console.print("[red]Error:[/red] TUI dashboard not available.")
            console.print(f"Missing dependencies: {e}")
            console.print("Install with: pip install textual")
            return
        except Exception as e:
            # Handle other import errors (like missing libsql, type annotation issues, etc.)
            console.print("[red]Error:[/red] TUI dashboard not available.")
            console.print("This may be due to missing dependencies or Python version compatibility.")
            console.print("Try: pip install textual libsql")
            console.print("If using Python < 3.10, you may need to update Python or install eval_type_backport.")
            return

        # Set environment variables for the TUI to use
        os.environ.setdefault("TUI_DB_URL", db_url)
        if debug:
            os.environ["TUI_DEBUG"] = "1"

        # Run the TUI by calling the module directly with sanitized argv
        try:
            tui_args = ["--url", db_url]
            if debug:
                tui_args.append("--debug")
            tui_main(tui_args)
        except KeyboardInterrupt:
            console.print("\n[blue]TUI closed.[/blue]")
        except Exception as e:
            console.print(f"\n[red]Error running TUI:[/red] {e}")
            if debug:
                raise
