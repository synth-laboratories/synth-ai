from __future__ import annotations

import click

from . import configure, deploy, init, run, setup


def register(cli):
    @cli.group("demo", invoke_without_command=True)
    @click.pass_context
    def demo(ctx: click.Context):
        if ctx.invoked_subcommand is not None:
            return

        click.echo(
            "\nWelcome to the Synth AI demo!\n"
            "\nRemote (Modal) workflow:\n"
            "  synth-ai demo init   # choose math-modal\n"
            "  synth-ai demo setup  # pair the SDK and store API keys\n"
            "  synth-ai demo deploy # deploy to Modal (public HTTPS URL)\n"
            "  synth-ai demo run    # submit the sample RL job to Synth cloud\n"
            "\nLocal workflow:\n"
            "  synth-ai demo init   # choose a local-friendly template (e.g. crafter-local)\n"
            "  synth-ai demo setup  # write keys to the local .env\n"
            "  synth-ai demo deploy # start the FastAPI server (keep terminal open)\n"
            "  uvx python run_local_rollout_traced.py\n"
            "  uvx python export_trace_sft.py --db traces/v3/synth_ai.db --output demo_sft.jsonl\n"
            "  # Optional lighter run\n"
            "  uvx python run_local_rollout.py\n"
            "  # Skip 'demo run' unless your task app is publicly reachable\n"
        )
        ctx.exit(0)

    # Register subcommands implemented in sibling modules
    for module in (init, setup, deploy, configure, run):
        module.register(demo)
