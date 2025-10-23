from __future__ import annotations

import click
from synth_ai.cli.task_app_deploy import deploy_command as main_deploy_command

from .. import setup
from . import configure, init, run


def register(cli):
    @cli.group("demo", invoke_without_command=True)
    @click.pass_context
    def demo(ctx: click.Context):
        if ctx.invoked_subcommand is not None:
            return

        click.echo(
            "\nWelcome to the Synth AI demo!\n"
            "\nRemote (Modal) workflow:\n"
            " 1. synth-ai demo init\n"
            " 2. synth-ai setup\n"
            " 3. synth-ai deploy\n"
            " 4. synth-ai run\n"
            "\nLocal workflow:\n"
            " 1. synth-ai demo init\n"
            " 2. synth-ai setup\n"
            " 3. synth-ai deploy\n"
            " 4. uvx python run_local_rollout_traced.py\n"
            " 5. uvx python export_trace_sft.py --db traces/v3/synth_ai.db --output demo_sft.jsonl\n"
        )
        ctx.exit(0)

    # Register subcommands implemented in sibling modules
    for module in (init, setup, configure, run):
        module.register(demo)

    demo.add_command(main_deploy_command, name="deploy")
