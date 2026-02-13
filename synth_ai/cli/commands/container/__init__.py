"""Container CLI commands.

Commands:
    synth container deploy  - Deploy a Container to the cloud
"""

import click

from .deploy import deploy


@click.group()
def container() -> None:
    """Container deployment and serving commands."""
    pass


container.add_command(deploy)

__all__ = ["container"]
