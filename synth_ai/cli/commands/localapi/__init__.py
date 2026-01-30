"""LocalAPI CLI commands.

Commands:
    synth localapi deploy  - Deploy a LocalAPI to the cloud
"""

import click

from .deploy import deploy


@click.group()
def localapi() -> None:
    """LocalAPI deployment and serving commands."""
    pass


localapi.add_command(deploy)

__all__ = ["localapi"]
