"""Harbor CLI commands for deployment management.

Commands:
    synth harbor upload     - Upload a new deployment
    synth harbor list       - List deployments
    synth harbor status     - Get deployment status
    synth harbor build      - Trigger a build
    synth harbor instances  - Manage instances
    synth harbor run        - Run rollouts
"""

import click

from .build import build
from .instances import instances
from .list import list_deployments
from .run import run
from .status import status
from .upload import upload


@click.group()
def harbor():
    """Harbor deployment management commands."""
    pass


harbor.add_command(upload)
harbor.add_command(list_deployments, name="list")
harbor.add_command(status)
harbor.add_command(build)
harbor.add_command(instances)
harbor.add_command(run)


__all__ = ["harbor"]
