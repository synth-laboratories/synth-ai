"""CLI subcommands for Synth AI.

This package exposes a top-level Click group named ``cli`` compatible with the
pyproject entry point ``synth_ai.cli:cli``.
"""

from __future__ import annotations

try:
    from ._typer_patch import patch_typer_make_metavar

    patch_typer_make_metavar()
except Exception:
    pass


from .root import cli  # new canonical CLI entrypoint

# Register subcommands from this package onto the group
try:
    from . import watch as _watch

    _watch.register(cli)
except Exception:
    pass
try:
    from . import recent as _recent

    _recent.register(cli)
except Exception:
    pass
try:
    from . import calc as _calc

    _calc.register(cli)
except Exception:
    pass
try:
    from . import demo as _demo

    _demo.register(cli)
except Exception:
    pass
try:
    from . import turso as _turso

    _turso.register(cli)
except Exception:
    pass
try:
    from synth_ai.api.train import register as _train_register

    _train_register(cli)
except Exception:
    pass

from importlib import import_module

from .task_app_serve import serve_command
from .task_apps import task_app_group

import_module(".task_app_list", __name__)
import_module(".task_app_deploy", __name__)
import_module(".task_app_modal_serve", __name__)

cli.add_command(serve_command)
cli.add_command(task_app_group, name="task-app")
cli.add_command(task_app_group.commands["deploy"], name="deploy")
cli.add_command(task_app_group.commands["modal-serve"], name="modal-serve")
