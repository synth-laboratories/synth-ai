"""Lazy loading Click group - imports commands only when invoked."""

import importlib

import click


class LazyGroup(click.Group):
    """A Click group that lazily loads subcommands on demand."""

    def __init__(self, *args, lazy_subcommands=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._lazy_subcommands = dict(lazy_subcommands or {})
        self._failed_imports = set()

    def list_commands(self, ctx):
        base = super().list_commands(ctx)
        lazy = [k for k in self._lazy_subcommands if k not in self._failed_imports]
        return sorted(set(base + lazy))

    def get_command(self, ctx, cmd_name):
        if cmd_name in self._failed_imports:
            return None
        if cmd_name in self._lazy_subcommands:
            return self._lazy_load(cmd_name)
        return super().get_command(ctx, cmd_name)

    def _lazy_load(self, cmd_name):
        import_path = self._lazy_subcommands[cmd_name]
        modname, attr = import_path.rsplit(":", 1)
        try:
            mod = importlib.import_module(modname)
            return getattr(mod, attr)
        except Exception:
            # Command has missing dependencies, skip it
            self._failed_imports.add(cmd_name)
            return None
