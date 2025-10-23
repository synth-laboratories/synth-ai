from __future__ import annotations

import argparse
from collections.abc import Callable

from .. import setup
from . import configure, deploy, init, run


def _add_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    names: list[str],
    configure_parser: Callable[[argparse.ArgumentParser], None],
) -> None:
    for name in names:
        parser = subparsers.add_parser(name)
        configure_parser(parser)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="synth-ai-demo")
    sub = parser.add_subparsers(dest="cmd")

    def _setup_opts(p: argparse.ArgumentParser) -> None:
        p.set_defaults(func=lambda _args: setup.setup())

    _add_parser(sub, ["setup", "demo.setup"], _setup_opts)

    def _init_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--template", type=str, default=None, help="Template id to instantiate")
        p.add_argument("--dest", type=str, default=None, help="Destination directory for files")
        p.add_argument("--force", action="store_true", help="Overwrite existing files in destination")
        p.set_defaults(
            func=lambda args: init.run_init(args.template, args.dest, args.force),
        )

    _add_parser(sub, ["init", "demo.init"], _init_opts)

    def _deploy_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--local", action="store_true", help="Run local FastAPI instead of Modal deploy")
        p.add_argument("--app", type=str, default=None, help="Path to Modal app.py for uv run modal deploy")
        p.add_argument("--name", type=str, default=None, help="Modal app name")
        p.add_argument("--script", type=str, default=None, help="Path to deploy_task_app.sh (optional legacy)")
        p.set_defaults(
            func=lambda args: deploy.run_deploy(
                local=args.local, app=args.app, name=args.name, script=args.script
            )
        )

    _add_parser(sub, ["deploy", "demo.deploy"], _deploy_opts)

    def _run_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=str, default=None, help="Path to TOML config (skip prompt)")
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--group-size", type=int, default=None)
        p.add_argument("--model", type=str, default=None)
        p.add_argument("--timeout", type=int, default=600)
        p.add_argument("--dry-run", action="store_true", help="Print request body and exit")
        p.set_defaults(
            func=lambda args: run.run_job(
                config=args.config,
                batch_size=args.batch_size,
                group_size=args.group_size,
                model=args.model,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
        )

    _add_parser(sub, ["run", "demo.run"], _run_opts)

    def _configure_opts(p: argparse.ArgumentParser) -> None:
        p.set_defaults(func=lambda _args: configure.run_configure())

    _add_parser(sub, ["configure", "demo.configure"], _configure_opts)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
