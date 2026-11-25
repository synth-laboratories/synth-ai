"""Command-line argument grammar and root verb parser for `synth`."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, NoReturn

from typing_extensions import override

from ._cli_surfaces import configure
from ._generated.models import MessageKind
from .environments import Environment

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

DEFAULT_WAIT_SECONDS = 3600.0
# Root options that take a value; the command is the first token after them.
ROOT_VALUE_OPTIONS = frozenset({"--base-url", "--env"})

COMMANDS: dict[str, str] = {
    "content": "Save output content",
    "labels": "Update run labels",
    "files": "Files",
    "uploads": "Upload records",
    "keys": "API keys",
    "organization": "Organization",
    "inspect": "Run details",
    "families": "Topology catalog",
    "runs": "List runs",
    "login": "Sign in",
    "whoami": "Signed-in account",
    "logout": "Sign out",
    "profiles": "Run profiles",
    "choices": "Per-role launch choices",
    "upload": "Upload files",
    "run": "Launch a run",
    "status": "Run state",
    "wait": "Wait for a run to finish",
    "logs": "Run events",
    "message": "Send a run message",
    "messages": "Run messages",
    "tasks": "Run tasks",
    "ls": "Recent runs",
    "pause": "Pause a run",
    "resume": "Resume a run",
    "cancel": "Cancel a run",
    "outputs": "Run outputs",
    "download": "Download run outputs",
    "report": "Run report",
    "allocation": "Per-actor allocation",
    "result": "Run answer",
    "usage": "Usage",
    "billing": "Balance",
    "limits": "Limits",
    "plan": "Special Free plan requests",
    "grants": "MCP OAuth grants (connected apps)",
    "promo": "Promotional credit",
    "artifacts": "Interactive artifacts",
    "experiments": "Multi-experiment plans",
    "research-log": "Project research log",
}

EPILOG = """exit codes:
  0  success
  1  client or usage error
  2  wait timeout; the run is still running
  3  run finished failed or cancelled
  4  server refusal
  5  run paused

--json prints the SDK record as one JSON document on stdout."""


class UsageError(Exception):
    """CLI syntax or usage refusal surfaced before API execution."""


class Parser(argparse.ArgumentParser):
    """ArgumentParser that raises UsageError instead of exiting the process."""

    @override
    def error(self, message: str) -> NoReturn:
        raise UsageError(message)


def index_requested(raw: Sequence[str]) -> bool:
    """Return True when the first positional CLI verb token is `index`."""
    tokens = iter(raw)
    for item in tokens:
        if item in ROOT_VALUE_OPTIONS:
            next(tokens, None)
        elif not item.startswith("-"):
            return item == "index"
    return False


def _configure_auth_and_account(
    created: Mapping[str, argparse.ArgumentParser], common: argparse.ArgumentParser
) -> None:
    runs = created["runs"].add_subparsers(dest="runs_action", required=True)
    runs_list = runs.add_parser("list", parents=[common])
    runs_list.add_argument("--limit", type=int, default=20)
    runs_list.add_argument("--cursor")
    runs_list.add_argument("--state")
    login = created["login"].add_mutually_exclusive_group()
    login.add_argument("--no-browser", action="store_true")
    login.add_argument("--device", action="store_true")
    promo = created["promo"].add_subparsers(dest="promo_action", required=True)
    redeem = promo.add_parser("redeem", parents=[common])
    redeem.add_argument("code")
    redeem.add_argument("--idempotency-key")
    created["profiles"].add_argument("--research-task")
    choices = created["choices"]
    choices.add_argument("--profile")
    choices.add_argument("--domicile", choices=("us_only", "us_or_europe", "unrestricted"))
    choices.add_argument("--parallel-workers", type=int, metavar="N")
    choices.add_argument("--wall", type=int, metavar="SECONDS")
    choices.add_argument("--budget", help="list choices this maximum charge cannot afford")
    created["usage"].add_argument("run_id", nargs="?", metavar="RUN")
    plan_common = Parser(add_help=False)
    plan_common.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    plan = created["plan"].add_subparsers(dest="plan_action", required=True)
    plan_request = plan.add_parser("request", parents=[plan_common])
    plan_request.add_argument("--use-case", required=True)
    plan_request.add_argument("--link", action="append", default=[], dest="links")
    plan_request.add_argument("--idempotency-key")
    plan.add_parser("requests", parents=[plan_common])
    plan.add_parser("withdraw", parents=[plan_common]).add_argument("request_id", metavar="REQUEST")
    grants = created["grants"].add_subparsers(dest="grants_action", required=True)
    grants.add_parser("revoke", parents=[plan_common]).add_argument("grant_id", metavar="GRANT")


def _configure_run_launch(run: argparse.ArgumentParser, *, stdin_piped: bool) -> None:
    task = run.add_mutually_exclusive_group(required=not stdin_piped)
    task.add_argument("goal", nargs="?")
    task.add_argument(
        "--goal-file",
        "--task-file",
        dest="task_file",
        metavar="PATH",
        help="read goal from UTF-8 file PATH, or '-' for stdin",
    )
    run.add_argument("--budget", required=True)
    run.add_argument("--idempotency-key")
    run.add_argument("--profile")
    run.add_argument("--optimizer", metavar="auto|TOPOLOGY@VERSION")
    run.add_argument(
        "--max-tasks",
        type=int,
        metavar="N",
        help="Swarms work-plan tasks, 1..4096, at least --parallel-workers (needs swarms_v2 pin)",
    )
    run.add_argument(
        "--max-repairs-per-task",
        type=int,
        metavar="N",
        help="Swarms repairs per rejected task, 0..32 (needs swarms_v2 pin)",
    )
    run.add_argument("--continue-from", metavar="RUN")
    run.add_argument("--research-task")
    run.add_argument("--domicile", choices=("us_only", "us_or_europe", "unrestricted"))
    run.add_argument("--file", action="append", default=[], metavar="PATH")
    run.add_argument(
        "--wall", type=int, metavar="SECONDS", help="unpaused run time, 1..86400 seconds"
    )
    run.add_argument(
        "--parallel-workers",
        type=int,
        metavar="N",
        help="workers the planner may run at once, 1..32 (default 1); the budget must fund all N",
    )
    run.add_argument("--tokens", type=int)
    run.add_argument("--pause-retention", type=int, metavar="SECONDS")
    run.add_argument("--label", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument(
        "--resources",
        metavar="PATH",
        help="JSON array of resource requests, sent as the launch's resources",
    )
    run.add_argument(
        "--assign",
        action="append",
        default=[],
        metavar="ACTOR.DIMENSION=ID",
        help="role choice from `synth choices`; DIMENSION is inference, sandbox or option",
    )
    run.add_argument(
        "--planner-decides",
        action="append",
        default=[],
        metavar="ACTOR.DIMENSION",
        help="leave this worker's inference or sandbox to the planner",
    )
    run.add_argument("--menu-revision", help="menu_revision of the `synth choices` read")
    run.add_argument("--wait", action="store_true")
    run.add_argument("--timeout", type=float, default=DEFAULT_WAIT_SECONDS)


def _configure_run_operations(created: Mapping[str, argparse.ArgumentParser]) -> None:
    upload = created["upload"]
    upload.add_argument("paths", nargs="+", metavar="PATH")
    upload.add_argument("--directory", action="store_true")
    upload.add_argument("--idempotency-key")
    for name in ("status", "pause", "resume", "cancel", "report", "allocation", "result"):
        created[name].add_argument("run_id", metavar="RUN")
    for name in ("pause", "resume", "cancel"):
        created[name].add_argument("--idempotency-key")
    wait = created["wait"]
    wait.add_argument("run_id", metavar="RUN")
    wait.add_argument("--timeout", type=float, default=DEFAULT_WAIT_SECONDS)
    logs = created["logs"]
    logs.add_argument("run_id", metavar="RUN")
    logs.add_argument("-f", "--follow", action="store_true")
    logs.add_argument("--limit", type=int, default=100)
    logs.add_argument("--cursor")
    for name in ("tasks", "outputs"):
        command = created[name]
        command.add_argument("run_id", metavar="RUN")
        command.add_argument("--limit", type=int, default=20)
        command.add_argument("--cursor")
    message = created["message"]
    message.add_argument("run_id", metavar="RUN")
    message.add_argument("text", metavar="TEXT")
    message.add_argument("--kind", choices=tuple(MessageKind), default=MessageKind.GUIDANCE)
    message.add_argument("--idempotency-key")
    messages = created["messages"]
    messages.add_argument("run_id", metavar="RUN")
    messages.add_argument("--cursor")
    messages.add_argument("--limit", type=int, default=20)
    message_mode = messages.add_mutually_exclusive_group()
    message_mode.add_argument("--follow", action="store_true")
    message_mode.add_argument("--message", dest="message_id", metavar="MESSAGE")
    ls = created["ls"]
    ls.add_argument("--limit", type=int, default=20)
    ls.add_argument("--cursor")
    ls.add_argument("--state")
    download = created["download"]
    download.add_argument("run_id", metavar="RUN")
    download.add_argument("output_id", nargs="?", metavar="OUTPUT")
    download.add_argument("-o", "--out", default=".", metavar="DIR")
    limits = created["limits"]
    limits.add_argument("run_id", nargs="?", metavar="RUN|set")
    limits.add_argument("set_run_id", nargs="?", metavar="RUN")
    limits.add_argument("--expected-revision", type=int)
    limits.add_argument("--budget")
    limits.add_argument("--wall", type=int)
    limits.add_argument("--tokens", type=int)
    limits.add_argument("--pause-retention", type=int)
    limits.add_argument("--resume", action="store_true")
    limits.add_argument("--reason")
    limits.add_argument("--idempotency-key")


def parser(*, stdin_piped: bool = False) -> argparse.ArgumentParser:
    """Build the top-level CLI parser for all public customer commands."""
    root = Parser(prog="synth", epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    root.add_argument("--base-url")
    root.add_argument("--env", dest="environment", type=Environment, choices=tuple(Environment))
    common = Parser(add_help=False)
    common.add_argument("--json", action="store_true", help="print the SDK record as JSON")
    verbs = root.add_subparsers(dest="command", required=True, metavar="COMMAND")
    created = {
        name: verbs.add_parser(name, help=label, parents=[common])
        for name, label in COMMANDS.items()
    }
    _configure_auth_and_account(created, common)
    _configure_run_launch(created["run"], stdin_piped=stdin_piped)
    _configure_run_operations(created)
    configure(created, common)
    return root
