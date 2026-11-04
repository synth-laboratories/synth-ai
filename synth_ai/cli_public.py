"""The ``synth`` command: Swarms, account and billing.

Only command groups whose product is in ``PUBLIC_PRODUCTS`` are registered, and
every registered verb is a row of ``synth_ai.surface.VERBS`` (one MCP tool each).
Other groups (``cli_swarms`` preflight, ``cli_account`` legacy reads) stay
importable but are not registered.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from synth_ai.client import SynthClient
from synth_ai.errors import SynthError
from synth_ai.products import PUBLIC_PRODUCTS
from synth_ai.public.errors import RunNotOrchestraAuthorityError, SwarmsAPIError, client_error
from synth_ai.public.models import RunOutput, RunSnapshot

ClientFactory = Callable[[], SynthClient]
Emit = Callable[[str], None]

# login, logout and auth status belong to the account product.
AUTH_COMMANDS = frozenset({"login", "logout", "auth"})

TASK_FILENAMES = ("task.md", "TASK.md", "task.txt")
MAXIMUM_TASK_BYTES = 8192
PROGRAM = re.compile(r"^(?P<id>[A-Za-z0-9_.-]{1,128})@(?P<version>[1-9][0-9]{0,9})$")


def parse_program(value: str) -> dict[str, Any]:
    """``swarms.general@1`` -> the launch ``program`` object ``{"family": {id, version}}``."""
    match = PROGRAM.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError("program must look like FAMILY@VERSION")
    return {"family": {"id": match["id"], "version": int(match["version"])}}


def read_task(path: str) -> str:
    """A task bundle is a text file, or a directory holding ``task.md``.

    The Swarms v1 contract has no file-upload route, so a directory holding
    anything besides the task file is refused rather than partly sent."""
    root = Path(path)
    if root.is_dir():
        entries = sorted(p.name for p in root.iterdir() if not p.name.startswith("."))
        names = [name for name in TASK_FILENAMES if name in entries]
        if len(names) != 1:
            raise client_error("task_file_missing", "swarms.run", cause="one of task.md, task.txt")
        if len(entries) != 1:
            raise client_error("task_inputs_unsupported", "swarms.run",
                               cause="contract has no file-upload route")
        root = root / names[0]
    try:
        raw = root.read_bytes()
    except OSError as error:
        raise client_error("task_unreadable", "swarms.run", cause=type(error).__name__) from error
    if not raw.strip() or len(raw) > MAXIMUM_TASK_BYTES:
        raise client_error("task_size_invalid", "swarms.run", cause=f"1..{MAXIMUM_TASK_BYTES} bytes")
    return raw.decode("utf-8")


def _json(value: BaseModel) -> str:
    return value.model_dump_json(exclude_none=True)


def save_outputs(outputs: list[RunOutput], directory: str) -> list[str]:
    """Write each output record as ``<output_id>.json``; never overwrite."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for output in outputs:
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", output.output_id) + ".json"
        path = target / name
        with path.open("x", encoding="utf-8") as stream:
            stream.write(output.model_dump_json(indent=2))
        written.append(str(path))
    return written


def run_streaming(client: SynthClient, args: argparse.Namespace, emit: Emit) -> RunSnapshot:
    """Launch (or attach), stream events by cursor, then collect outputs.

    Every page prints its ``resume_cursor``; rerun with ``--attach RUN --cursor C``
    to resume after an interruption."""
    if args.attach:
        run_id = args.attach
    else:
        message = read_task(args.task) if args.task else args.message
        if not message:
            raise client_error("message_required", "swarms.run", cause="MESSAGE or --task")
        launched = client.swarms.launch(
            message, args.budget, project_id=args.project_id, program=args.program
        )
        run_id = launched.run_id
        emit(json.dumps({"run_id": run_id, "state": launched.state}))
    try:
        for page in client.swarms.follow(run_id, cursor=args.cursor,
                                         timeout_seconds=args.timeout):
            for event in page.items:
                emit(_json(event))
            emit(json.dumps({"resume_cursor": page.resume_cursor}))
        closed = client.swarms.get(run_id)
    except RunNotOrchestraAuthorityError:
        # Imported runs have no event history; wait on the snapshot instead.
        closed = client.swarms.wait(run_id, timeout_seconds=args.timeout)
    outputs = client.swarms.outputs(run_id).items
    if args.out:
        for path in save_outputs(outputs, args.out):
            emit(json.dumps({"saved": path}))
    return closed


def _swarms(group: argparse._SubParsersAction) -> None:
    swarms = group.add_parser("swarms").add_subparsers(dest="swarms_command", required=True)

    run = swarms.add_parser(
        "run", help="launch, stream events and collect outputs",
        description="Launch a run (or attach to one), stream its events as JSON lines with a "
                    "resume_cursor per page, then print the closed run.",
    )
    run.add_argument("message", nargs="?", help="what the run should do")
    run.add_argument("--task", metavar="PATH", help="task file, or directory holding task.md")
    run.add_argument("--program", type=parse_program, metavar="FAMILY@VERSION",
                     help="e.g. swarms.general@1")
    run.add_argument("--budget", metavar="USD", help="maximum charge in USD")
    run.add_argument("--project-id", help="default: SYNTH_PROJECT_ID")
    run.add_argument("--out", metavar="DIR", help="write each output record to DIR")
    run.add_argument("--attach", metavar="RUN_ID", help="follow an existing run")
    run.add_argument("--cursor", help="resume_cursor to resume from (with --attach)")
    run.add_argument("--timeout", type=float, metavar="SECONDS",
                     help="stop following after SECONDS; the run keeps going")
    run.set_defaults(streaming=True)

    launch = swarms.add_parser("launch", help="launch and print the run")
    launch.add_argument("message", help="what the run should do")
    launch.add_argument("--budget", required=True, metavar="USD", help="maximum charge in USD")
    launch.add_argument("--program", type=parse_program, metavar="FAMILY@VERSION")
    launch.add_argument("--project-id", help="default: SYNTH_PROJECT_ID")
    launch.set_defaults(handler=lambda c, a: c.swarms.launch(
        a.message, a.budget, project_id=a.project_id, program=a.program))

    for name in ("status", "cancel", "outputs"):
        swarms.add_parser(name).add_argument("run_id")
    swarms.choices["status"].set_defaults(handler=lambda c, a: c.swarms.get(a.run_id))
    swarms.choices["cancel"].set_defaults(handler=lambda c, a: c.swarms.cancel(a.run_id))
    swarms.choices["outputs"].set_defaults(handler=lambda c, a: c.swarms.outputs(a.run_id))

    output = swarms.add_parser("output")
    output.add_argument("run_id")
    output.add_argument("output_id")
    output.set_defaults(handler=lambda c, a: c.swarms.output(a.run_id, a.output_id))

    wait = swarms.add_parser("wait")
    wait.add_argument("run_id")
    wait.add_argument("--timeout", type=float, metavar="SECONDS")
    wait.set_defaults(handler=lambda c, a: c.swarms.wait(a.run_id, timeout_seconds=a.timeout))

    events = swarms.add_parser("events")
    events.add_argument("run_id")
    events.add_argument("--cursor", help="next_cursor or resume_cursor of a previous page")
    events.add_argument("--limit", type=int)
    events.set_defaults(handler=lambda c, a: c.swarms.events(
        a.run_id, cursor=a.cursor, limit=a.limit))

    listing = swarms.add_parser("list")
    listing.add_argument("--cursor")
    listing.add_argument("--limit", type=int)
    listing.add_argument("--project-id")
    listing.set_defaults(handler=lambda c, a: c.swarms.list(
        cursor=a.cursor, limit=a.limit, project_id=a.project_id))


def _account(group: argparse._SubParsersAction) -> None:
    account = group.add_parser("account")
    account.set_defaults(handler=lambda c, a: c.account.get())
    for name in AUTH_COMMANDS:
        # Parsed by cli_auth; registered here so they appear in help.
        group.add_parser(name, add_help=False)


def _billing(group: argparse._SubParsersAction) -> None:
    billing = group.add_parser("billing").add_subparsers(dest="billing_command", required=True)
    billing.add_parser("balance").set_defaults(handler=lambda c, a: c.billing.balance())
    history = billing.add_parser("history")
    history.add_argument("--cursor", help="next_cursor of a previous page")
    history.add_argument("--limit", type=int)
    history.set_defaults(handler=lambda c, a: c.billing.history(cursor=a.cursor, limit=a.limit))
    billing.add_parser("plans").set_defaults(handler=lambda c, a: c.billing.plans())


GROUPS: dict[str, Callable[[argparse._SubParsersAction], None]] = {
    "swarms": _swarms,
    "account": _account,
    "billing": _billing,
}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="synth")
    commands = root.add_subparsers(dest="command", required=True)
    for product in PUBLIC_PRODUCTS:
        GROUPS[product](commands)
    return root


def _subparsers(node: argparse.ArgumentParser) -> argparse._SubParsersAction | None:
    return next((a for a in node._actions if isinstance(a, argparse._SubParsersAction)), None)


def registered_commands() -> frozenset[str]:
    action = _subparsers(parser())
    assert action is not None
    return frozenset(action.choices)


def registered_verbs() -> frozenset[tuple[str, ...]]:
    """Every leaf command path, excluding the auth commands."""
    verbs: set[tuple[str, ...]] = set()
    top = _subparsers(parser())
    assert top is not None
    for name, node in top.choices.items():
        if name in AUTH_COMMANDS:
            continue
        child = _subparsers(node)
        if child is None:
            verbs.add((name,))
        else:
            verbs.update((name, leaf) for leaf in child.choices)
    return frozenset(verbs)


def describe_error(error: SwarmsAPIError) -> str:
    """One JSON line: the typed refusal, its outcome and the HTTP status."""
    return json.dumps({
        "error": error.error.model_dump(exclude_none=True),
        "outcome": error.outcome,
        "status": error.http_status,
    })


def main(
    argv: list[str] | None = None,
    *,
    client_factory: ClientFactory = SynthClient,
    emit: Emit = print,
) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in AUTH_COMMANDS and "account" in PUBLIC_PRODUCTS:
        from synth_ai.cli_auth import main as auth_main

        return auth_main(arguments)
    args = parser().parse_args(arguments)
    if getattr(args, "streaming", False) and not args.attach and args.budget is None:
        parser().error("swarms run requires --budget unless --attach is given")
    try:
        with client_factory() as client:
            if getattr(args, "streaming", False):
                emit(_json(run_streaming(client, args, emit)))
            else:
                emit(_json(args.handler(client, args)))
    except SwarmsAPIError as error:
        # Refusals are reported once; the SDK never retries a 4xx.
        print(describe_error(error), file=sys.stderr)
        return 1
    except SynthError as error:
        print(json.dumps({"error": {"code": error.code or "sdk_error", "service": "synth-ai",
                                    "stage": args.command, "cause": type(error).__name__},
                          "outcome": None, "status": None}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
