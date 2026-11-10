"""The `synth` command line: task verbs over the typed client."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NoReturn, TypeVar

import httpx
from pydantic import BaseModel, TypeAdapter
from typing_extensions import override

from . import _cli_view as view
from . import auth
from ._generated.models import InputFile, RunInputs, RunSnapshot
from ._generated.policy import MESSAGE_MAX_BYTES
from ._validation import message_text
from ._version import __version__
from .aclient import AsyncSynthClient
from .errors import (
    ResponseContractError,
    SwarmsAPIError,
    SynthError,
    TransportFailureError,
    UpgradeRequiredError,
    WaitTimeoutError,
)
from .swarms import AsyncSwarms

EXIT_OK = 0
EXIT_CLIENT_ERROR = 1
EXIT_WAIT_TIMEOUT = 2
EXIT_RUN_UNSUCCESSFUL = 3
EXIT_SERVER_REFUSAL = 4
UNSUCCESSFUL_STATES = frozenset({"failed", "cancelled"})
DEFAULT_WAIT_SECONDS = 3600.0

COMMANDS: dict[str, str] = {
    "login": "Sign in",
    "whoami": "Signed-in account",
    "logout": "Sign out",
    "profiles": "Run profiles",
    "upload": "Upload files",
    "run": "Launch a run",
    "status": "Run state",
    "wait": "Wait for a run to finish",
    "logs": "Run events",
    "tasks": "Run tasks",
    "ls": "Recent runs",
    "pause": "Pause a run",
    "resume": "Resume a run",
    "cancel": "Cancel a run",
    "outputs": "Run outputs",
    "download": "Download run outputs",
    "report": "Run report",
    "result": "Run answer",
    "usage": "Usage",
    "billing": "Balance",
    "limits": "Limits",
}

EPILOG = """exit codes:
  0  success
  1  client or usage error
  2  wait timeout; the run is still running
  3  run finished failed or cancelled
  4  server refusal

--json prints the SDK record as one JSON document on stdout."""


class UsageError(Exception):
    pass


class Parser(argparse.ArgumentParser):
    @override
    def error(self, message: str) -> NoReturn:
        raise UsageError(message)


def parser() -> argparse.ArgumentParser:
    root = Parser(prog="synth", epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    root.add_argument("--base-url")
    common = Parser(add_help=False)
    common.add_argument("--json", action="store_true", help="print the SDK record as JSON")
    verbs = root.add_subparsers(dest="command", required=True, metavar="COMMAND")

    created = {
        name: verbs.add_parser(name, help=label, parents=[common])
        for name, label in COMMANDS.items()
    }

    def add(name: str) -> argparse.ArgumentParser:
        return created[name]

    add("login").add_argument("--no-browser", action="store_true")
    add("whoami")
    add("logout")
    add("profiles").add_argument("--research-task")
    add("upload").add_argument("paths", nargs="+", metavar="PATH")
    run = add("run")
    run.add_argument("goal")
    run.add_argument("--budget", required=True)
    run.add_argument("--profile")
    run.add_argument("--research-task")
    run.add_argument("--file", action="append", default=[], metavar="PATH")
    run.add_argument(
        "--wall", type=int, metavar="SECONDS", help="run wall-time bound, 1..3600 seconds"
    )
    run.add_argument(
        "--parallel-workers",
        type=int,
        metavar="N",
        help="workers the planner may run at once, 1..10 (default 1); the budget must fund all N",
    )
    run.add_argument("--label", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--wait", action="store_true")
    run.add_argument("--timeout", type=float, default=DEFAULT_WAIT_SECONDS)
    for name in ("status", "pause", "resume", "cancel", "report", "result"):
        add(name).add_argument("run_id", metavar="RUN")
    wait = add("wait")
    wait.add_argument("run_id", metavar="RUN")
    wait.add_argument("--timeout", type=float, default=DEFAULT_WAIT_SECONDS)
    logs = add("logs")
    logs.add_argument("run_id", metavar="RUN")
    logs.add_argument("-f", "--follow", action="store_true")
    for name in ("tasks", "outputs"):
        command = add(name)
        command.add_argument("run_id", metavar="RUN")
        command.add_argument("--limit", type=int, default=20)
        command.add_argument("--cursor")
    logs.add_argument("--limit", type=int, default=100)
    logs.add_argument("--cursor")
    ls = add("ls")
    ls.add_argument("--limit", type=int, default=20)
    ls.add_argument("--cursor")
    ls.add_argument("--state")
    download = add("download")
    download.add_argument("run_id", metavar="RUN")
    download.add_argument("output_id", nargs="?", metavar="OUTPUT")
    download.add_argument("-o", "--out", default=".", metavar="DIR")
    add("usage").add_argument("run_id", nargs="?", metavar="RUN")
    add("billing")
    add("limits").add_argument("run_id", nargs="?", metavar="RUN")
    return root


def read_task(path: str) -> str:
    """Read a task text file, or a bundle directory holding exactly one task file."""
    source = Path(path)
    if source.is_dir():
        found = [p for p in source.iterdir() if not p.name.startswith(".")]
        if len(found) != 1 or found[0].name not in {"task.md", "TASK.md", "task.txt"}:
            raise ValueError("task bundle must contain exactly one task text file")
        source = found[0]
    raw = source.read_bytes()
    if not raw.strip() or len(raw) > MESSAGE_MAX_BYTES:
        raise ValueError(f"task must contain 1..{MESSAGE_MAX_BYTES} bytes")
    return message_text(raw.decode("utf-8"))


def parse_labels(values: Sequence[str]) -> dict[str, str] | None:
    labels: dict[str, str] = {}
    for value in values:
        key, sep, item = value.partition("=")
        if not sep or not key:
            raise ValueError("label must be KEY=VALUE: " + value)
        labels[key] = item
    return labels or None


M = TypeVar("M", bound=BaseModel)


class Output:
    def __init__(self, as_json: bool) -> None:
        self.as_json = as_json

    def record(self, value: M, render: Callable[[M], str]) -> None:
        print(
            value.model_dump_json(exclude_unset=True) if self.as_json else render(value), flush=True
        )

    def records(self, values: Sequence[M], render: Callable[[Sequence[M]], str]) -> None:
        if self.as_json:
            print(
                "[" + ",".join(v.model_dump_json(exclude_unset=True) for v in values) + "]",
                flush=True,
            )
        else:
            print(render(values), flush=True)


def finished_code(snapshot: RunSnapshot) -> int:
    return EXIT_RUN_UNSUCCESSFUL if snapshot.state in UNSUCCESSFUL_STATES else EXIT_OK


async def execute(args: argparse.Namespace, http_client: httpx.AsyncClient | None) -> int:
    out = Output(args.json)
    async with AsyncSynthClient(base_url=args.base_url, http_client=http_client) as client:
        swarms = client.swarms
        command = str(args.command)
        if command == "whoami":
            out.record(await client.account.get(), view.account)
        elif command == "profiles":
            out.record(await swarms.profiles(research_task=args.research_task), view.profiles)
        elif command == "upload":
            out.records([await swarms.upload_file(path) for path in args.paths], view.files)
        elif command == "run":
            labels = parse_labels(args.label)
            uploaded = [await swarms.upload_file(path) for path in args.file]
            inputs = (
                RunInputs(
                    files=[
                        InputFile(
                            file_id=f.file_id,
                            revision_id=f.revision_id,
                            digest_sha256=f.digest_sha256,
                        )
                        for f in uploaded
                    ]
                )
                if uploaded
                else None
            )
            created = await swarms.launch(
                args.goal,
                budget=args.budget,
                profile=args.profile,
                research_task=args.research_task,
                wall_time_seconds=args.wall,
                parallel_workers=args.parallel_workers,
                inputs=inputs,
                labels=labels,
            )
            if not args.wait:
                out.record(created, lambda s: s.run_id)
                return EXIT_OK
            if not args.json:
                print(created.run_id, flush=True)
            final = await swarms.wait(created.run_id, timeout=args.timeout)
            out.record(final, lambda s: "state  " + s.state)
            if not args.json:
                finished = await swarms.result(final)
                if finished.answer is not None:
                    print(finished.answer, flush=True)
                closing = view.charge_line(finished.charge)
                if closing is not None:
                    print(closing, file=sys.stderr, flush=True)
            return finished_code(final)
        elif command == "status":
            out.record(await swarms.get(args.run_id), view.snapshot)
        elif command == "wait":
            final = await swarms.wait(args.run_id, timeout=args.timeout)
            out.record(final, view.snapshot)
            return finished_code(final)
        elif command == "logs":
            if args.follow:
                await follow(swarms, args.run_id, args.json)
            else:
                page = await swarms.events(args.run_id, cursor=args.cursor, limit=args.limit)
                out.record(page, view.events)
        elif command == "tasks":
            page = await swarms.tasks(args.run_id, cursor=args.cursor, limit=args.limit)
            out.record(page, view.tasks)
        elif command == "ls":
            page = await swarms.list(limit=args.limit, cursor=args.cursor, state=args.state)
            out.record(page, view.runs)
        elif command == "pause":
            out.record(await swarms.pause(args.run_id), view.receipt)
        elif command == "resume":
            out.record(await swarms.resume(args.run_id), view.receipt)
        elif command == "cancel":
            out.record(await swarms.cancel(args.run_id), view.snapshot)
        elif command == "outputs":
            page = await swarms.outputs(args.run_id, cursor=args.cursor, limit=args.limit)
            out.record(page, view.outputs)
        elif command == "download":
            out.records(await download(swarms, args.run_id, args.output_id, args.out), view.saved)
        elif command == "report":
            out.record(await swarms.report(args.run_id), view.report)
        elif command == "result":
            finished = await swarms.result(args.run_id)
            print(view.result_json(finished) if args.json else view.result(finished), flush=True)
            closing = view.charge_line(finished.charge)
            if closing is not None and not args.json:
                print(closing, file=sys.stderr, flush=True)
        elif command == "usage":
            if args.run_id is None:
                out.record(await client.account.usage_summary(), view.usage_summary)
            else:
                page = await client.account.usage_records(run_id=args.run_id)
                out.record(page, view.usage_records)
        elif command == "billing":
            out.record(await client.billing.balance(), view.balance)
        elif command == "limits":
            if args.run_id is None:
                out.record(await client.account.limits(), view.limits)
            else:
                out.record(await swarms.limits(args.run_id), view.run_limits)
        else:
            raise UsageError("unknown command " + command)
    return EXIT_OK


async def follow(swarms: AsyncSwarms, run_id: str, as_json: bool) -> None:
    """Stream events; the stored checkpoint advances only after a line is written."""
    checkpoint = view.load_checkpoint(run_id)
    async for item in swarms.follow(run_id, checkpoint=checkpoint):
        print(item.model_dump_json(exclude_unset=True) if as_json else view.event(item), flush=True)
        view.save_checkpoint(checkpoint)
    view.save_checkpoint(checkpoint)


async def download(
    swarms: AsyncSwarms, run_id: str, output_id: str | None, directory: str
) -> list[view.SavedOutput]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    saved: list[view.SavedOutput] = []
    async for item in swarms.iter_outputs(run_id):
        if output_id is not None and item.output_id != output_id:
            continue
        reference = item.reference or {}
        revision, digest = reference.get("revision_id"), reference.get("digest_sha256")
        if revision is None or digest is None:
            if output_id is not None:
                raise ValueError("output " + item.output_id + " has no content yet")
            continue
        name = Path(item.filename).name
        if name in {"", ".", ".."}:
            raise ValueError("output " + item.output_id + " has no usable filename")
        path = await swarms.download_output(
            run_id, item.output_id, str(target / name), revision_id=revision, digest_sha256=digest
        )
        saved.append(view.SavedOutput(output=item, path=str(path)))
    if output_id is not None and not saved:
        raise ValueError("output not found: " + output_id)
    return saved


def paste_redirect(authorization: str) -> str:
    print(authorization, file=sys.stderr, flush=True)
    return str(sys.stdin.readline())


def run_auth(args: argparse.Namespace) -> int:
    out = Output(args.json)
    if args.command == "login":
        auth.login(
            args.base_url,
            open_browser=not args.no_browser,
            redirect_source=paste_redirect if args.no_browser else None,
        )
    else:
        auth.logout(args.base_url)
    out.record(auth.status(args.base_url), view.auth_status)
    return EXIT_OK


def fail(code: str, detail: str, as_json: bool, problem: BaseModel | None = None) -> None:
    if as_json:
        body = (
            problem.model_dump_json()
            if problem is not None
            else TypeAdapter(dict[str, str]).dump_json({"code": code, "detail": detail}).decode()
        )
        print(body, file=sys.stderr, flush=True)
    else:
        print(code + ": " + detail, file=sys.stderr, flush=True)


def upgrade_hint(error: UpgradeRequiredError) -> str:
    minimum = "" if error.minimum_version is None else f" (minimum {error.minimum_version})"
    return f"synth-ai {__version__} is no longer supported{minimum}; run: {error.UPGRADE_HINT}"


def main(argv: Sequence[str] | None = None, *, http_client: httpx.AsyncClient | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in raw
    try:
        args = parser().parse_args(raw)
    except UsageError as error:
        fail("usage_error", str(error), as_json)
        return EXIT_CLIENT_ERROR
    try:
        if args.command in {"login", "logout"}:
            return run_auth(args)
        return asyncio.run(execute(args, http_client))
    except WaitTimeoutError as error:
        fail(error.code, str(error), as_json)
        return EXIT_WAIT_TIMEOUT
    except (TransportFailureError, ResponseContractError) as error:
        fail(error.problem.code, error.problem.detail, as_json, error.problem)
        return EXIT_CLIENT_ERROR
    except UpgradeRequiredError as error:
        fail(error.problem.code, error.problem.detail, as_json, error.problem)
        if not as_json:
            print(upgrade_hint(error), file=sys.stderr, flush=True)
        return EXIT_SERVER_REFUSAL
    except SwarmsAPIError as error:
        fail(error.problem.code, error.problem.detail, as_json, error.problem)
        return EXIT_SERVER_REFUSAL
    except SynthError as error:
        fail(error.code, str(error), as_json)
        return EXIT_CLIENT_ERROR
    except (ValueError, OSError, UsageError) as error:
        fail("client_error", str(error), as_json)
        return EXIT_CLIENT_ERROR


if __name__ == "__main__":
    sys.exit(main())
