"""The `synth` command line: task verbs over the typed client."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NoReturn, TypeVar

import httpx
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError
from typing_extensions import override

from . import _cli_view as view
from . import auth
from ._cli_surfaces import configure, execute_resource
from ._generated.models import (
    InputFile,
    LaunchRequest,
    OptimizerPin,
    OptimizerRequest,
    OwnerResourceRequest,
    RoleAssignment,
    RunInputs,
    RunSnapshot,
)
from ._generated.policy import MESSAGE_MAX_BYTES
from ._validation import message_text
from ._version import __version__
from .aclient import AsyncSynthClient
from .environments import Environment, resolve_base_url
from .errors import (
    DirectoryUploadError,
    ResponseContractError,
    RunPausedError,
    SwarmsAPIError,
    SynthError,
    TransportFailureError,
    UpgradeRequiredError,
    UploadTransferError,
    WaitTimeoutError,
)
from .swarms import AsyncSwarms

EXIT_OK = 0
EXIT_CLIENT_ERROR = 1
EXIT_WAIT_TIMEOUT = 2
EXIT_RUN_UNSUCCESSFUL = 3
EXIT_SERVER_REFUSAL = 4
EXIT_RUN_PAUSED = 5
UNSUCCESSFUL_STATES = frozenset({"failed", "cancelled"})
DEFAULT_WAIT_SECONDS = 3600.0

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
    pass


class Parser(argparse.ArgumentParser):
    @override
    def error(self, message: str) -> NoReturn:
        raise UsageError(message)


def parse_optimizer(value: str | None) -> OptimizerRequest | None:
    """Parse the public optimizer selector; Core owns resolution and availability."""
    if value is None:
        return None  # Omitted selection uses Core's declared environment default.
    if value == "auto":
        return "auto"
    topology, separator, version = value.partition("@")
    if separator == "" or not version.isascii() or not version.isdecimal():
        raise UsageError("optimizer must be auto or TOPOLOGY@VERSION")
    return OptimizerPin.model_validate({"topology": topology, "version": int(version)})


RESOURCES = TypeAdapter(list[OwnerResourceRequest])


def parse_resources(path: str | None) -> list[OwnerResourceRequest] | None:
    """Read the public owner resource requests (the launch body's `resources` array)."""
    if path is None:
        return None
    try:
        document = Path(path).read_bytes()
    except OSError as error:
        raise UsageError(
            "--resources file unreadable: " + path + ": " + error.__class__.__name__
        ) from error
    try:
        resources = RESOURCES.validate_json(document)
    except ValidationError as error:
        first = error.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "document"
        raise UsageError(
            "--resources must be a JSON array of resource requests: " + where + ": " + first["msg"]
        ) from error
    return resources


def mutation_key(action: str, run_id: str, values: dict[str, JsonValue] | None = None) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            json.dumps(
                ["synth-cli", action, run_id, values], sort_keys=True, separators=(",", ":")
            ),
        )
    )


def parser() -> argparse.ArgumentParser:
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

    def add(name: str) -> argparse.ArgumentParser:
        return created[name]

    runs = add("runs").add_subparsers(dest="runs_action", required=True)
    runs_list = runs.add_parser("list", parents=[common])
    runs_list.add_argument("--limit", type=int, default=20)
    runs_list.add_argument("--cursor")
    runs_list.add_argument("--state")
    add("login").add_argument("--no-browser", action="store_true")
    add("whoami")
    add("logout")
    add("profiles").add_argument("--research-task")
    choices = add("choices")
    choices.add_argument("--profile")
    choices.add_argument("--domicile", choices=("us_only", "us_or_europe", "unrestricted"))
    choices.add_argument("--parallel-workers", type=int, metavar="N")
    choices.add_argument("--wall", type=int, metavar="SECONDS")
    choices.add_argument("--budget", help="list choices this maximum charge cannot afford")
    upload = add("upload")
    upload.add_argument("paths", nargs="+", metavar="PATH")
    upload.add_argument("--directory", action="store_true")
    upload.add_argument("--idempotency-key")
    run = add("run")
    run.add_argument("goal")
    run.add_argument("--budget", required=True)
    run.add_argument("--idempotency-key")
    run.add_argument("--profile")
    run.add_argument("--optimizer", metavar="auto|TOPOLOGY@VERSION")
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
        help="workers the planner may run at once, 1..10 (default 1); the budget must fund all N",
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
    for name in ("status", "pause", "resume", "cancel", "report", "allocation", "result"):
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
    configure(created, common)
    limits = add("limits")
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
    for name in ("pause", "resume", "cancel"):
        add(name).add_argument("--idempotency-key")
    plan_common = Parser(add_help=False)
    plan_common.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    plan = add("plan").add_subparsers(dest="plan_action", required=True)
    plan_request = plan.add_parser("request", parents=[plan_common])
    plan_request.add_argument("--use-case", required=True)
    plan_request.add_argument("--link", action="append", default=[], dest="links")
    plan_request.add_argument("--idempotency-key")
    plan.add_parser("requests", parents=[plan_common])
    plan.add_parser("withdraw", parents=[plan_common]).add_argument("request_id", metavar="REQUEST")
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


ASSIGN_FIELDS = {
    "inference": "inference_profile",
    "sandbox": "sandbox_profile",
    "option": "sandbox_option",
}


def parse_role_assignments(
    assign: Sequence[str], planner_decides: Sequence[str]
) -> dict[str, RoleAssignment] | None:
    """--assign worker.inference=luna-high and --planner-decides worker_2.sandbox."""
    roles: dict[str, dict[str, JsonValue]] = {}
    for value in assign:
        target, sep, choice = value.partition("=")
        actor, dot, dimension = target.partition(".")
        if not sep or not dot or not actor or not choice or dimension not in ASSIGN_FIELDS:
            raise ValueError("--assign must be ACTOR.inference|sandbox|option=ID: " + value)
        fields = roles.setdefault(actor, {})
        if ASSIGN_FIELDS[dimension] in fields:
            raise ValueError("--assign names " + target + " twice")
        fields[ASSIGN_FIELDS[dimension]] = choice
    for value in planner_decides:
        actor, dot, dimension = value.partition(".")
        if not dot or not actor or dimension not in ("inference", "sandbox"):
            raise ValueError("--planner-decides must be ACTOR.inference|sandbox: " + value)
        decided = roles.setdefault(actor, {}).setdefault("planner_decides", [])
        if not isinstance(decided, list) or dimension in decided:
            raise ValueError("--planner-decides names " + value + " twice")
        decided.append(dimension)
    return {actor: RoleAssignment.model_validate(fields) for actor, fields in roles.items()} or None


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
    async with AsyncSynthClient(
        base_url=args.base_url, environment=args.environment, http_client=http_client
    ) as client:
        swarms = client.swarms
        command = str(args.command)
        if command in {
            "files",
            "uploads",
            "keys",
            "organization",
            "inspect",
            "families",
            "content",
            "labels",
        } or (command == "billing" and args.resource_action is not None):
            value = await execute_resource(client, args)
            print(
                "null" if value is None else value.model_dump_json(exclude_unset=True), flush=True
            )
        elif command == "whoami":
            out.record(await client.account.get(), view.account)
        elif command == "profiles":
            out.record(await swarms.profiles(research_task=args.research_task), view.profiles)
        elif command == "choices":
            out.record(
                await swarms.launch_choices(
                    profile=args.profile,
                    domicile_requirement=args.domicile,
                    parallel_workers=args.parallel_workers,
                    wall_time_seconds=args.wall,
                    budget=args.budget,
                ),
                view.launch_choices,
            )
        elif command == "upload":
            if args.directory:
                if len(args.paths) != 1 or not args.idempotency_key:
                    raise UsageError("directory upload requires one path and --idempotency-key")
                files = await swarms.upload_directory(
                    args.paths[0], idempotency_key=args.idempotency_key
                )
            else:
                if args.idempotency_key:
                    raise UsageError("--idempotency-key requires --directory")
                files = [await swarms.upload_file(path) for path in args.paths]
            out.records(files, view.files)
        elif command == "run":
            if args.idempotency_key is not None and not 1 <= len(args.idempotency_key) <= 128:
                raise UsageError("idempotency key must contain 1 to 128 characters")
            # Every flag is validated before any upload: a refused limit or label must
            # not leave uploaded files behind. Files join the validated request after.
            request = swarms.prepare_launch(
                args.goal,
                budget=args.budget,
                profile=args.profile,
                research_task=args.research_task,
                domicile_requirement=args.domicile,
                wall_time_seconds=args.wall,
                maximum_tokens=args.tokens,
                pause_retention_seconds=args.pause_retention,
                parallel_workers=args.parallel_workers,
                inputs=None,
                labels=parse_labels(args.label),
                resources=parse_resources(args.resources),
                routing=None,
                guidance=None,
                role_assignments=parse_role_assignments(args.assign, args.planner_decides),
                launch_menu_revision=args.menu_revision,
                optimizer=parse_optimizer(args.optimizer),
            )
            uploaded = [
                await swarms.upload_file(
                    path,
                    idempotency_key=(
                        mutation_key("run-upload", args.idempotency_key, {"index": index})
                        if args.idempotency_key
                        else None
                    ),
                )
                for index, path in enumerate(args.file)
            ]
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
            if inputs is not None:
                request = LaunchRequest.model_validate(
                    {
                        **request.model_dump(exclude_none=True),
                        "inputs": inputs.model_dump(exclude_none=True),
                    }
                )
            created = await swarms.launch_request(request, idempotency_key=args.idempotency_key)
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
        elif command in {"ls", "runs"}:
            page = await swarms.list(limit=args.limit, cursor=args.cursor, state=args.state)
            out.record(page, view.runs)
        elif command == "pause":
            out.record(
                await swarms.pause(
                    args.run_id,
                    idempotency_key=args.idempotency_key or mutation_key("pause", args.run_id),
                ),
                view.receipt,
            )
        elif command == "resume":
            out.record(
                await swarms.resume(
                    args.run_id,
                    idempotency_key=args.idempotency_key or mutation_key("resume", args.run_id),
                ),
                view.receipt,
            )
        elif command == "cancel":
            out.record(
                await swarms.cancel(args.run_id, idempotency_key=args.idempotency_key),
                view.snapshot,
            )
        elif command == "outputs":
            page = await swarms.outputs(args.run_id, cursor=args.cursor, limit=args.limit)
            out.record(page, view.outputs)
        elif command == "download":
            out.records(await download(swarms, args.run_id, args.output_id, args.out), view.saved)
        elif command == "report":
            out.record(await swarms.report(args.run_id), view.report)
        elif command == "allocation":
            out.record(await swarms.allocation(args.run_id), view.allocation)
        elif command == "result":
            finished = await swarms.result(args.run_id)
            print(view.result_json(finished) if args.json else view.result(finished), flush=True)
            closing = view.charge_line(finished.charge)
            if closing is not None and not args.json:
                print(closing, file=sys.stderr, flush=True)
            markdown = view.result_file_line(finished)
            if markdown is not None and not args.json:
                print(markdown, file=sys.stderr, flush=True)
        elif command == "usage":
            if args.run_id is None:
                out.record(await client.account.usage_summary(), view.usage_summary)
            else:
                page = await client.account.usage_records(run_id=args.run_id)
                out.record(page, view.usage_records)
        elif command == "billing":
            out.record(await client.billing.balance(), view.balance)
        elif command == "limits":
            if args.run_id == "set":
                if args.set_run_id is None or args.expected_revision is None:
                    raise UsageError("limits set requires RUN and --expected-revision")
                values = dict(
                    expected_revision=args.expected_revision,
                    budget=args.budget,
                    wall_time_seconds=args.wall,
                    maximum_tokens=args.tokens,
                    pause_retention_seconds=args.pause_retention,
                    resume=args.resume,
                    reason=args.reason,
                )
                key = args.idempotency_key or mutation_key("limits", args.set_run_id, values)
                out.record(
                    await swarms.update_limits(args.set_run_id, **values, idempotency_key=key),
                    view.run_limits,
                )
            elif args.set_run_id is not None:
                raise UsageError("unexpected argument after RUN")
            elif args.run_id is None:
                out.record(await client.account.limits(), view.limits)
            else:
                out.record(await swarms.limits(args.run_id), view.run_limits)
        elif command == "plan":
            if args.plan_action == "request":
                # The use case is customer content: it goes to the SDK and nowhere else.
                request = await client.account.request_special_free(
                    use_case=args.use_case, links=args.links, idempotency_key=args.idempotency_key
                )
                out.record(request, view.plan_request)
            elif args.plan_action == "requests":
                out.record(await client.account.plan_requests(), view.plan_requests)
            else:
                withdrawn = await client.account.withdraw_plan_request(args.request_id)
                out.record(withdrawn, view.plan_request)
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
    base_url = resolve_base_url(args.base_url, args.environment)
    if args.command == "login":
        auth.login(
            base_url,
            open_browser=not args.no_browser,
            redirect_source=paste_redirect if args.no_browser else None,
        )
    else:
        auth.logout(base_url)
    out.record(auth.status(base_url), view.auth_status)
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
    except RunPausedError as error:
        fail(error.code, str(error), as_json)
        return EXIT_RUN_PAUSED
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
        if error.retry_after_seconds is None:
            fail(error.problem.code, error.problem.detail, as_json, error.problem)
        elif as_json:
            # The wait arrives as a header; carry it beside the problem's own members.
            document = error.problem.model_dump(mode="json")
            document["retry_after_seconds"] = error.retry_after_seconds
            print(json.dumps(document), file=sys.stderr, flush=True)
        else:
            fail(error.problem.code, error.problem.detail, False)
            print(f"retry after {error.retry_after_seconds} seconds", file=sys.stderr, flush=True)
        return EXIT_SERVER_REFUSAL
    except DirectoryUploadError as error:
        receipt = {
            "code": error.code,
            "name": error.name,
            "idempotency_key": error.idempotency_key,
            "file_idempotency_key": error.file_idempotency_key,
            "completed_files": [file.model_dump(mode="json") for file in error.completed_files],
        }
        # A partial upload is not rolled back. Keep the committed identities and
        # caller's retry key on stderr in either output mode.
        print(json.dumps(receipt), file=sys.stderr, flush=True)
        return EXIT_CLIENT_ERROR
    except UploadTransferError as error:
        if as_json:
            print(
                json.dumps(
                    {
                        "code": error.code,
                        "detail": str(error),
                        "upload_id": error.upload_id,
                        "idempotency_key": error.idempotency_key,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
        else:
            fail(
                error.code,
                f"{error} (upload_id={error.upload_id}, idempotency_key={error.idempotency_key})",
                False,
            )
        return EXIT_CLIENT_ERROR
    except SynthError as error:
        fail(error.code, str(error), as_json)
        return EXIT_CLIENT_ERROR
    except (ValueError, OSError, UsageError) as error:
        fail("client_error", str(error), as_json)
        return EXIT_CLIENT_ERROR


if __name__ == "__main__":
    sys.exit(main())
