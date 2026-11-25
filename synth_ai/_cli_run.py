"""Execution helpers for run launch, log streaming, output download, and auth commands."""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from . import _cli_view as view
from . import auth
from ._cli_parser import UsageError
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
from ._observer import validate_wait
from ._protocol import CompletionMilestone
from ._validation import message_text, validate_idempotency_key
from .environments import resolve_base_url

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Sequence

    from ._auth_models import DeviceAuthorization
    from .swarms import AsyncSwarms

EXIT_OK = 0
EXIT_RUN_UNSUCCESSFUL = 3
UNSUCCESSFUL_STATES = frozenset({"failed", "cancelled"})
ASSIGN_FIELDS = {
    "inference": "inference_profile",
    "sandbox": "sandbox_profile",
    "option": "sandbox_option",
}
RESOURCES = TypeAdapter(list[OwnerResourceRequest])
M = TypeVar("M", bound=BaseModel)


class RecordPrinter(Protocol):
    as_json: bool

    def record(self, value: M, render: Callable[[M], str]) -> None: ...


def parse_optimizer(
    value: str | None, *, max_tasks: int | None = None, max_repairs_per_task: int | None = None
) -> OptimizerRequest | None:
    """Parse the public optimizer selector; Core owns resolution and availability."""
    controls = {
        name: number
        for name, number in (
            ("max_tasks", max_tasks),
            ("max_repairs_per_task", max_repairs_per_task),
        )
        if number is not None
    }
    if value is None or value == "auto":
        if controls:
            raise UsageError(
                "--max-tasks and --max-repairs-per-task need --optimizer swarms_v2@VERSION"
            )
        return None if value is None else "auto"
    topology, separator, version = value.partition("@")
    if separator == "" or not version.isascii() or not version.isdecimal():
        raise UsageError("optimizer must be auto or TOPOLOGY@VERSION")
    if controls and topology != "swarms_v2":
        raise UsageError(
            "--max-tasks and --max-repairs-per-task need --optimizer swarms_v2@VERSION"
        )
    pin: dict[str, object] = {"topology": topology, "version": int(version)}
    if controls:
        pin["params"] = controls
    try:
        return OptimizerPin.model_validate(pin)
    except ValidationError as error:
        first = error.errors(include_input=False)[0]
        where = ".".join(str(part) for part in first["loc"])
        raise UsageError("--optimizer " + value + ": " + where + ": " + first["msg"]) from error


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
        return RESOURCES.validate_json(document)
    except ValidationError as error:
        first = error.errors(include_input=False)[0]
        where = ".".join(str(part) for part in first["loc"]) or "document"
        raise UsageError(
            "--resources must be a JSON array of resource requests: " + where + ": " + first["msg"]
        ) from error


def mutation_key(action: str, run_id: str, values: dict[str, JsonValue] | None = None) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            json.dumps(
                ["synth-cli", action, run_id, values], sort_keys=True, separators=(",", ":")
            ),
        )
    )


def read_task(path: str) -> str:
    """Read a task text file, '-' (stdin), or a bundle directory holding one task file."""
    if path == "-":
        buffer = getattr(sys.stdin, "buffer", None)
        if buffer is not None:
            raw = bytes(buffer.read(MESSAGE_MAX_BYTES + 1))
        else:
            raw = sys.stdin.read(MESSAGE_MAX_BYTES + 1).encode("utf-8")
    else:
        source = Path(path)
        if source.is_dir():
            found = [p for p in source.iterdir() if not p.name.startswith(".")]
            if len(found) != 1 or found[0].name not in {"task.md", "TASK.md", "task.txt"}:
                raise ValueError("task bundle must contain exactly one task text file")
            source = found[0]
        with source.open("rb") as stream:
            raw = stream.read(MESSAGE_MAX_BYTES + 1)
    if not raw.strip() or len(raw) > MESSAGE_MAX_BYTES:
        raise ValueError(f"task must contain 1..{MESSAGE_MAX_BYTES} bytes")
    return message_text(raw.decode("utf-8"))


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


def finished_code(snapshot: RunSnapshot) -> int:
    return EXIT_RUN_UNSUCCESSFUL if snapshot.state in UNSUCCESSFUL_STATES else EXIT_OK


async def execute_run(
    swarms: AsyncSwarms,
    args: argparse.Namespace,
    out: RecordPrinter,
    *,
    stdin_piped: bool,
) -> int:
    """Validate flags, upload attached files, launch the run, and optionally wait."""
    try:
        validate_idempotency_key(args.idempotency_key)
    except ValueError as error:
        raise UsageError(str(error)) from error
    if args.wait:
        try:
            validate_wait(args.timeout, 1.0, CompletionMilestone.FINISHED)
        except ValueError as error:
            raise UsageError(str(error)) from error
    if args.goal is not None and args.task_file is not None:
        raise UsageError("provide a goal or --goal-file, not both")
    if args.task_file is not None:
        goal = read_task(args.task_file)
    elif args.goal is not None:
        goal = args.goal
    elif stdin_piped:
        goal = read_task("-")
    else:
        raise UsageError("provide a goal, --goal-file PATH, or pipe stdin")
    request = swarms.prepare_launch(
        goal,
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
        optimizer=parse_optimizer(
            args.optimizer,
            max_tasks=args.max_tasks,
            max_repairs_per_task=args.max_repairs_per_task,
        ),
        continue_from=args.continue_from,
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
    if uploaded:
        inputs = RunInputs(
            files=[
                InputFile(
                    file_id=f.file_id,
                    revision_id=f.revision_id,
                    digest_sha256=f.digest_sha256,
                )
                for f in uploaded
            ]
        )
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
    await asyncio.to_thread(target.mkdir, parents=True, exist_ok=True)
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


def show_device_code(device: DeviceAuthorization) -> None:
    print(
        f"Open {device.verification_uri} and enter {device.user_code}",
        file=sys.stderr,
        flush=True,
    )
    if device.verification_uri_complete is not None:
        print(device.verification_uri_complete, file=sys.stderr, flush=True)


def run_auth(args: argparse.Namespace, out: RecordPrinter) -> int:
    base_url = resolve_base_url(args.base_url, args.environment)
    if args.command == "login" and args.device:
        auth.login_device(base_url, show=show_device_code)
    elif args.command == "login":
        auth.login(
            base_url,
            open_browser=not args.no_browser,
            redirect_source=paste_redirect if args.no_browser else None,
        )
    else:
        auth.logout(base_url)
    out.record(auth.status(base_url), view.auth_status)
    return EXIT_OK
