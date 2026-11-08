"""Thin JSON-lines CLI over generated operations and run helpers."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from . import auth
from ._generated.operations import OPERATIONS
from ._generated.policy import MESSAGE_MAX_BYTES
from ._validation import message_text
from .aclient import AsyncSynthClient
from .errors import RunFailedError, SwarmsAPIError, SynthError, WaitTimeoutError
from .models import RoutingConstraints

COMMANDS: dict[str, dict[str, str]] = {
    "swarms": {
        "status": "getRun",
        "report": "getRunReport",
        "sandboxes": "getRunSandboxes",
        "list": "listRuns",
        "events": "listRunEvents",
        "cancel": "cancelRun",
        "pause": "pauseRun",
        "resume": "resumeRun",
        "program": "getRunProgram",
        "configuration": "getRunConfiguration",
        "tasks": "listRunTasks",
        "outputs": "listRunOutputs",
        "output": "getRunOutput",
        "output-content": "getRunOutputContent",
        "families": "listFamilies",
        "upload-text": "uploadFile",
        "uploads-prepare": "createUpload",
        "uploads-get": "getUpload",
        "uploads-finalize": "finalizeUpload",
        "files": "listFiles",
        "file": "getFile",
        "labels": "updateRunLabels",
        "download": "getRunOutputContent",
    },
    "account": {"get": "getAccount", "accept-terms": "acceptTerms"},
    "api-keys": {
        "list": "listApiKeys",
        "mint": "createApiKey",
        "update-grants": "updateApiKey",
        "rotate": "rotateApiKey",
        "revoke": "revokeApiKey",
    },
    "billing": {
        "balance": "getBalance",
        "history": "listBillingHistory",
        "statements": "listStatements",
        "plans": "listPlans",
        "checkout": "createCheckout",
        "checkout-status": "getCheckout",
    },
    "usage": {
        "summary": "getUsageSummary",
        "records": "listUsageRecords",
        "export": "createUsageExport",
        "export-status": "getUsageExport",
    },
    "limits": {"list": "listLimits"},
}


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="synth")
    root.add_argument("--base-url")
    groups = root.add_subparsers(dest="group", required=True)
    for group, commands in COMMANDS.items():
        parent = groups.add_parser(group)
        verbs = parent.add_subparsers(dest="verb", required=True)
        for verb, oid in commands.items():
            op = OPERATIONS[oid]
            command = verbs.add_parser(verb)
            command.set_defaults(operation=oid)
            for name in op["path_params"]:
                command.add_argument(name)
            for name in op["query_params"]:
                command.add_argument(
                    "--" + name.replace("_", "-"), type=int if name == "limit" else str
                )
            if op["request_model"]:
                command.add_argument("--json", default="{}", help="JSON request object")
            if op["idempotent"]:
                command.add_argument("--idempotency-key")
            if op["binary"]:
                command.add_argument("--out")
        if group == "swarms":
            upload = verbs.add_parser("upload")
            upload.add_argument("filename")
            upload.add_argument("--media-type")
            upload.add_argument("--idempotency-key")
            for verb in ["launch", "run"]:
                command = verbs.add_parser(verb)
                command.add_argument("message", nargs="?")
                command.add_argument("--task")
                command.add_argument("--budget", required=True)
                selection = command.add_mutually_exclusive_group()
                selection.add_argument("--program")
                selection.add_argument(
                    "--method", choices=["status-quo", "fork-flush", "forgetting"]
                )
                command.add_argument("--method-task", help="published task a named method runs on")
                command.add_argument("--method-overrides", help="JSON research parameter object")
                command.add_argument("--idempotency-key")
                command.add_argument("--timeout", type=float, default=600)
                command.add_argument(
                    "--until",
                    choices=["finished", "execution_closed", "result_available"],
                    default="finished",
                )
                for dimension in ("providers", "models"):
                    choices = command.add_mutually_exclusive_group()
                    choices.add_argument("--" + dimension + "-only", nargs="+")
                    choices.add_argument("--" + dimension + "-exclude", nargs="+")
            command = verbs.add_parser("wait")
            command.add_argument("run_id")
            command.add_argument("--timeout", type=float, default=600)
            command.add_argument(
                "--until",
                choices=["finished", "execution_closed", "result_available"],
                default="finished",
            )
            for verb in ["follow", "attach", "watch"]:
                command = verbs.add_parser(verb)
                command.add_argument("run_id")
                command.add_argument("--last-event-id")
                if verb == "watch":
                    command.add_argument("--timeout", type=float, default=600)
                    command.add_argument(
                        "--until",
                        choices=["finished", "execution_closed", "result_available"],
                        default="finished",
                    )
    groups.add_parser("login")
    groups.add_parser("logout")
    auth_group = groups.add_parser("auth")
    auth_group.add_subparsers(dest="verb", required=True).add_parser("status")
    return root


def emit(value: Any) -> None:
    if isinstance(value, BaseModel):
        print(value.model_dump_json(exclude_unset=True), flush=True)
    elif isinstance(value, bytes):
        print(
            json.dumps({"encoding": "base64", "content": base64.b64encode(value).decode()}),
            flush=True,
        )
    else:
        print(json.dumps(value), flush=True)


def read_task(path: str) -> str:
    source = Path(path)
    if source.is_dir():
        files = [p for p in source.iterdir() if not p.name.startswith(".")]
        if len(files) != 1 or files[0].name not in {"task.md", "TASK.md", "task.txt"}:
            raise ValueError("task bundle must contain exactly one task text file")
        source = files[0]
    raw = source.read_bytes()
    if not raw.strip() or len(raw) > MESSAGE_MAX_BYTES:
        raise ValueError(f"task must contain 1..{MESSAGE_MAX_BYTES} bytes")
    return message_text(raw.decode("utf-8"))


async def execute(args: argparse.Namespace) -> None:
    async with AsyncSynthClient(base_url=args.base_url) as client:

        async def wait_and_emit(run_id: str) -> None:
            try:
                emit(
                    await client.swarms.wait(
                        run_id, timeout=args.timeout, until=args.until, raise_on_failure=True
                    )
                )
            except RunFailedError as error:
                if error.snapshot is not None:
                    emit(error.snapshot)
                raise

        if hasattr(args, "operation"):
            op = OPERATIONS[args.operation]
            body = json.loads(args.json) if hasattr(args, "json") else None
            if getattr(args, "name", None):
                body = {"name": args.name}
            if args.operation == "getRunOutputContent":
                if getattr(args, "out", None):
                    saved = await client.swarms.download_output(
                        args.run_id, args.output_id, args.out,
                        revision_id=args.revision_id, digest_sha256=args.digest_sha256,
                    )
                    emit({"saved": str(saved)})
                else:
                    emit(await client.swarms.output_content(
                        args.run_id, args.output_id,
                        revision_id=args.revision_id, digest_sha256=args.digest_sha256,
                    ))
                return
            result = await client._transport.invoke(
                args.operation,
                path={name: getattr(args, name) for name in op["path_params"]},
                query={name: getattr(args, name) for name in op["query_params"]},
                body=body,
                idempotency_key=getattr(args, "idempotency_key", None),
            )
            if isinstance(result, bytes) and getattr(args, "out", None):
                with Path(args.out).open("xb") as output:
                    output.write(result)
                emit({"saved": args.out})
            else:
                emit(result)
        elif args.verb == "upload":
            emit(
                await client.swarms.upload_file(
                    args.filename,
                    media_type=args.media_type,
                    idempotency_key=args.idempotency_key,
                )
            )
        elif args.verb in {"run", "launch"}:
            message = read_task(args.task) if args.task else args.message
            program = None
            if args.program:
                name, separator, version = args.program.rpartition("@")
                if not separator or not name or not version:
                    raise ValueError("program must be FAMILY@VERSION")
                program = {"id": name, "version": version}
            method: dict[str, Any] | None = None
            if args.method in {"fork-flush", "forgetting"}:
                release = {
                    "fork-flush": "research.fork_flush@1",
                    "forgetting": "research.forgetting@1",
                }[args.method]
                if not args.method_task:
                    raise ValueError(f"--method {args.method} requires --method-task")
                method = {"release": release, "task": args.method_task}
                if args.method_overrides:
                    overrides = json.loads(args.method_overrides)
                    if not isinstance(overrides, dict):
                        raise ValueError("method overrides must be a JSON object")
                    method["overrides"] = overrides
            elif args.method_overrides or args.method_task:
                raise ValueError(
                    "--method-task and --method-overrides require fork-flush or forgetting"
                )
            routing: dict[str, Any] = {}
            for dimension in ("providers", "models"):
                for mode in ("only", "exclude"):
                    value = getattr(args, dimension + "_" + mode)
                    if value is not None:
                        routing[dimension] = {mode: value}
            created = await client.swarms.launch(
                message,
                routing=RoutingConstraints.model_validate(routing) if routing else None,
                budget=args.budget,
                program=program,
                method=method,
                idempotency_key=args.idempotency_key,
            )
            emit(created)
            if args.verb == "run":
                await wait_and_emit(created.run_id)
        elif args.verb == "wait":
            await wait_and_emit(args.run_id)
        elif args.verb == "watch":
            if args.timeout <= 0:
                raise ValueError("timeout must be positive")
            try:
                async with asyncio.timeout(args.timeout):
                    async for event in client.swarms.follow(
                        args.run_id, last_event_id=args.last_event_id
                    ):
                        emit(event)
                    await wait_and_emit(args.run_id)
            except TimeoutError:
                raise WaitTimeoutError("Timed out watching " + args.run_id) from None
        else:
            async for event in client.swarms.follow(args.run_id, last_event_id=args.last_event_id):
                emit(event)


def main() -> None:
    args = parser().parse_args()
    try:
        if args.group == "login":
            auth.login(args.base_url)
            emit({"authenticated": True})
        elif args.group == "logout":
            auth.logout(args.base_url)
            emit({"authenticated": False})
        elif args.group == "auth":
            emit(auth.status(args.base_url))
        else:
            asyncio.run(execute(args))
    except SwarmsAPIError as error:
        print(error.problem.model_dump_json(), file=sys.stderr)
        raise SystemExit(1) from None
    except (SynthError, ValueError, OSError) as error:
        print(json.dumps({"code": "client_error", "detail": str(error)}), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
