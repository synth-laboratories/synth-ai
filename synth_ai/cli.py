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
from .aclient import AsyncSynthClient
from .errors import RunFailedError, SwarmsAPIError, SynthError, WaitTimeoutError

COMMANDS: dict[str, dict[str, str]] = {
    "swarms": {
        "status": "getRun",
        "list": "listRuns",
        "events": "listRunEvents",
        "cancel": "cancelRun",
        "outputs": "listRunOutputs",
        "output": "getRunOutput",
        "output-content": "getRunOutputContent",
        "create-project": "createProject",
        "project": "getProject",
        "projects": "listProjects",
        "families": "listFamilies",
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
    "usage": {"summary": "getUsageSummary", "records": "listUsageRecords"},
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
            if oid == "createProject":
                command.add_argument("--name")
            if op["binary"]:
                command.add_argument("--out")
        if group == "swarms":
            for verb in ["launch", "run"]:
                command = verbs.add_parser(verb)
                command.add_argument("message", nargs="?")
                command.add_argument("--task")
                command.add_argument("--project-id")
                command.add_argument("--budget", required=True)
                command.add_argument("--program")
                command.add_argument("--idempotency-key")
                command.add_argument("--timeout", type=float, default=600)
            command = verbs.add_parser("wait")
            command.add_argument("run_id")
            command.add_argument("--timeout", type=float, default=600)
            for verb in ["follow", "attach", "watch"]:
                command = verbs.add_parser(verb)
                command.add_argument("run_id")
                command.add_argument("--last-event-id")
                if verb == "watch":
                    command.add_argument("--timeout", type=float, default=600)
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
    if not raw.strip() or len(raw) > 8192:
        raise ValueError("task must contain 1..8192 bytes")
    return raw.decode("utf-8")


async def execute(args: argparse.Namespace) -> None:
    async with AsyncSynthClient(base_url=args.base_url) as client:
        if hasattr(args, "operation"):
            op = OPERATIONS[args.operation]
            body = json.loads(args.json) if hasattr(args, "json") else None
            if getattr(args, "name", None):
                body = {"name": args.name}
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
        elif args.verb in {"run", "launch"}:
            message = read_task(args.task) if args.task else args.message
            program = None
            if args.program:
                name, separator, version = args.program.rpartition("@")
                if not separator or not name or not version:
                    raise ValueError("program must be FAMILY@VERSION")
                program = {"id": name, "version": version}
            created = await client.swarms.launch(
                message,
                project_id=args.project_id,
                budget=args.budget,
                program=program,
                idempotency_key=args.idempotency_key,
            )
            emit(created)
            if args.verb == "run":
                emit(await client.swarms.wait(created.run_id, timeout=args.timeout))
        elif args.verb == "wait":
            emit(await client.swarms.wait(args.run_id, timeout=args.timeout))
        elif args.verb == "watch":
            if args.timeout <= 0:
                raise ValueError("timeout must be positive")
            try:
                async with asyncio.timeout(args.timeout):
                    async for event in client.swarms.follow(
                        args.run_id, last_event_id=args.last_event_id
                    ):
                        emit(event)
                    try:
                        emit(await client.swarms.wait(args.run_id, timeout=args.timeout))
                    except RunFailedError:
                        emit(await client.swarms.get(args.run_id))
                        raise
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
