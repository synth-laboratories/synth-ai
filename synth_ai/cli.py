"""The `synth` command line: task verbs over the typed client."""

from __future__ import annotations

import asyncio
import io
import json
import sys
from typing import TYPE_CHECKING, TypeVar

import httpx
from pydantic import BaseModel, TypeAdapter

from . import _cli_view as view
from ._cli_parser import (
    COMMANDS as COMMANDS,
)
from ._cli_parser import (
    DEFAULT_WAIT_SECONDS as DEFAULT_WAIT_SECONDS,
)
from ._cli_parser import (
    EPILOG as EPILOG,
)
from ._cli_parser import (
    ROOT_VALUE_OPTIONS as ROOT_VALUE_OPTIONS,
)
from ._cli_parser import (
    Parser as Parser,
)
from ._cli_parser import (
    UsageError as UsageError,
)
from ._cli_parser import (
    index_requested as index_requested,
)
from ._cli_parser import (
    parser as parser,
)
from ._cli_run import (
    EXIT_OK as EXIT_OK,
)
from ._cli_run import (
    EXIT_RUN_UNSUCCESSFUL as EXIT_RUN_UNSUCCESSFUL,
)
from ._cli_run import (
    UNSUCCESSFUL_STATES as UNSUCCESSFUL_STATES,
)
from ._cli_run import (
    download as download,
)
from ._cli_run import (
    execute_run,
)
from ._cli_run import (
    finished_code as finished_code,
)
from ._cli_run import (
    follow as follow,
)
from ._cli_run import (
    mutation_key as mutation_key,
)
from ._cli_run import (
    parse_guidance as parse_guidance,
)
from ._cli_run import (
    parse_labels as parse_labels,
)
from ._cli_run import (
    parse_optimizer as parse_optimizer,
)
from ._cli_run import (
    parse_resources as parse_resources,
)
from ._cli_run import (
    parse_role_assignments as parse_role_assignments,
)
from ._cli_run import (
    paste_redirect as paste_redirect,
)
from ._cli_run import (
    read_task as read_task,
)
from ._cli_run import (
    run_auth as _run_auth_impl,
)
from ._cli_run import (
    show_device_code as show_device_code,
)
from ._cli_surfaces import execute_resource
from ._generated.models import MessageKind
from ._generated.operations import CONTRACT_MAJOR, CONTRACT_VERSION
from ._version import __version__
from .aclient import AsyncSynthClient
from .errors import (
    DirectoryUploadError,
    LaunchValidationError,
    ResponseContractError,
    RunPausedError,
    SwarmsAPIError,
    SynthError,
    TransportFailureError,
    UpgradeRequiredError,
    UploadTransferError,
    WaitTimeoutError,
)

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Sequence

    from ._retry import RetryPolicy

EXIT_CLIENT_ERROR = 1
EXIT_WAIT_TIMEOUT = 2
EXIT_SERVER_REFUSAL = 4
EXIT_RUN_PAUSED = 5
INDEX_NOT_AVAILABLE = (
    "Index is not available on the Swarms API; launch a run with files and a message."
)
_RESOURCE_COMMANDS = frozenset(
    {
        "files",
        "uploads",
        "keys",
        "organization",
        "inspect",
        "content",
        "labels",
        "artifacts",
        "experiments",
        "research-log",
    }
)
M = TypeVar("M", bound=BaseModel)

__all__ = [
    "COMMANDS",
    "DEFAULT_WAIT_SECONDS",
    "EPILOG",
    "EXIT_CLIENT_ERROR",
    "EXIT_OK",
    "EXIT_RUN_PAUSED",
    "EXIT_RUN_UNSUCCESSFUL",
    "EXIT_SERVER_REFUSAL",
    "EXIT_WAIT_TIMEOUT",
    "INDEX_NOT_AVAILABLE",
    "ROOT_VALUE_OPTIONS",
    "UNSUCCESSFUL_STATES",
    "Output",
    "Parser",
    "UsageError",
    "download",
    "execute",
    "fail",
    "finished_code",
    "follow",
    "index_requested",
    "main",
    "mutation_key",
    "parse_guidance",
    "parse_labels",
    "parse_optimizer",
    "parse_resources",
    "parse_role_assignments",
    "parser",
    "paste_redirect",
    "read_task",
    "run_auth",
    "show_device_code",
    "upgrade_hint",
]


def _stdin_has_pipe() -> bool:
    return isinstance(sys.stdin, io.TextIOBase) and not sys.stdin.isatty()


class Output:
    """Format SDK models as single-line JSON or human tables on stdout."""

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


def run_auth(args: argparse.Namespace) -> int:
    """Execute login/logout auth command."""
    return _run_auth_impl(args, Output(args.json))


async def execute(
    args: argparse.Namespace,
    http_client: httpx.AsyncClient | None,
    retry_policy: RetryPolicy | None = None,
) -> int:
    """Dispatch one parsed CLI invocation against AsyncSynthClient."""
    out = Output(args.json)
    async with AsyncSynthClient(
        base_url=args.base_url,
        environment=args.environment,
        http_client=http_client,
        retry_policy=retry_policy,
    ) as client:
        swarms = client.swarms
        command = str(args.command)
        if (
            command in _RESOURCE_COMMANDS
            or (command == "billing" and args.resource_action is not None)
            or (command == "runs" and args.runs_action != "list")
        ):
            value = await execute_resource(client, args)
            rendered = (
                view.render_billing(args.resource_action, value)
                if command == "billing" and not args.json
                else None
            )
            if rendered is not None:
                print(rendered, flush=True)
            else:
                print(
                    "null" if value is None else value.model_dump_json(exclude_unset=True),
                    flush=True,
                )
        elif command == "families":
            out.record(await swarms.families(), view.families)
        elif command == "promo":
            out.record(
                await client.account.redeem_promo(args.code, idempotency_key=args.idempotency_key),
                view.promo_redemption,
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
            return await execute_run(swarms, args, out, stdin_piped=_stdin_has_pipe())
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
                out.record(
                    await swarms.events(args.run_id, cursor=args.cursor, limit=args.limit),
                    view.events,
                )
        elif command == "tasks":
            out.record(
                await swarms.tasks(args.run_id, cursor=args.cursor, limit=args.limit), view.tasks
            )
        elif command == "message":
            out.record(
                await swarms.send_message(
                    args.run_id,
                    args.text,
                    kind=MessageKind(args.kind),
                    idempotency_key=args.idempotency_key or mutation_key("message", args.run_id),
                ),
                view.message,
            )
        elif command == "messages":
            if args.message_id is not None:
                out.record(await swarms.message(args.run_id, args.message_id), view.message)
            elif args.follow:
                async for item in swarms.follow_messages(args.run_id):
                    out.record(item, view.message)
            else:
                out.record(
                    await swarms.messages(args.run_id, cursor=args.cursor, limit=args.limit),
                    view.messages,
                )
        elif command in {"ls", "runs"}:
            out.record(
                await swarms.list(limit=args.limit, cursor=args.cursor, state=args.state), view.runs
            )
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
            out.record(
                await swarms.outputs(args.run_id, cursor=args.cursor, limit=args.limit),
                view.outputs,
            )
        elif command == "download":
            out.records(await download(swarms, args.run_id, args.output_id, args.out), view.saved)
        elif command == "report":
            out.record(await swarms.report(args.run_id), view.report)
        elif command == "allocation":
            out.record(await swarms.allocation(args.run_id), view.allocation)
        elif command == "result":
            finished = await swarms.result(args.run_id)
            print(view.result_json(finished) if args.json else view.result(finished), flush=True)
            if not args.json:
                for line in (
                    view.status_line(finished),
                    view.accounting_line(finished),
                    view.charge_line(finished.charge),
                    view.result_file_line(finished),
                ):
                    if line is not None:
                        print(line, file=sys.stderr, flush=True)
        elif command == "usage":
            if args.run_id is None:
                out.record(
                    await client.account.usage_summary(since=args.since, until=args.until),
                    view.usage_summary,
                )
            else:
                out.record(
                    await client.account.usage_records(
                        since=args.since,
                        until=args.until,
                        run_id=args.run_id,
                        cursor=args.cursor,
                        limit=args.limit,
                    ),
                    view.usage_records,
                )
        elif command == "billing":
            out.record(await client.billing.balance(), view.balance)
        elif command == "limits":
            if args.run_id == "set":
                if args.set_run_id is None or args.expected_revision is None:
                    raise UsageError("limits set requires RUN and --expected-revision")
                values = {
                    "expected_revision": args.expected_revision,
                    "budget": args.budget,
                    "wall_time_seconds": args.wall,
                    "maximum_tokens": args.tokens,
                    "pause_retention_seconds": args.pause_retention,
                    "resume": args.resume,
                    "reason": args.reason,
                }
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
                req = await client.account.request_special_free(
                    use_case=args.use_case, links=args.links, idempotency_key=args.idempotency_key
                )
                out.record(req, view.plan_request)
            elif args.plan_action == "requests":
                out.record(await client.account.plan_requests(), view.plan_requests)
            else:
                out.record(
                    await client.account.withdraw_plan_request(args.request_id), view.plan_request
                )
        elif command == "grants":
            out.record(
                await client.account.revoke_oauth_grant(args.grant_id), view.oauth_grant_revoked
            )
        elif command == "version":
            if args.server:
                out.record(await client.version(), view.server_version)
            elif args.json:
                print(
                    json.dumps(
                        {
                            "client_version": __version__,
                            "contract_version": CONTRACT_VERSION,
                            "contract_major": CONTRACT_MAJOR,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
            else:
                print(
                    view.facts(
                        [
                            ("client", __version__),
                            ("contract", f"{CONTRACT_VERSION} (major {CONTRACT_MAJOR})"),
                        ]
                    ),
                    flush=True,
                )
        else:
            raise UsageError("unknown command " + command)
    return EXIT_OK


def fail(code: str, detail: str, as_json: bool, problem: BaseModel | None = None) -> None:
    detail = " ".join(detail.split())  # A refusal detail is one line (SYN-5666).
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


def main(
    argv: Sequence[str] | None = None,
    *,
    http_client: httpx.AsyncClient | None = None,
    retry_policy: RetryPolicy | None = None,
) -> int:
    """Run one CLI command."""
    raw = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in raw
    try:
        args = parser(stdin_piped=_stdin_has_pipe()).parse_args(raw)
    except SystemExit as exit_exc:
        if exit_exc.code is None:
            return EXIT_OK
        if isinstance(exit_exc.code, int):
            return exit_exc.code
        return EXIT_CLIENT_ERROR
    except UsageError as error:
        if index_requested(raw):
            fail("index_not_available", INDEX_NOT_AVAILABLE, as_json)
        else:
            fail("usage_error", str(error), as_json)
        return EXIT_CLIENT_ERROR
    try:
        if args.command in {"login", "logout"}:
            return run_auth(args)
        return asyncio.run(execute(args, http_client, retry_policy))
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
            document = error.problem.model_dump(mode="json")
            document["retry_after_seconds"] = error.retry_after_seconds
            print(json.dumps(document), file=sys.stderr, flush=True)
        else:
            fail(error.problem.code, error.problem.detail, False)
            print(f"retry after {error.retry_after_seconds} seconds", file=sys.stderr, flush=True)
        return (
            EXIT_CLIENT_ERROR if isinstance(error, LaunchValidationError) else EXIT_SERVER_REFUSAL
        )
    except DirectoryUploadError as error:
        receipt = {
            "code": error.code,
            "name": error.name,
            "idempotency_key": error.idempotency_key,
            "file_idempotency_key": error.file_idempotency_key,
            "completed_files": [file.model_dump(mode="json") for file in error.completed_files],
        }
        print(json.dumps(receipt), file=sys.stderr, flush=True)
        return EXIT_CLIENT_ERROR
    except UploadTransferError as error:
        if as_json:
            payload = {
                "code": error.code,
                "detail": str(error),
                "upload_id": error.upload_id,
                "idempotency_key": error.idempotency_key,
            }
            print(json.dumps(payload), file=sys.stderr, flush=True)
        else:
            detail = (
                f"{error} (upload_id={error.upload_id}, idempotency_key={error.idempotency_key})"
            )
            fail(error.code, detail, False)
        return EXIT_CLIENT_ERROR
    except SynthError as error:
        fail(error.code, str(error), as_json)
        return EXIT_CLIENT_ERROR
    except httpx.HTTPError as error:
        fail("network_error", f"HTTP transport error ({type(error).__name__})", as_json)
        return EXIT_CLIENT_ERROR
    except (ValueError, OSError, UsageError) as error:
        fail("client_error", str(error), as_json)
        return EXIT_CLIENT_ERROR


if __name__ == "__main__":
    sys.exit(main())
