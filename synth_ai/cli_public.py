"""The ``synth`` command: Swarms, account and billing.

Only command groups whose product is in ``PUBLIC_PRODUCTS`` are registered.
Other groups (``cli_swarms`` preflight, ``cli_account`` legacy reads) stay
importable but are not registered.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable

from pydantic import BaseModel

from synth_ai.client import SynthClient
from synth_ai.errors import SynthError
from synth_ai.products import PUBLIC_PRODUCTS
from synth_ai.public.errors import SwarmsAPIError

ClientFactory = Callable[[], SynthClient]
Handler = Callable[[SynthClient, argparse.Namespace], BaseModel | bytes]

# login, logout and auth status belong to the account product.
AUTH_COMMANDS = frozenset({"login", "logout", "auth"})


def _swarms(group: argparse._SubParsersAction) -> None:
    swarms = group.add_parser("swarms").add_subparsers(dest="swarms_command", required=True)

    run = swarms.add_parser("run")
    run.add_argument("message")
    run.add_argument("--budget", required=True, help="maximum charge in USD")
    run.add_argument("--project-id")
    run.add_argument("--timeout", type=float)
    run.set_defaults(handler=lambda c, a: c.swarms.run(
        a.message, a.budget, project_id=a.project_id, timeout_seconds=a.timeout))

    launch = swarms.add_parser("launch")
    launch.add_argument("message")
    launch.add_argument("--budget", required=True, help="maximum charge in USD")
    launch.add_argument("--project-id")
    launch.set_defaults(handler=lambda c, a: c.swarms.launch(
        a.message, a.budget, project_id=a.project_id))

    for name in ("get", "cancel", "outputs"):
        command = swarms.add_parser(name)
        command.add_argument("run_id")
    swarms.choices["get"].set_defaults(handler=lambda c, a: c.swarms.get(a.run_id))
    swarms.choices["cancel"].set_defaults(handler=lambda c, a: c.swarms.cancel(a.run_id))
    swarms.choices["outputs"].set_defaults(handler=lambda c, a: c.swarms.outputs(a.run_id))

    wait = swarms.add_parser("wait")
    wait.add_argument("run_id")
    wait.add_argument("--timeout", type=float)
    wait.set_defaults(handler=lambda c, a: c.swarms.wait(a.run_id, timeout_seconds=a.timeout))

    events = swarms.add_parser("events")
    events.add_argument("run_id")
    events.add_argument("--after", type=int)
    events.add_argument("--limit", type=int)
    events.set_defaults(handler=lambda c, a: c.swarms.events(
        a.run_id, after=a.after, limit=a.limit))

    listing = swarms.add_parser("list")
    listing.add_argument("--cursor")
    listing.add_argument("--limit", type=int)
    listing.set_defaults(handler=lambda c, a: c.swarms.list(cursor=a.cursor, limit=a.limit))


def _account(group: argparse._SubParsersAction) -> None:
    account = group.add_parser("account")
    account.add_argument("--keys", action="store_true", help="list API keys (session only)")
    account.set_defaults(handler=lambda c, a: c.account.keys() if a.keys else c.account.get())
    for name in AUTH_COMMANDS:
        # Parsed by cli_auth; registered here so they appear in help.
        group.add_parser(name, add_help=False)


def _billing(group: argparse._SubParsersAction) -> None:
    billing = group.add_parser("billing").add_subparsers(dest="billing_command", required=True)
    billing.add_parser("balance").set_defaults(handler=lambda c, a: c.billing.balance())
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


def registered_commands() -> frozenset[str]:
    action = next(a for a in parser()._actions if isinstance(a, argparse._SubParsersAction))
    return frozenset(action.choices)


def _render(value: BaseModel | bytes) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value.model_dump_json(indent=2, exclude_none=True)


def main(argv: list[str] | None = None, *, client_factory: ClientFactory = SynthClient) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in AUTH_COMMANDS and "account" in PUBLIC_PRODUCTS:
        from synth_ai.cli_auth import main as auth_main

        return auth_main(arguments)
    args = parser().parse_args(arguments)
    try:
        with client_factory() as client:
            print(_render(args.handler(client, args)))
    except SwarmsAPIError as error:
        print(json.dumps({"error": error.error.model_dump(exclude_none=True)}), file=sys.stderr)
        return 1
    except SynthError as error:
        print(json.dumps({"error": {"code": error.code or "sdk_error", "service": "synth-ai",
                                    "stage": args.command, "cause": type(error).__name__}}),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
