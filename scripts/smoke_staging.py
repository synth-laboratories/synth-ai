"""Live smoke against SYNTH_API_BASE_URL. Prints status codes and error codes only.

Default: GET /healthz and GET /api/v1/account (+ the /v1/account alias), with
SYNTH_API_KEY when set. ``--keyed`` uses SMOKE_STAGING_API_KEY (else SYNTH_API_KEY)
through the SDK: account, billing, families, project create, one capped launch, and
run read/events/outputs/cancel (see ``keyed``).
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from collections.abc import Callable
from typing import TypeVar

import httpx
from synth_ai.errors import SynthError
from synth_ai.public.client import Account, Billing, PublicTransport, Swarms
from synth_ai.public.errors import SwarmsAPIError
from synth_ai.public.models import Limits, Money

T = TypeVar("T")
PATHS = ("/healthz", "/api/v1/account", "/v1/account")


def base_url() -> str:
    return os.environ.get("SYNTH_API_BASE_URL", "https://api.usesynth.ai").rstrip("/")


def unkeyed() -> None:
    key = os.environ.get("SYNTH_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    print(f"api_key_present={bool(key)}")
    with httpx.Client(timeout=20.0) as http:
        for path in PATHS:
            try:
                status = http.get(f"{base_url()}{path}", headers=headers).status_code
            except httpx.HTTPError as error:
                print(f"GET {path} error={type(error).__name__}")
                continue
            print(f"GET {path} {status}")


def _call(name: str, call: Callable[[], T]) -> T | None:
    """Run one SDK call; print its status/error code only (never values)."""
    try:
        value = call()
    except SwarmsAPIError as error:
        print(f"{name} refused status={error.http_status} code={error.error.code} "
              f"stage={error.error.stage} outcome={error.outcome}")
        return None
    except SynthError as error:
        print(f"{name} failed code={error.code} cause={type(error).__name__}")
        return None
    print(f"{name} ok type={type(value).__name__}{_detail(value)}")
    return value


def _detail(value: object) -> str:
    items = getattr(value, "items", None)
    if isinstance(items, list):
        return f" items={len(items)} complete={getattr(value, 'complete', None)}"
    families = getattr(value, "families", None)
    if isinstance(families, list):
        return f" families={[f'{f.id}@{f.version}' for f in families]}"
    state = getattr(value, "state", None)
    return f" state={state}" if state is not None else ""


def keyed() -> int:
    """account, balance, families, project create, launch, then read/events/outputs/cancel.

    Until the staging host flip, launch is refused (502 orchestra_start_refused); the
    reads and cancel then run against the newest existing run, whose cancel answers
    409 run_already_closed when it is closed."""
    key = os.environ.get("SMOKE_STAGING_API_KEY") or os.environ.get("SYNTH_API_KEY")
    print(f"api_key_present={bool(key)} base_url={base_url()}")
    if not key:
        return 2
    transport = PublicTransport(base_url(), api_key=key)
    swarms, account, billing = Swarms(transport), Account(transport), Billing(transport)
    _call("account.get", account.get)
    _call("billing.balance", billing.balance)
    _call("billing.history", lambda: billing.history(limit=5))
    _call("billing.plans", billing.plans)
    families = _call("swarms.families", swarms.families)
    general = next((f for f in families.families if f.id == "swarms.general"), None) \
        if families else None
    program = {"id": "swarms.general", "version": general.version if general else "1"}
    project = _call("swarms.create_project", lambda: swarms.create_project(
        "sdk-smoke-p6-2", objective="SDK keyed smoke", idempotency_key="sdk-smoke-p6-2-project"))
    if project is not None:
        _call("swarms.project", lambda: swarms.project(project.project_id))
    capped = Limits(maximum_charge=Money.usd("0.01"), wall_time_seconds=60, maximum_tasks=1,
                    maximum_parallel_tasks=1)
    project_id = project.project_id if project else os.environ.get("SYNTH_PROJECT_ID", "")
    launched = _call("swarms.launch", lambda: swarms.launch(
        "Reply with the single word ok.", project_id=project_id, limits=capped,
        program=program, idempotency_key=str(uuid.uuid4())))
    run_id = launched.run_id if launched else None
    if run_id is None:
        page = _call("swarms.list", lambda: swarms.list(limit=1))
        run_id = page.items[0].run_id if page and page.items else None
        print(f"read target={'newest_existing_run' if run_id else 'none'}")
    else:
        print("read target=launched_run")
    if run_id is None:
        transport.close()
        return 1
    _call("swarms.get", lambda: swarms.get(run_id))
    _call("swarms.events", lambda: swarms.events(run_id, limit=5))
    _call("swarms.outputs", lambda: swarms.outputs(run_id))
    _call("swarms.cancel", lambda: swarms.cancel(run_id, idempotency_key=str(uuid.uuid4())))
    transport.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyed", action="store_true")
    args = parser.parse_args()
    if args.keyed:
        return keyed()
    unkeyed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
