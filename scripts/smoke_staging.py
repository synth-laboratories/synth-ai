"""Live smoke against SYNTH_API_BASE_URL. Prints status codes and error codes only.

Default: GET /healthz and GET /api/v1/account (+ the /v1/account alias), with
SYNTH_API_KEY when set. ``--keyed`` uses SMOKE_STAGING_API_KEY (else SYNTH_API_KEY)
through the SDK: account get, billing balance, swarms list, and one launch with a
zero budget and one task, which is expected to be refused with a typed error.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from collections.abc import Callable

import httpx

from synth_ai.public.client import Account, Billing, PublicTransport, Swarms
from synth_ai.public.errors import SwarmsAPIError
from synth_ai.public.models import Limits, Money

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


def keyed() -> int:
    key = os.environ.get("SMOKE_STAGING_API_KEY") or os.environ.get("SYNTH_API_KEY")
    print(f"api_key_present={bool(key)}")
    if not key:
        return 2
    transport = PublicTransport(base_url(), api_key=key)
    swarms, account, billing = Swarms(transport), Account(transport), Billing(transport)
    zero = Limits(maximum_charge=Money.usd(0), wall_time_seconds=60, maximum_tasks=1,
                  maximum_parallel_tasks=1)
    project = os.environ.get("SYNTH_PROJECT_ID", "smoke-project")
    calls: list[tuple[str, Callable[[], object]]] = [
        ("account.get", account.get),
        ("billing.balance", billing.balance),
        ("billing.plans", billing.plans),
        ("swarms.list", lambda: swarms.list(limit=1)),
        ("swarms.launch", lambda: swarms.launch(
            "smoke", project_id=project, limits=zero, idempotency_key=str(uuid.uuid4()))),
    ]
    for name, call in calls:
        try:
            value = call()
        except SwarmsAPIError as error:
            print(f"{name} refused status={error.http_status} code={error.error.code} "
                  f"stage={error.error.stage} outcome={error.outcome}")
            continue
        detail = ""
        if name == "swarms.list":
            detail = f" items={len(value.items)} complete={value.complete}"  # type: ignore[attr-defined]
        print(f"{name} ok type={type(value).__name__}{detail}")
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
