"""Smoke Synth Tag delegate, steer, and receipt flow."""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request

from synth_ai import SynthClient

TERMINAL_STATUSES = {"done", "failed"}
SUCCESS_STATUSES = {"done"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-url", default=os.getenv("SYNTH_BACKEND_URL"))
    parser.add_argument(
        "--request",
        default="Summarize what Synth Tag v1 proves in one paragraph.",
    )
    parser.add_argument(
        "--steer",
        default="Keep the answer under 100 words and include a receipt summary.",
    )
    parser.add_argument("--definition-of-done", default=None)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--timebox-seconds", type=int, default=600)
    parser.add_argument(
        "--artifact-grace-seconds",
        type=float,
        default=120.0,
        help="After terminal status, poll receipt for artifact_urls (default: 120s).",
    )
    parser.add_argument(
        "--require-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail unless terminal status is done and artifact_urls is non-empty (default: on).",
    )
    return parser


def _artifact_url_ok(
    url: str,
    *,
    backend_url: str,
    api_key: str | None = None,
    timeout_seconds: float = 30.0,
) -> tuple[bool, str]:
    absolute = url
    if url.startswith("/"):
        absolute = f"{backend_url.rstrip('/')}{url}"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(absolute, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            code = getattr(response, "status", None) or response.getcode()
            if int(code) != 200:
                return False, f"HTTP {code}"
            return True, "ok"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, str(exc.reason or exc)


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def main() -> int:
    args = _parser().parse_args()
    if not args.backend_url:
        raise SystemExit("--backend-url or SYNTH_BACKEND_URL is required")
    client = SynthClient(
        api_key=os.environ.get("SYNTH_API_KEY"),
        base_url=args.backend_url,
    )
    tag_sessions = client.research.factories.tag.sessions
    session = tag_sessions.create(
        args.request,
        definition_of_done=args.definition_of_done,
        timebox_seconds=args.timebox_seconds,
    )
    print(f"session_id={session.session_id}")
    print(f"run_id={session.run_id}")
    print(f"status={session.status.value}")

    if args.steer and session.status.value not in TERMINAL_STATUSES:
        session = tag_sessions.messages.send(session.session_id, args.steer)
        print(f"steer_run_id={session.run_id}")
        print(f"steer_status={session.status.value}")

    if args.wait:
        deadline = time.monotonic() + args.timeout_seconds
        while session.status.value not in TERMINAL_STATUSES:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Tag session {session.session_id}")
            time.sleep(args.poll_seconds)
            session = tag_sessions.get(session.session_id)
            print(f"status={session.status.value}")

    artifact_deadline = time.monotonic() + args.artifact_grace_seconds
    while (
        args.require_artifacts
        and session.status.value in TERMINAL_STATUSES
        and not session.receipt.artifact_urls
        and time.monotonic() < artifact_deadline
    ):
        time.sleep(args.poll_seconds)
        session = tag_sessions.get(session.session_id)
        print(
            f"artifact_wait status={session.status.value} "
            f"artifact_count={len(session.receipt.artifact_urls)}"
        )

    receipt = session.receipt
    print("receipt:")
    print(f"  run_id={receipt.run_id}")
    print(f"  state={receipt.state}")
    print(f"  run_url={receipt.run_url}")
    print(f"  artifact_urls={receipt.artifact_urls}")
    if receipt.artifact_empty_reason:
        print(f"  artifact_empty_reason={receipt.artifact_empty_reason}")

    if not args.require_artifacts:
        return 0

    if session.status.value not in SUCCESS_STATUSES:
        return _fail(
            "Tag smoke failed: session status="
            f"{session.status.value!r}, expected one of {sorted(SUCCESS_STATUSES)!r}"
        )
    if receipt.state not in SUCCESS_STATUSES:
        return _fail(
            f"Tag smoke failed: receipt.state={receipt.state!r}, expected 'done'"
        )
    if not receipt.artifact_urls:
        reason = receipt.artifact_empty_reason or "unknown"
        return _fail(
            "Tag smoke failed: artifact_urls empty "
            f"(artifact_empty_reason={reason!r})"
        )

    for url in receipt.artifact_urls:
        ok, detail = _artifact_url_ok(
            url,
            backend_url=args.backend_url,
            api_key=os.environ.get("SYNTH_API_KEY"),
        )
        print(f"  artifact_check url={url} result={detail}")
        if not ok:
            return _fail(f"Tag smoke failed: artifact URL not reachable ({detail}): {url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
