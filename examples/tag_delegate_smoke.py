"""Smoke Synth Tag delegate, steer, and receipt flow."""

from __future__ import annotations

import argparse
import os
import time

from synth_ai import SynthClient

TERMINAL_STATUSES = {"done", "failed"}


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
    parser.add_argument("--timebox-seconds", type=int, default=120)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.backend_url:
        raise SystemExit("--backend-url or SYNTH_BACKEND_URL is required")
    client = SynthClient(
        api_key=os.environ.get("SYNTH_API_KEY"),
        base_url=args.backend_url,
    )
    session = client.research.tag.create_session(
        args.request,
        definition_of_done=args.definition_of_done,
        timebox_seconds=args.timebox_seconds,
    )
    print(f"session_id={session.session_id}")
    print(f"run_id={session.run_id}")
    print(f"status={session.status.value}")

    if args.steer and session.status.value not in TERMINAL_STATUSES:
        session = client.research.tag.send_message(session.session_id, args.steer)
        print(f"steer_run_id={session.run_id}")
        print(f"steer_status={session.status.value}")

    if args.wait:
        deadline = time.monotonic() + args.timeout_seconds
        while session.status.value not in TERMINAL_STATUSES:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Tag session {session.session_id}")
            time.sleep(args.poll_seconds)
            session = client.research.tag.get_session(session.session_id)
            print(f"status={session.status.value}")

    receipt = session.receipt
    print("receipt:")
    print(f"  run_id={receipt.run_id}")
    print(f"  state={receipt.state}")
    print(f"  run_url={receipt.run_url}")
    print(f"  artifact_urls={receipt.artifact_urls}")
    if receipt.artifact_empty_reason:
        print(f"  artifact_empty_reason={receipt.artifact_empty_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
