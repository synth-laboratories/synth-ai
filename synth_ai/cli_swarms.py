"""Swarms launch preview: qualification plus remaining rate capacity, no run started.

Only preflight is exposed here. It reserves no budget and starts no work, so the
companion CLI stays free of workload and purchasing commands; runs are created
through the SDK or MCP.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from synth_ai.client import SynthClient
from synth_ai.options import ReadOptions
from synth_ai.sdk.swarms.types.launch import LaunchRequest
from synth_ai.sdk.swarms.types.runs import LaunchPreflight
from synth_ai.types.rate_limits import describe_rate_limit

MAXIMUM_REQUEST_BYTES = 256 * 1024


def read_launch_request(path: str) -> LaunchRequest:
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise ValueError(f"Launch request file is unreadable: {error.strerror}") from error
    if len(raw) > MAXIMUM_REQUEST_BYTES:
        raise ValueError(f"Launch request file exceeds {MAXIMUM_REQUEST_BYTES} bytes")
    return LaunchRequest.model_validate_json(raw)


def preflight(client: SynthClient, request: LaunchRequest) -> LaunchPreflight:
    return client.swarms.preflight(
        request.legacy_objective,
        message=request.message,
        project_id=request.project_id,
        limits=request.limits,
        inputs=request.inputs,
        providers=request.providers,
        models=request.models,
        labels=request.labels,
        repo=request.repo,
        guidance=request.guidance,
        resources=request.resources,
        method=request.method,
        request=ReadOptions(read_retry_count=0),
    )


def render_preflight(result: LaunchPreflight, *, as_json: bool) -> list[str]:
    if as_json:
        body = result.model_dump(mode="json")
        body["rate_limits"] = [dataclasses.asdict(item) for item in result.rate_limits]
        return [json.dumps(body, sort_keys=True)]
    lines = [f"ready: {'yes' if result.ready else 'no'}"]
    lines += [f"blocker: {json.dumps(blocker, ensure_ascii=True)}" for blocker in result.blockers]
    if result.expires_at is not None:
        lines.append(f"expires: {result.expires_at.isoformat()}")
    if result.rate_limits:
        lines += [describe_rate_limit(item) for item in result.rate_limits]
    else:
        lines.append("rate limit: not reported")
    return lines


def run_swarms(args: argparse.Namespace) -> int:
    request = read_launch_request(args.request)
    with SynthClient() as client:
        result = preflight(client, request)
    for line in render_preflight(result, as_json=args.json):
        print(line)
    return 0


def add_swarms_commands(commands) -> None:
    swarms = commands.add_parser("swarms").add_subparsers(dest="swarms_command", required=True)
    command = swarms.add_parser("preflight")
    command.add_argument(
        "--request", required=True, help="Launch request JSON (message, project_id, limits, ...)"
    )
    command.add_argument("--json", action="store_true")
