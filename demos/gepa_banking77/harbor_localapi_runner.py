#!/usr/bin/env python3
"""Harbor runner for Banking77 LocalAPI.

Reads Harbor input JSON (stdin or /tmp/rollout.json), runs the LocalAPI rollout
handler directly, and prints a Harbor-compatible JSON result.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

from synth_ai.data.enums import SuccessStatus
from synth_ai.sdk.localapi._impl.contracts import RolloutRequest

import localapi_banking77


def _log(message: str, **fields: Any) -> None:
    payload = {"msg": message, **fields}
    print(f"[HARBOR_RUNNER] {json.dumps(payload, ensure_ascii=True)}", file=sys.stderr, flush=True)


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise RuntimeError(f"Failed to read JSON from {path}: {exc}") from exc


def _normalize_request(input_data: Dict[str, Any]) -> tuple[Dict[str, Any], list[str]]:
    """Extract TaskApp request and ensure inference_url is set."""
    flags: list[str] = []
    request_data = input_data.get("task_app_request")
    if request_data is None:
        flags.append("missing_task_app_request")
        request_data = input_data
    if not isinstance(request_data, dict):
        raise ValueError("task_app_request is missing or invalid")

    # If Harbor injected an interceptor URL, prefer it.
    inference_url = input_data.get("inference_url")
    if inference_url:
        policy = request_data.get("policy") or {}
        policy_config = policy.get("config") or {}
        policy_config["inference_url"] = inference_url
        policy["config"] = policy_config
        request_data["policy"] = policy
    else:
        policy = request_data.get("policy") or {}
        policy_config = policy.get("config") or {}
        if not policy_config.get("inference_url"):
            flags.append("missing_inference_url")

    # Ensure trace_correlation_id propagates
    if "trace_correlation_id" not in request_data and "trace_correlation_id" in input_data:
        request_data["trace_correlation_id"] = input_data["trace_correlation_id"]

    return request_data, flags


async def _run_rollout(request_dict: Dict[str, Any]) -> Dict[str, Any]:
    try:
        request = RolloutRequest.model_validate(request_dict)
    except AttributeError:
        request = RolloutRequest.parse_obj(request_dict)  # type: ignore[call-arg]

    response = await localapi_banking77.run_rollout(request, None)

    reward_info = response.reward_info
    reward = reward_info.outcome_reward if reward_info else 0.0
    details = reward_info.details if reward_info else {}
    error_flags = details.get("error_flags", []) if isinstance(details, dict) else []

    success = response.success_status == SuccessStatus.SUCCESS
    return {
        "trace_correlation_id": response.trace_correlation_id or request.trace_correlation_id,
        "metrics": {"reward_mean": reward, "details": details},
        "success": bool(success),
        "error": response.status_detail,
        "error_flags": error_flags,
        "inference_url": response.inference_url,
        "trace": response.trace,
        "artifact": response.artifact,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Harbor runner for Banking77 LocalAPI")
    parser.add_argument("--stdio", action="store_true", help="Read input JSON from stdin")
    parser.add_argument("--input", type=Path, default=Path("/tmp/rollout.json"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/result.json"))
    args = parser.parse_args()

    if args.stdio:
        raw = sys.stdin.read().strip()
        if not raw:
            raise RuntimeError("No input on stdin")
        input_data = json.loads(raw)
    else:
        input_data = _load_json(args.input)

    if not isinstance(input_data, dict):
        raise ValueError("Input payload must be a JSON object")

    try:
        request_dict, normalize_flags = _normalize_request(input_data)
    except Exception as exc:
        _log("input_normalize_failed", error=str(exc), exception_type=type(exc).__name__)
        result = {
            "trace_correlation_id": input_data.get("trace_correlation_id", ""),
            "metrics": {"reward_mean": 0.0, "details": {}},
            "success": False,
            "error": f"Runner normalize exception: {exc}",
            "error_flags": ["normalize_failed"],
        }
    else:
        _log("input_parsed", keys=list(input_data.keys()), flags=normalize_flags)
        try:
            result = asyncio.run(_run_rollout(request_dict))
        except Exception as exc:
            trace_id = request_dict.get("trace_correlation_id") or input_data.get("trace_correlation_id", "")
            _log("rollout_failed", error=str(exc), exception_type=type(exc).__name__)
            result = {
                "trace_correlation_id": trace_id,
                "metrics": {"reward_mean": 0.0, "details": {}},
                "success": False,
                "error": f"Runner exception: {exc}",
                "error_flags": ["runner_exception"] + normalize_flags,
            }
        else:
            result["error_flags"] = list({*(result.get("error_flags") or []), *normalize_flags})

    output_json = json.dumps(result)
    if args.stdio:
        print(output_json)
    else:
        args.output.write_text(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
