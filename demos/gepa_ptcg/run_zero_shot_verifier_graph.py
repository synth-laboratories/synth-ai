#!/usr/bin/env python3
"""Run the zero-shot verifier graph over backend traces captured from a PTCG eval run.

Typical flow:
1) Run eval + trace capture:
   ./.venv/bin/python demos/gepa_ptcg/run_demo.py --local --react --num-games 10 --out-dir demos/gepa_ptcg/artifacts/runs

2) Run zero-shot verifier graph over the downloaded traces:
   ./.venv/bin/python demos/gepa_ptcg/run_zero_shot_verifier_graph.py --local --run-dir <the printed run dir>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from synth_ai.core.env import PROD_BASE_URL, mint_demo_api_key
from synth_ai.sdk.graphs.completions import GraphCompletionsSyncClient


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_session_trace(trace_payload: dict[str, Any]) -> dict[str, Any] | None:
    # Common shapes:
    # - {"trace": {...}}
    # - {"session_trace": {...}}
    # - direct trace dict with "event_history"/"schema_version"
    if not isinstance(trace_payload, dict):
        return None
    if isinstance(trace_payload.get("trace"), dict):
        return trace_payload["trace"]
    if isinstance(trace_payload.get("session_trace"), dict):
        return trace_payload["session_trace"]
    if "event_history" in trace_payload and isinstance(trace_payload.get("event_history"), list):
        return trace_payload
    return None


def _extract_trace_id(session_trace: dict[str, Any]) -> str | None:
    meta = session_trace.get("metadata")
    if isinstance(meta, dict):
        cid = meta.get("trace_correlation_id") or meta.get("correlation_id")
        return str(cid) if isinstance(cid, (str, int)) else None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-shot verifier graph over PTCG traces")
    parser.add_argument("--local", action="store_true", help="Use localhost:8000 backend")
    parser.add_argument("--backend-url", type=str, default="", help="Override backend base URL")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory produced by run_demo.py")
    parser.add_argument(
        "--shape",
        type=str,
        default="single",
        choices=["single", "mapreduce", "rlm"],
        help="Zero-shot verifier graph shape",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional verifier model override (backend-dependent). If unset, backend default is used.",
    )
    args = parser.parse_args()

    if args.backend_url:
        backend_url = args.backend_url
    elif args.local:
        backend_url = "http://127.0.0.1:8000"
    else:
        backend_url = PROD_BASE_URL

    api_key = os.getenv("SYNTH_API_KEY", "").strip()
    if not api_key:
        api_key = mint_demo_api_key(backend_url=backend_url)
        os.environ["SYNTH_API_KEY"] = api_key

    run_dir = Path(args.run_dir).expanduser().resolve()
    traces_dir = run_dir / "backend_traces"
    info_path = run_dir / "task_app_info.json"
    out_path = run_dir / "zero_shot_verifier_outputs.jsonl"

    if not traces_dir.exists():
        raise FileNotFoundError(f"Traces dir not found: {traces_dir}")
    if not info_path.exists():
        raise FileNotFoundError(
            f"Missing task_app_info.json (expected from run_demo.py): {info_path}"
        )

    info = _load_json(info_path)
    rubrics = info.get("rubrics")
    if not isinstance(rubrics, dict):
        raise ValueError("task_app_info.json missing 'rubrics' dict")

    graph_id = f"zero_shot_verifier_rubric_{args.shape}"
    client = GraphCompletionsSyncClient(backend_url, api_key)

    trace_files = sorted([p for p in traces_dir.rglob("*.json") if p.is_file()])
    if not trace_files:
        raise RuntimeError(f"No .json trace files found under {traces_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for trace_file in trace_files:
            payload = _load_json(trace_file)
            session_trace = _resolve_session_trace(payload)
            if session_trace is None:
                continue
            trace_id = _extract_trace_id(session_trace) or trace_file.stem

            input_data: dict[str, Any] = {
                "trace": session_trace,
                "rubric": rubrics,
                "options": {"event": True, "outcome": True},
            }
            if args.model:
                input_data["options"]["model"] = args.model

            output = client.run_output(job_id=graph_id, input_data=input_data)
            record = {
                "trace_file": str(trace_file.relative_to(run_dir)),
                "trace_id": trace_id,
                "graph_id": graph_id,
                "output": output,
            }
            handle.write(json.dumps(record, default=str) + "\n")
            written += 1

    print(f"Wrote {written} verifier outputs to {out_path}")


if __name__ == "__main__":
    main()

