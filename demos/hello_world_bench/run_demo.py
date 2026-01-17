#!/usr/bin/env python3
"""
hello_world_bench

Contrived OpenCode task:
- Edit output.txt to contain exactly: "Hello, world!"
- Then stop.

NOTE: We use output.txt (not README.md) because OpenCode's `write` tool description
explicitly says "NEVER proactively create documentation files (*.md) or README files".
The model follows this instruction literally and loops forever on markdown files.

This routes OpenCode through the Synth interceptor to test whether edit/write works at all.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from synth_ai.core.env import PROD_BASE_URL, mint_demo_api_key


def normalize_interceptor_base(inference_url: str) -> tuple[str, str | None]:
    """Normalize interceptor base URL and extract correlation ID if present."""
    parsed = urlparse(inference_url)
    base_path = parsed.path or ""
    for suffix in ["/v1/chat/completions", "/chat/completions", "/responses", "/v1/responses"]:
        if base_path.endswith(suffix):
            base_path = base_path[: -len(suffix)]
            break
    base = f"{parsed.scheme}://{parsed.netloc}{base_path}".rstrip("/")
    cid_values = parse_qs(parsed.query).get("cid", [])
    correlation_id = cid_values[0] if cid_values else None
    return base, correlation_id


async def run_opencode_agent(
    prompt: str,
    sandbox_dir: Path,
    *,
    model: str,
    timeout: int,
    inference_url: str,
    api_key: str,
) -> dict[str, Any]:
    opencode_bin = os.environ.get("OPENCODE_BIN") or str(
        Path.home() / ".opencode" / "bin" / "opencode"
    )
    if not Path(opencode_bin).exists():
        raise RuntimeError(f"opencode binary not found: {opencode_bin}")

    base_url, correlation_id = normalize_interceptor_base(inference_url)
    if correlation_id:
        base_url = f"{base_url}/{correlation_id}"

    model_id = model.split("/", 1)[1] if "/" in model else model
    model_with_provider = f"openai/{model_id}"

    opencode_config = {
        "$schema": "https://opencode.ai/config.json",
        "model": model_with_provider,
        "provider": {
            "openai": {
                "options": {
                    # OpenCode requires some API key string to be present client-side.
                    "apiKey": api_key,
                    # Critical: route via interceptor (OpenCode will call /responses off this base).
                    "baseURL": base_url,
                }
            }
        },
        "permission": {
            "*": "allow",
            "external_directory": "allow",
            "bash": "allow",
            "read": "allow",
            "write": "allow",
            "edit": "allow",
            "list": "allow",
            "glob": "allow",
            "grep": "allow",
        },
    }

    (sandbox_dir / "opencode.json").write_text(json.dumps(opencode_config, indent=2))

    cmd = [
        opencode_bin,
        "run",
        "--format",
        "json",
        "--model",
        model_with_provider,
        prompt,
    ]
    if os.environ.get("OPENCODE_DEBUG") == "1":
        cmd.extend(["--print-logs", "--log-level", "DEBUG"])

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key

    print(f"[hello_world_bench] Running OpenCode: cwd={sandbox_dir}")
    print(f"[hello_world_bench] Interceptor baseURL: {base_url}")
    print(f"[hello_world_bench] Command: {cmd[:3]} ...")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(sandbox_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []

    async def read_stream(stream: asyncio.StreamReader, chunks: list[bytes], prefix: str) -> None:
        while True:
            chunk = await stream.read(1024)
            if not chunk:
                break
            chunks.append(chunk)
            text = chunk.decode("utf-8", errors="replace")
            for line in text.splitlines():
                if line.strip():
                    print(f"[OpenCode] {prefix}: {line}", flush=True)

    stdout_task = asyncio.create_task(read_stream(proc.stdout, stdout_chunks, "STDOUT"))
    stderr_task = asyncio.create_task(read_stream(proc.stderr, stderr_chunks, "STDERR"))

    try:
        await asyncio.wait_for(proc.wait(), timeout=float(timeout))
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    finally:
        for t in (stdout_task, stderr_task):
            t.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)

    return {
        "returncode": proc.returncode,
        "stdout": b"".join(stdout_chunks).decode("utf-8", errors="replace"),
        "stderr": b"".join(stderr_chunks).decode("utf-8", errors="replace"),
    }


def write_sandbox_files(sandbox_dir: Path) -> None:
    sandbox_dir.mkdir(parents=True, exist_ok=True)

    # Prevent OpenCode from searching for AGENTS.md forever.
    (sandbox_dir / "AGENTS.md").write_text(
        "# Agent Instructions\n\nEdit output.txt per the task prompt, then stop.\n",
        encoding="utf-8",
    )

    # Use output.txt (NOT README.md) because OpenCode's `write` tool description
    # explicitly says "NEVER proactively create documentation files (*.md) or README files".
    (sandbox_dir / "output.txt").write_text(
        "TODO: Replace this entire file with exactly one line: Hello, world!\n",
        encoding="utf-8",
    )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="hello_world_bench: OpenCode write/edit sanity check"
    )
    parser.add_argument("--local", action="store_true", help="Use localhost:8000 backend")
    parser.add_argument("--model", type=str, default="gpt-5-nano")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    backend_base = "http://localhost:8000" if args.local else PROD_BASE_URL
    print(f"[hello_world_bench] Backend: {backend_base}")

    # Ensure backend is alive.
    r = httpx.get(f"{backend_base}/health", timeout=10)
    print(f"[hello_world_bench] Backend health: {r.status_code}")

    # Mint demo key if needed (client-side placeholder; backend uses stored env keys).
    api_key = os.environ.get("SYNTH_API_KEY", "")
    if not api_key:
        print("[hello_world_bench] No SYNTH_API_KEY, minting demo key...")
        api_key = mint_demo_api_key(backend_url=backend_base)
        os.environ["SYNTH_API_KEY"] = api_key
    print(f"[hello_world_bench] Demo API key present: {api_key[:16]}...")

    trial_id = f"hello-world-{uuid.uuid4().hex[:12]}"
    correlation_id = f"hello_world_{uuid.uuid4().hex[:12]}"

    # Register passthrough trial in interceptor registry.
    register_url = f"{backend_base}/api/interceptor/v1/debug/register_trial/{trial_id}"
    reg = httpx.post(register_url, json={}, timeout=10)
    print(f"[hello_world_bench] Register trial: {reg.status_code} {register_url}")
    if reg.status_code >= 400:
        print(reg.text)
        raise RuntimeError(
            "Failed to register debug trial (did you restart backend after adding the route?)"
        )

    # This path is intentionally "chat/completions-ish"; OpenCode will call /responses off baseURL.
    inference_url = (
        f"{backend_base}/api/interceptor/v1/{trial_id}/chat/completions?cid={correlation_id}"
    )

    with tempfile.TemporaryDirectory(prefix="hello_world_bench_") as td:
        sandbox_dir = Path(td) / "sandbox"
        write_sandbox_files(sandbox_dir)

        prompt = (
            "TASK:\n"
            "1) First, READ output.txt using the read tool (required before editing/writing).\n"
            "2) Then use the write OR edit tool to modify output.txt.\n"
            "3) After your change, output.txt MUST contain EXACTLY this single line (no extra whitespace):\n"
            "Hello, world!\n"
            "4) Do NOT run bash. Do NOT read any other files.\n"
            "5) After writing the file, respond with exactly: DONE\n"
        )

        started = time.time()
        result = await run_opencode_agent(
            prompt,
            sandbox_dir,
            model=args.model,
            timeout=args.timeout,
            inference_url=inference_url,
            api_key=api_key,
        )
        elapsed = time.time() - started

        print(
            f"[hello_world_bench] OpenCode returncode={result['returncode']} elapsed={elapsed:.2f}s"
        )

        output_path = sandbox_dir / "output.txt"
        final_output = (
            output_path.read_text(encoding="utf-8") if output_path.exists() else "<missing>"
        )
        print("[hello_world_bench] Final output.txt:")
        print("-----")
        print(final_output.rstrip("\n"))
        print("-----")

        if final_output.strip() == "Hello, world!":
            print("[hello_world_bench] ✅ Success: output.txt matches exactly")
        else:
            print("[hello_world_bench] ❌ Failure: output.txt did not match")


if __name__ == "__main__":
    asyncio.run(main())
