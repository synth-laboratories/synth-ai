#!/usr/bin/env python3
"""Capture OpenCode request - just save it, don't forward."""

import asyncio
import json
import os
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

captured = []

app = FastAPI()


@app.post("/v1/responses")
@app.post("/responses")
async def capture(request: Request):
    body = await request.json()
    captured.append(body)

    # Save immediately
    out = Path(__file__).parent / "captured_messages.json"
    with open(out, "w") as f:
        json.dump(body, f, indent=2)
    print(f"SAVED to {out}")

    # Return minimal response so OpenCode doesn't hang forever
    return JSONResponse({"error": "capture only"}, status_code=500)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def catch_all(request: Request, path: str):
    print(f"[{request.method}] /{path}")
    return JSONResponse({"error": "not implemented"}, status_code=404)


async def main():
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    port = 18765

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    await asyncio.sleep(1)

    with tempfile.TemporaryDirectory(prefix="capture_") as td:
        sandbox = Path(td)

        cfg = {
            "$schema": "https://opencode.ai/config.json",
            "model": "openai/gpt-4o-mini",
            "provider": {
                "openai": {"options": {"apiKey": api_key, "baseURL": f"http://127.0.0.1:{port}/v1"}}
            },
            "permission": {"*": "allow"},
        }
        (sandbox / "opencode.json").write_text(json.dumps(cfg))
        (sandbox / "output.txt").write_text("placeholder\n")
        (sandbox / "AGENTS.md").write_text("Follow the task.\n")

        prompt = "Edit output.txt to contain exactly: Hello, world!"
        print(f"Running opencode: {prompt}")

        opencode = str(Path.home() / ".opencode" / "bin" / "opencode")
        proc = await asyncio.create_subprocess_exec(
            opencode,
            "run",
            "--format",
            "json",
            "--model",
            "openai/gpt-4o-mini",
            prompt,
            cwd=str(sandbox),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "OPENAI_API_KEY": api_key},
        )

        try:
            await asyncio.wait_for(proc.communicate(), timeout=30)
        except TimeoutError:
            proc.kill()

        print(f"Exit: {proc.returncode}")

    server.should_exit = True
    print(f"\nCaptured {len(captured)} requests")
    if captured:
        print(f"Saved to: {Path(__file__).parent / 'captured_messages.json'}")


if __name__ == "__main__":
    asyncio.run(main())
