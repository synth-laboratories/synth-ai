"""Offline concurrent-upload memory and event-loop benchmark; no provider traffic."""

from __future__ import annotations

import argparse
import asyncio
import json
import resource
import tempfile
import time
import tracemalloc
from pathlib import Path

import httpx
from pydantic import JsonValue
from typing_extensions import override

from synth_ai import AsyncSynthClient


class Sink(httpx.AsyncBaseTransport):
    def __init__(self, size: int) -> None:
        self.size = size
        self.transferred = 0
        self.digest = "0" * 64

    @override
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            assert isinstance(request.stream, httpx.AsyncByteStream)
            async for chunk in request.stream:
                self.transferred += len(chunk)
                await asyncio.sleep(0)
            return httpx.Response(200)
        file: dict[str, JsonValue] = {
            "file_id": "file_AAAAAAAAAAAAAAAAAAAAAA",
            "revision_id": "rev_AAAAAAAAAAAAAAAAAAAAAA",
            "name": "task.txt",
            "media_type": "text/plain",
            "size_bytes": self.size,
            "digest_sha256": self.digest,
            "created_at": "2026-11-09T00:00:00Z",
        }
        if request.url.path.endswith("/uploads"):
            body = json.loads(request.content)
            self.digest = body["digest_sha256"]
            file["digest_sha256"] = self.digest
            return httpx.Response(
                201,
                json={
                    "upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA",
                    "state": "pending",
                    "file": file,
                    "put": {
                        "url": "https://storage.example.test/blob",
                        "headers": {"content-length": str(self.size)},
                        "expires_at": "2099-01-01T00:00:00Z",
                    },
                },
            )
        return httpx.Response(
            200,
            json={"upload_id": "upl_AAAAAAAAAAAAAAAAAAAAAA", "state": "committed", "file": file},
        )


async def measure(directory: Path, concurrency: int, size: int) -> dict[str, JsonValue]:
    with tempfile.TemporaryDirectory(dir=directory, prefix="upload-benchmark-") as temporary:
        source = Path(temporary) / "task.txt"
        with source.open("wb") as stream:
            for _ in range(size // (1 << 20)):
                stream.write(b"x" * (1 << 20))
        delays: list[float] = []
        running = True

        async def ticker() -> None:
            while running:
                started = time.monotonic()
                await asyncio.sleep(0.001)
                delays.append(max(0.0, time.monotonic() - started - 0.001))

        sink = Sink(size)
        tracemalloc.start()
        started = time.monotonic()
        clock = asyncio.create_task(ticker())
        await asyncio.sleep(0)
        async with httpx.AsyncClient(transport=sink) as http:
            async with AsyncSynthClient(
                "fixture", base_url="https://api.example.test", http_client=http
            ) as client:
                await asyncio.gather(
                    *(client.swarms.upload_file(str(source)) for _ in range(concurrency))
                )
        running = False
        await clock
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "concurrency": concurrency,
            "file_bytes": size,
            "transferred_bytes": sink.transferred,
            "peak_python_bytes": peak,
            "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "max_loop_delay_seconds": max(delays, default=0.0),
            "elapsed_seconds": time.monotonic() - started,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--mib", type=int, default=32)
    args = parser.parse_args()
    print(
        json.dumps(asyncio.run(measure(args.directory, args.concurrency, args.mib << 20)), indent=2)
    )


if __name__ == "__main__":
    main()
