#!/usr/bin/env python3
"""
Quick test of Daytona provisioning only - no full eval.

Usage:
    export DAYTONA_API_KEY=...
    uv run python demos/engine_bench/test_daytona_provision.py
"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from daytona_helper import DaytonaTaskAppRunner


async def main():
    print("=" * 60)
    print("TESTING DAYTONA PROVISIONING")
    print("=" * 60)

    if not os.environ.get("DAYTONA_API_KEY"):
        print("ERROR: DAYTONA_API_KEY not set")
        sys.exit(1)

    start = time.time()
    runner = DaytonaTaskAppRunner(use_snapshot=True)

    try:
        # Test 1: Provision
        t0 = time.time()
        await runner.provision()
        provision_time = time.time() - t0
        print(f"✅ Provision: {provision_time:.1f}s (sandbox: {runner.sandbox_id[:12]}...)")
        print(f"   From snapshot: {runner._created_from_snapshot}")
        print(f"   Preview URL: {runner.preview_url}")

        # Test 2: Upload file
        t0 = time.time()
        test_content = b"print('hello from daytona')"
        await asyncio.to_thread(
            runner.sandbox.fs.upload_file,
            test_content,
            "/app/test.py",
        )
        upload_time = time.time() - t0
        print(f"✅ Upload file: {upload_time:.1f}s")

        # Test 3: Run command
        t0 = time.time()
        result = await asyncio.to_thread(runner.sandbox.process.exec, "python3 /app/test.py")
        exec_time = time.time() - t0
        output = getattr(result, "output", "") or getattr(result, "result", "") or ""
        print(f"✅ Run command: {exec_time:.1f}s (output: {output.strip()})")

        total = time.time() - start
        print()
        print("=" * 60)
        print(f"TOTAL TIME: {total:.1f}s")
        if runner._created_from_snapshot:
            print("✅ Used cached snapshot - fast path!")
        else:
            print("⚠️ No snapshot found - consider running create_daytona_snapshot.py")
        print("=" * 60)

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
