#!/usr/bin/env python3
"""Stress test for banking77 task app to reproduce crashes."""

import asyncio
import time
import httpx

TASK_APP_URL = "http://localhost:8001"
API_KEY = "sk_env_30c78a787bac223c716918181209f263"
INTERCEPTOR_URL = "http://localhost:8000/api/interceptor/v1"

async def single_rollout(seed: int, client: httpx.AsyncClient) -> tuple[int, float, str]:
    """Execute a single rollout and return (seed, duration, status)."""
    start = time.time()
    payload = {
        "run_id": f"stress-test-{seed}",
        "mode": "rl",  # Valid modes are "rl" or "eval"
        "env": {
            "env_name": "banking77",
            "seed": seed,
            "config": {"split": "train"},
        },
        "policy": {
            "config": {
                "model": "gpt-4.1-nano",
                "provider": "openai",
                "inference_url": INTERCEPTOR_URL,
                "temperature": 0.0,
                "max_completion_tokens": 256,
            }
        },
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "X-API-Key": API_KEY,
        "Content-Type": "application/json",
    }

    try:
        response = await client.post(
            f"{TASK_APP_URL}/rollout",
            json=payload,
            headers=headers,
            timeout=120.0,  # 2 minute timeout
        )
        elapsed = time.time() - start
        return seed, elapsed, f"status={response.status_code}"
    except Exception as e:
        elapsed = time.time() - start
        return seed, elapsed, f"error={type(e).__name__}: {e}"


async def stress_test(
    num_requests: int = 10,
    concurrent: int = 5,
    delay_between_batches: float = 0.0,
):
    """Run stress test with specified concurrency."""
    print(f"\n{'='*60}")
    print(f"STRESS TEST: {num_requests} requests, {concurrent} concurrent")
    print(f"{'='*60}\n")

    # First check health
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(
                f"{TASK_APP_URL}/health",
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=5.0,
            )
            print(f"[HEALTH] Task app: {health.status_code} - {health.text[:100]}")
        except Exception as e:
            print(f"[HEALTH] Task app FAILED: {e}")
            return

    # Run stress test
    results = []
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        # Process in batches
        for batch_start in range(0, num_requests, concurrent):
            batch_end = min(batch_start + concurrent, num_requests)
            batch_seeds = list(range(batch_start, batch_end))

            print(f"\n[BATCH {batch_start//concurrent + 1}] Starting seeds {batch_seeds}...")
            batch_start_time = time.time()

            batch_results = await asyncio.gather(
                *[single_rollout(seed, client) for seed in batch_seeds],
                return_exceptions=True,
            )

            batch_elapsed = time.time() - batch_start_time

            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"  [ERROR] {type(result).__name__}: {result}")
                    results.append((None, 0.0, f"exception={result}"))
                else:
                    seed, elapsed, status = result
                    results.append(result)
                    print(f"  [RESULT] seed={seed} elapsed={elapsed:.2f}s {status}")

            print(f"  [BATCH DONE] {len(batch_results)} requests in {batch_elapsed:.2f}s")

            # Check health after each batch
            try:
                health = await client.get(
                    f"{TASK_APP_URL}/health",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    timeout=5.0,
                )
                print(f"  [HEALTH] {health.status_code}")
            except Exception as e:
                print(f"  [HEALTH] FAILED: {type(e).__name__}: {e}")
                print("\n*** TASK APP CRASHED! ***\n")
                break

            if delay_between_batches > 0:
                await asyncio.sleep(delay_between_batches)

    total_elapsed = time.time() - start_time

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    successful = [r for r in results if "status=200" in r[2]]
    failed = [r for r in results if "status=200" not in r[2]]

    print(f"Total requests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_elapsed:.2f}s")

    if successful:
        durations = [r[1] for r in successful]
        print(f"Avg duration: {sum(durations)/len(durations):.2f}s")
        print(f"Max duration: {max(durations):.2f}s")
        print(f"Min duration: {min(durations):.2f}s")

    if failed:
        print("\nFailed requests:")
        for r in failed[:10]:
            print(f"  {r}")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-requests", type=int, default=10)
    parser.add_argument("-c", "--concurrent", type=int, default=5)
    parser.add_argument("-d", "--delay", type=float, default=0.0)
    args = parser.parse_args()

    await stress_test(
        num_requests=args.num_requests,
        concurrent=args.concurrent,
        delay_between_batches=args.delay,
    )


if __name__ == "__main__":
    asyncio.run(main())
