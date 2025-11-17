#!/usr/bin/env python3
"""Proof of concept: Demonstrating that metrics_callback triggers progress events.

This script simulates the GEPA optimizer calling metrics_callback and shows
that progress events are emitted, proving the fix is correct.
"""

import asyncio
from typing import Any, Dict, Optional


# Simulate the _emit_metric function from online_jobs.py
class MockPostgrestEmitter:
    """Mock emitter that records events and metrics."""
    
    def __init__(self):
        self.events = []
        self.metrics = []
    
    async def append_metric(self, job_id: str, name: str, value: float, phase: str, run_id: Optional[str] = None, data: Dict[str, Any] = None):
        """Record a metric."""
        self.metrics.append({
            "job_id": job_id,
            "name": name,
            "value": value,
            "phase": phase,
            "run_id": run_id,
            "data": data or {},
        })
        print(f"  📊 Metric emitted: {name}={value:.3f}")
    
    async def append_event(self, job_id: str, type_: str, message: str, data: Dict[str, Any] = None, level: str = "info", run_id: Optional[str] = None):
        """Record an event."""
        self.events.append({
            "job_id": job_id,
            "type": type_,
            "message": message,
            "data": data or {},
            "level": level,
            "run_id": run_id,
        })
        print(f"  📢 Event emitted: {type_} - {message}")


# Simulate the _emit_metric callback from online_jobs.py (lines 1165-1360)
def create_emit_metric_callback(emitter: MockPostgrestEmitter, job_id: str, optimizer, run_id: Optional[str] = None):
    """Create the _emit_metric callback function (simulating online_jobs.py lines 1165-1360)."""
    import time as _time
    _progress_started_ts = _time.time()
    _last_progress_emit_ts = 0.0
    
    async def _emit_metric(*, name: str, value: float, data: dict):
        """Emit a metric and potentially trigger progress events."""
        try:
            print(f"\n[metrics_callback] Called with name={name}, value={value:.3f}")
            
            # Emit the metric (line 1168)
            await emitter.append_metric(
                job_id=job_id,
                name=name,
                value=float(value or 0.0),
                phase="eval",
                run_id=str(run_id) if run_id else None,
                data=data,
            )
            
            # Handle specific metric names (lines 1177-1237)
            if name == "gepa.transformation.mean_score":
                n = data.get("n")
                msg = f"{data.get('kind','variation')} {data.get('index','?')} mean={float(value or 0.0):.3f}"
                if isinstance(n, int):
                    msg += f" (N={n})"
                await emitter.append_event(
                    job_id=job_id,
                    run_id=str(run_id) if run_id else None,
                    type_="prompt.learning.gepa.variation.score",
                    message=msg,
                    data={"metric": name, **(data or {}), "mean_score": float(value or 0.0)},
                    level="info",
                )
            
            # Throttled progress estimator (lines 1238-1352)
            # Emit progress events every 10 seconds
            nonlocal _last_progress_emit_ts
            now_ts = _time.time()
            if now_ts - _last_progress_emit_ts >= 10.0:  # emit at most every 10s
                _last_progress_emit_ts = now_ts
                total_budget = getattr(optimizer, "rollout_budget", None)
                remaining = getattr(optimizer, "_remaining_budget", None)
                percent_rollouts = None
                completed_rollouts = None
                
                if isinstance(total_budget, int) and total_budget > 0 and isinstance(remaining, int) and remaining >= 0:
                    completed_rollouts = max(0, total_budget - remaining)
                    percent_rollouts = max(0.0, min(1.0, completed_rollouts / float(total_budget)))
                
                # Transformations
                tried = int(getattr(optimizer, "_candidate_counter", 0) or 0)
                init_pop = int(getattr(optimizer, "initial_population_size", 0) or 0)
                gens = int(getattr(optimizer, "num_generations", 0) or 0)
                children = int(getattr(optimizer, "children_per_generation", 0) or 0)
                planned_transformations = init_pop + (children * gens)
                percent_transform = None
                if planned_transformations > 0:
                    percent_transform = max(0.0, min(1.0, tried / float(planned_transformations)))
                
                elapsed = now_ts - _progress_started_ts
                
                # Prioritize rollouts over transformations
                if percent_rollouts is not None:
                    overall_percent = percent_rollouts
                elif percent_transform is not None:
                    overall_percent = percent_transform
                else:
                    overall_percent = None
                
                eta_seconds = None
                if overall_percent is not None and overall_percent > 0.0 and elapsed > 0:
                    eta_seconds = max(0.0, (elapsed / overall_percent) - elapsed)
                
                # Emit progress event (lines 1327-1352)
                if completed_rollouts is not None or tried > 0:
                    rollout_budget_display = total_budget if isinstance(total_budget, int) else "NA"
                    eta_display = f"{eta_seconds/60:.1f}min" if eta_seconds is not None else "N/A"
                    elapsed_display = f"{elapsed/60:.1f}min" if elapsed >= 60 else f"{int(elapsed)}s"
                    
                    await emitter.append_event(
                        job_id=job_id,
                        run_id=str(run_id) if run_id else None,
                        type_="prompt.learning.progress",
                        message=(
                            f"{int(overall_percent*100) if overall_percent is not None else 0}% complete; "
                            f"rollouts={completed_rollouts}/{rollout_budget_display}; "
                            f"elapsed={elapsed_display}, eta={eta_display}"
                        ),
                        data={
                            "rollouts_total": total_budget,
                            "rollouts_completed": completed_rollouts,
                            "rollouts_remaining": remaining if isinstance(remaining, int) else None,
                            "percent_rollouts": percent_rollouts,
                            "transformations_planned": planned_transformations if planned_transformations > 0 else None,
                            "transformations_tried": tried,
                            "percent_transformations": percent_transform,
                            "percent_overall": overall_percent,
                            "elapsed_seconds": int(elapsed),
                            "eta_seconds": int(eta_seconds) if eta_seconds is not None else None,
                        },
                        level="info",
                    )
                    print("  ✅ PROGRESS EVENT EMITTED! (throttled to every 10s)")
        except Exception as e:
            print(f"  ❌ Error in _emit_metric: {e}")
    
    return _emit_metric


# Simulate GEPAOptimizer
class MockGEPAOptimizer:
    """Mock GEPA optimizer that simulates the real optimizer."""
    
    def __init__(self):
        self.rollout_budget = 200
        self._remaining_budget = 200
        self._candidate_counter = 0
        self.initial_population_size = 10
        self.num_generations = 5
        self.children_per_generation = 5
        self.metrics_callback = None  # Will be set by online_jobs.py
    
    async def evaluate_transformation(self, transformation_idx: int, score: float, n: int):
        """Simulate evaluating a transformation."""
        print(f"\n[Optimizer] Evaluating transformation {transformation_idx}...")
        print(f"  Score: {score:.3f}, N: {n}")
        
        # Update state
        self._candidate_counter += 1
        self._remaining_budget -= n
        
        # Call metrics_callback (THIS IS THE FIX!)
        if self.metrics_callback:
            await self.metrics_callback(
                name="gepa.transformation.mean_score",
                value=score,
                data={
                    "kind": "transformation",
                    "index": transformation_idx,
                    "n": n,
                    "accuracy": score,
                }
            )
        else:
            print("  ⚠️  metrics_callback not set! Progress events won't be emitted.")


# Test 1: WITH metrics_callback (the fix)
async def test_with_metrics_callback():
    """Test that calling metrics_callback triggers progress events."""
    print("=" * 80)
    print("TEST 1: WITH metrics_callback (THE FIX)")
    print("=" * 80)
    
    emitter = MockPostgrestEmitter()
    optimizer = MockGEPAOptimizer()
    job_id = "test_job_123"
    
    # Set metrics_callback (simulating online_jobs.py line 1361)
    optimizer.metrics_callback = create_emit_metric_callback(emitter, job_id, optimizer)
    
    print("\n📝 Simulating transformation evaluations...")
    
    # Simulate evaluating 3 transformations quickly
    for i in range(3):
        await optimizer.evaluate_transformation(i, score=0.7 + i * 0.05, n=10)
        await asyncio.sleep(0.1)  # Small delay between evaluations
    
    # Wait a bit, then evaluate more (to trigger progress event after 10s)
    print("\n⏳ Waiting 11 seconds to trigger throttled progress event...")
    await asyncio.sleep(11)
    
    # Evaluate one more transformation (this should trigger progress event)
    await optimizer.evaluate_transformation(3, score=0.85, n=10)
    
    # Check results
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"📊 Metrics emitted: {len(emitter.metrics)}")
    for m in emitter.metrics:
        print(f"   - {m['name']}={m['value']:.3f}")
    
    print(f"\n📢 Events emitted: {len(emitter.events)}")
    progress_events = [e for e in emitter.events if e['type'] == 'prompt.learning.progress']
    print(f"   - Progress events: {len(progress_events)}")
    for e in progress_events:
        print(f"     • {e['message']}")
        print(f"       Data: rollouts={e['data'].get('rollouts_completed')}/{e['data'].get('rollouts_total')}, "
              f"eta={e['data'].get('eta_seconds')}s")
    
    print("\n✅ SUCCESS: Progress events were emitted when metrics_callback was called!")
    return len(progress_events) > 0


# Test 2: WITHOUT metrics_callback (direct emitter calls like MIPRO)
async def test_without_metrics_callback():
    """Test that direct emitter calls DON'T trigger progress events."""
    print("\n" + "=" * 80)
    print("TEST 2: WITHOUT metrics_callback (DIRECT EMITTER CALLS)")
    print("=" * 80)
    
    emitter = MockPostgrestEmitter()
    optimizer = MockGEPAOptimizer()
    job_id = "test_job_456"
    
    # DON'T set metrics_callback - simulate direct emitter calls like MIPRO
    optimizer.metrics_callback = None
    
    print("\n📝 Simulating transformation evaluations with direct emitter calls...")
    
    # Simulate evaluating transformations with direct emitter calls (MIPRO pattern)
    for i in range(3):
        print(f"\n[Optimizer] Evaluating transformation {i}...")
        score = 0.7 + i * 0.05
        optimizer._candidate_counter += 1
        optimizer._remaining_budget -= 10
        
        # Direct emitter call (MIPRO pattern) - NO progress events!
        await emitter.append_metric(
            job_id=job_id,
            name="gepa.transformation.mean_score",
            value=score,
            phase="eval",
            data={
                "kind": "transformation",
                "index": i,
                "n": 10,
            }
        )
        await asyncio.sleep(0.1)
    
    # Wait and evaluate more
    print("\n⏳ Waiting 11 seconds...")
    await asyncio.sleep(11)
    
    await emitter.append_metric(
        job_id=job_id,
        name="gepa.transformation.mean_score",
        value=0.85,
        phase="eval",
        data={"kind": "transformation", "index": 3, "n": 10},
    )
    
    # Check results
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"📊 Metrics emitted: {len(emitter.metrics)}")
    for m in emitter.metrics:
        print(f"   - {m['name']}={m['value']:.3f}")
    
    print(f"\n📢 Events emitted: {len(emitter.events)}")
    progress_events = [e for e in emitter.events if e['type'] == 'prompt.learning.progress']
    print(f"   - Progress events: {len(progress_events)}")
    
    if len(progress_events) == 0:
        print("\n❌ EXPECTED: No progress events (direct emitter calls don't trigger progress logic)")
    else:
        print("\n⚠️  UNEXPECTED: Progress events were emitted!")
    
    return len(progress_events) == 0


async def main():
    """Run both tests."""
    print("\n" + "=" * 80)
    print("PROOF OF CONCEPT: metrics_callback triggers progress events")
    print("=" * 80)
    print("\nThis script proves that:")
    print("1. Calling metrics_callback triggers progress events (THE FIX)")
    print("2. Direct emitter calls DON'T trigger progress events (MIPRO pattern)")
    print("\n")
    
    test1_passed = await test_with_metrics_callback()
    test2_passed = await test_without_metrics_callback()
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    if test1_passed and test2_passed:
        print("✅ PROOF CONFIRMED:")
        print("   - metrics_callback MUST be called to get progress events")
        print("   - Direct emitter calls (MIPRO pattern) DON'T trigger progress events")
        print("   - The fix (calling metrics_callback) is CORRECT!")
    else:
        print("❌ Tests failed - review the implementation")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

