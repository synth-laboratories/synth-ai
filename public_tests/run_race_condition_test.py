#!/usr/bin/env python3
"""
Simple test runner to demonstrate the DuckDB race condition issue.

This script runs a minimal reproduction of the race condition that occurs
in the Crafter rollout script. It's designed to fail with the current
implementation and pass once the race condition is fixed.

Usage:
    python run_race_condition_test.py
"""

import asyncio
import tempfile
import os
import uuid
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add the parent directory to sys.path so we can import synth_ai modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
from synth_ai.tracing_v2.session_tracer import (
    SessionTracer, SessionEventMessage, TimeRecord,
    RuntimeEvent
)


def test_race_condition_minimal():
    """
    Minimal test that reproduces the exact race condition from the rollout script.
    
    This simulates multiple episode tracers trying to insert sessions with the
    same session_id simultaneously, causing "Duplicate key" constraint violations.
    """
    print("🧪 Testing DuckDB race condition (minimal reproduction)")
    print("=" * 60)
    
    # Create temporary database and traces directory  
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_race.duckdb")
        traces_dir = os.path.join(temp_dir, "traces")
        os.makedirs(traces_dir, exist_ok=True)
        
        try:
            session_id_base = f"episode_test_{uuid.uuid4()}"
            num_workers = 8
            
            def worker_simulate_episode(worker_id: int):
                """Simulate what happens in run_episode_async function."""
                try:
                    # Each worker creates its own tracer (like in the real script)
                    tracer = SessionTracer(
                        traces_dir=traces_dir,
                        duckdb_path=db_path
                    )
                    
                    # Force race condition: multiple workers use same session_id
                    # This simulates the bug where session IDs aren't truly unique
                    session_id = f"{session_id_base}_{worker_id % 3}"  # Only 3 unique IDs for 8 workers
                    
                    print(f"Worker {worker_id}: Starting session {session_id}")
                    
                    # Start session (like in run_episode_async)
                    tracer.start_session(session_id)
                    tracer.start_timestep(f"turn_{worker_id}")
                    
                    # Add some data (like the game data in Crafter)
                    message = SessionEventMessage(
                        content={"action": "move_right", "reward": 0.0},
                        message_type="observation"
                    )
                    tracer.record_message(message)
                    
                    event = RuntimeEvent(
                        system_instance_id=f"crafter_agent_{worker_id}",
                        actions=[2],  # move_right
                        metadata={"step": 0, "reward": 0.0}
                    )
                    tracer.record_event(event)
                    
                    # End session - THIS IS WHERE THE RACE CONDITION OCCURS
                    # Multiple workers try to insert the same session_id
                    tracer.end_session()
                    tracer.close()
                    
                    print(f"Worker {worker_id}: ✅ SUCCESS")
                    return {"worker_id": worker_id, "success": True, "error": None}
                    
                except Exception as e:
                    error_msg = str(e)
                    if "Duplicate key" in error_msg:
                        print(f"Worker {worker_id}: ❌ RACE CONDITION ERROR")
                    else:
                        print(f"Worker {worker_id}: ❌ OTHER ERROR: {error_msg}")
                    return {"worker_id": worker_id, "success": False, "error": error_msg}
            
            print(f"Running {num_workers} workers concurrently...")
            print(f"Using {num_workers % 3 + 1} unique session IDs to force collisions\n")
            
            # Run workers concurrently (like the actual rollout script)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(worker_simulate_episode, i) 
                    for i in range(num_workers)
                ]
                
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
            
            # Analyze results
            successful = [r for r in results if r["success"]]
            failed = [r for r in results if not r["success"]]
            race_errors = [
                r for r in failed 
                if "Duplicate key" in r["error"] and "violates primary key constraint" in r["error"]
            ]
            
            print(f"\n📊 RESULTS:")
            print(f"  Total workers: {num_workers}")
            print(f"  Successful: {len(successful)}")
            print(f"  Failed: {len(failed)}")
            print(f"  Race condition errors: {len(race_errors)}")
            
            if race_errors:
                print(f"\n🚨 RACE CONDITION DETECTED!")
                print(f"   {len(race_errors)} workers failed due to duplicate key constraints")
                print(f"   This is the same error occurring in the rollout script.")
                print(f"\n💡 TO FIX THIS:")
                print(f"   Option A: Add database write semaphore to serialize insertions")
                print(f"   Option B: Use 'ON CONFLICT DO NOTHING' in INSERT statements")
                return False
            else:
                print(f"\n✅ NO RACE CONDITION DETECTED!")
                print(f"   All workers completed successfully.")
                print(f"   The race condition fix is working correctly.")
                return True
                 
        except Exception as e:
            print(f"\nUnexpected error in sync test: {e}")
            return False


async def test_race_condition_async():
    """
    Async version that more closely simulates the actual rollout script.
    """
    print("\n🧪 Testing DuckDB race condition (async version)")
    print("=" * 60)
    
    # Create temporary database and traces directory
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_async_race.duckdb")
        traces_dir = os.path.join(temp_dir, "traces")
        os.makedirs(traces_dir, exist_ok=True)
        
        try:
            session_id_base = f"async_episode_{uuid.uuid4()}"
            num_tasks = 6
            
            async def async_simulate_episode(task_id: int):
                """Async simulation of episode execution."""
                try:
                    tracer = SessionTracer(
                        traces_dir=traces_dir,
                        duckdb_path=db_path
                    )
                    
                    # Force race condition with async tasks
                    session_id = f"{session_id_base}_{task_id % 2}"  # Only 2 unique IDs for 6 tasks
                    
                    print(f"Async Task {task_id}: Starting session {session_id}")
                    
                    tracer.start_session(session_id)
                    tracer.start_timestep(f"async_turn_{task_id}")
                    
                    # Add data
                    event = RuntimeEvent(
                        system_instance_id=f"async_crafter_{task_id}",
                        actions=[1],  # move_left
                        metadata={"async_task": task_id}
                    )
                    tracer.record_event(event)
                    
                    # Small delay to increase race condition chance
                    await asyncio.sleep(0.001)
                    
                    # End session - race condition point
                    tracer.end_session()
                    tracer.close()
                    
                    print(f"Async Task {task_id}: ✅ SUCCESS")
                    return {"task_id": task_id, "success": True, "error": None}
                    
                except Exception as e:
                    error_msg = str(e)
                    if "Duplicate key" in error_msg:
                        print(f"Async Task {task_id}: ❌ RACE CONDITION ERROR")
                    else:
                        print(f"Async Task {task_id}: ❌ OTHER ERROR: {error_msg}")
                    return {"task_id": task_id, "success": False, "error": error_msg}
            
            print(f"Running {num_tasks} async tasks concurrently...")
            print(f"Using {num_tasks % 2 + 1} unique session IDs to force collisions\n")
            
            # Run async tasks (like asyncio.gather in rollout script)
            tasks = [async_simulate_episode(i) for i in range(num_tasks)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "task_id": i, 
                        "success": False, 
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            
            # Analyze async results
            successful = [r for r in processed_results if r["success"]]
            failed = [r for r in processed_results if not r["success"]]
            race_errors = [
                r for r in failed 
                if "Duplicate key" in r["error"]
            ]
            
            print(f"\n📊 ASYNC RESULTS:")
            print(f"  Total async tasks: {num_tasks}")
            print(f"  Successful: {len(successful)}")
            print(f"  Failed: {len(failed)}")
            print(f"  Race condition errors: {len(race_errors)}")
            
            if race_errors:
                print(f"\n🚨 ASYNC RACE CONDITION DETECTED!")
                print(f"   {len(race_errors)} tasks failed due to duplicate key constraints")
                return False
            else:
                                print(f"\n✅ NO ASYNC RACE CONDITION DETECTED!")
                return True
                 
        except Exception as e:
            print(f"\nUnexpected error in async test: {e}")
            return False


def main():
    """Main test runner."""
    print("🔍 DuckDB Race Condition Test Suite")
    print("=" * 70)
    print("This test reproduces the race condition from run_rollouts_for_models_and_compare.py")
    print("where multiple concurrent episodes cause 'Duplicate key' constraint violations.\n")
    
    # Run synchronous test
    sync_success = test_race_condition_minimal()
    
    # Run asynchronous test
    async_success = asyncio.run(test_race_condition_async())
    
    # Final summary
    print(f"\n" + "=" * 70)
    print(f"🏁 FINAL RESULTS:")
    print(f"  Synchronous test: {'✅ PASSED' if sync_success else '❌ FAILED (race condition)'}")
    print(f"  Asynchronous test: {'✅ PASSED' if async_success else '❌ FAILED (race condition)'}")
    
    if not sync_success or not async_success:
        print(f"\n💥 RACE CONDITION CONFIRMED!")
        print(f"   This is the same issue causing warnings in the rollout script.")
        print(f"   The database insertion logic is not thread/async-safe.")
        print(f"\n🔧 RECOMMENDED FIXES:")
        print(f"   1. Add a semaphore to serialize database writes")
        print(f"   2. Use 'INSERT ... ON CONFLICT DO NOTHING' in the DB manager")
        print(f"   3. Implement proper database transaction locking")
        
        return 1  # Exit code indicates failure
    else:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   No race conditions detected. The fix is working correctly!")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 