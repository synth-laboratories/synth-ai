#!/usr/bin/env python3
"""
Tests for DuckDB race condition issues in concurrent environments.

These tests reproduce the race condition that occurs when multiple
SessionTracer instances try to insert session traces simultaneously,
causing "Duplicate key" constraint violations.

These tests will FAIL with the current implementation and should PASS
once the race condition is fixed (either with semaphore or ON CONFLICT DO NOTHING).
"""

import pytest
import asyncio
import tempfile
import os
import uuid
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
from synth_ai.tracing_v2.session_tracer import (
    SessionTracer, SessionEventMessage, TimeRecord,
    RuntimeEvent, EnvironmentEvent, SessionTrace
)


class TestDuckDBRaceConditions:
    """Test race conditions in concurrent DuckDB operations."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary DuckDB path inside a temp directory (file doesn't exist yet)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_concurrency.duckdb")
            yield db_path
    
    @pytest.fixture  
    def temp_traces_dir(self):
        """Create a temporary traces directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.mark.slow
    def test_concurrent_session_insertion_race_condition(self, temp_db_path, temp_traces_dir):
        """
        Test that reproduces the race condition with concurrent session insertions.
        
        This test will FAIL with current implementation due to race condition.
        It should PASS once the fix is implemented.
        """
        num_concurrent_sessions = 10
        session_id_base = f"test_session_{uuid.uuid4()}"
        
        def create_and_insert_session(worker_id: int):
            """Create a session tracer and insert a session trace."""
            try:
                # Each worker creates its own tracer (and thus its own DB connection)
                tracer = SessionTracer(
                    traces_dir=temp_traces_dir,
                    duckdb_path=temp_db_path
                )
                
                # Use the same session_id to force the race condition
                session_id = f"{session_id_base}_{worker_id % 3}"  # Only 3 unique IDs for 10 workers
                
                # Start session and add some data
                tracer.start_session(session_id)
                tracer.start_timestep(f"step_{worker_id}")
                
                # Add a message
                message = SessionEventMessage(
                    content=f"Test message from worker {worker_id}",
                    message_type="test",
                    time_record=TimeRecord(
                        event_time=datetime.now().isoformat(),
                        message_time=0
                    )
                )
                tracer.record_message(message)
                
                # Add an event
                event = RuntimeEvent(
                    system_instance_id=f"test_system_{worker_id}",
                    actions=[f"action_{worker_id}"],
                    metadata={"worker_id": worker_id}
                )
                tracer.record_event(event)
                
                # End session (this triggers the DB insertion)
                tracer.end_session()
                tracer.close()
                
                return {"worker_id": worker_id, "success": True, "error": None}
                
            except Exception as e:
                return {"worker_id": worker_id, "success": False, "error": str(e)}
        
        # Run concurrent insertions
        with ThreadPoolExecutor(max_workers=num_concurrent_sessions) as executor:
            futures = [
                executor.submit(create_and_insert_session, i) 
                for i in range(num_concurrent_sessions)
            ]
            
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        duplicate_key_errors = [
            r for r in failed 
            if "Duplicate key" in r["error"] and "violates primary key constraint" in r["error"]
        ]
        
        print(f"\nConcurrency test results:")
        print(f"  Total workers: {num_concurrent_sessions}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Duplicate key errors: {len(duplicate_key_errors)}")
        
        if failed:
            print("\nFailure details:")
            for f in failed:
                print(f"  Worker {f['worker_id']}: {f['error']}")
        
        # This assertion will FAIL with current implementation
        # It should PASS once race condition is fixed
        assert len(duplicate_key_errors) == 0, (
            f"Race condition detected: {len(duplicate_key_errors)} duplicate key errors occurred. "
            f"This indicates the database insertion is not thread-safe."
        )
        
        # Additional verification: check that data was actually inserted
        with DuckDBTraceManager(temp_db_path) as db:
            sessions_df = db.conn.execute("SELECT session_id FROM session_traces").df()
            assert len(sessions_df) > 0, "No sessions were inserted successfully"

    @pytest.mark.slow 
    async def test_async_concurrent_session_insertion_race_condition(self, temp_db_path, temp_traces_dir):
        """
        Test race condition with async concurrent session insertions.
        
        This simulates the actual use case from run_rollouts_for_models_and_compare.py
        where multiple async coroutines create sessions simultaneously.
        """
        num_concurrent_sessions = 8
        session_id_base = f"async_test_session_{uuid.uuid4()}"
        
        async def create_and_insert_session_async(worker_id: int):
            """Async version of session creation and insertion."""
            try:
                # Create tracer with unique session ID that might collide
                tracer = SessionTracer(
                    traces_dir=temp_traces_dir,
                    duckdb_path=temp_db_path
                )
                
                # Force potential collision by using same base session ID
                session_id = f"{session_id_base}_{worker_id % 2}"  # Only 2 unique IDs for 8 workers
                
                tracer.start_session(session_id)
                tracer.start_timestep(f"async_step_{worker_id}")
                
                # Add some realistic game data
                message = SessionEventMessage(
                    content={"action": "move_right", "observation": {"health": 9, "hunger": 8}},
                    message_type="observation",
                    time_record=TimeRecord(
                        event_time=datetime.now().isoformat(),
                        message_time=worker_id
                    )
                )
                tracer.record_message(message)
                
                event = RuntimeEvent(
                    system_instance_id=f"crafter_agent_async_{worker_id}",
                    actions=[2],  # move_right action
                    metadata={
                        "step": worker_id,
                        "reward": 0.0,
                        "done": False,
                        "invalid_action": False
                    }
                )
                tracer.record_event(event)
                
                # Small async delay to increase chance of race condition
                await asyncio.sleep(0.001 * worker_id)
                
                # End session - this is where the race condition occurs
                tracer.end_session()
                tracer.close()
                
                return {"worker_id": worker_id, "success": True, "error": None}
                
            except Exception as e:
                return {"worker_id": worker_id, "success": False, "error": str(e)}
        
        # Create and run concurrent async tasks
        tasks = [
            create_and_insert_session_async(i) 
            for i in range(num_concurrent_sessions)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert any exceptions to error results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "worker_id": i, 
                    "success": False, 
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        # Analyze results
        successful = [r for r in processed_results if r["success"]]
        failed = [r for r in processed_results if not r["success"]]
        duplicate_key_errors = [
            r for r in failed 
            if "Duplicate key" in r["error"] and "violates primary key constraint" in r["error"]
        ]
        
        print(f"\nAsync concurrency test results:")
        print(f"  Total async tasks: {num_concurrent_sessions}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Duplicate key errors: {len(duplicate_key_errors)}")
        
        if failed:
            print("\nAsync failure details:")
            for f in failed:
                print(f"  Task {f['worker_id']}: {f['error']}")
        
        # This assertion will FAIL with current implementation
        # Should PASS once race condition is fixed
        assert len(duplicate_key_errors) == 0, (
            f"Async race condition detected: {len(duplicate_key_errors)} duplicate key errors occurred. "
            f"This indicates the database insertion is not async-safe."
        )

    @pytest.mark.slow
    def test_experiment_linking_race_condition(self, temp_db_path, temp_traces_dir):
        """
        Test race condition specifically in experiment linking logic.
        
        This reproduces the exact scenario from the rollout script where
        sessions are linked to experiments after insertion.
        """
        experiment_id = str(uuid.uuid4())
        session_id_base = f"link_test_session_{uuid.uuid4()}"
        
        # Create experiment first
        with DuckDBTraceManager(temp_db_path) as db:
            db.create_experiment(
                experiment_id=experiment_id,
                name="Test Experiment",
                description="Test experiment for race condition"
            )
        
        def create_session_and_link(worker_id: int):
            """Create session and link to experiment - reproduces actual rollout logic."""
            try:
                tracer = SessionTracer(
                    traces_dir=temp_traces_dir,
                    duckdb_path=temp_db_path
                )
                
                # Use same session ID to force race
                session_id = f"{session_id_base}_{worker_id % 3}"
                
                tracer.start_session(session_id)
                tracer.start_timestep("test_step")
                
                # Add minimal data
                event = RuntimeEvent(
                    system_instance_id=f"test_runtime_{worker_id}",
                    metadata={"test": True}
                )
                tracer.record_event(event)
                
                # End session (inserts into DB)
                tracer.end_session()
                tracer.close()
                
                # Now try to link to experiment (this is where the race happens)
                with DuckDBTraceManager(temp_db_path) as db:
                    db.link_session_to_experiment(session_id, experiment_id)
                
                return {"worker_id": worker_id, "success": True, "error": None}
                
            except Exception as e:
                return {"worker_id": worker_id, "success": False, "error": str(e)}
        
        # Run concurrent linking operations
        num_workers = 6
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(create_session_and_link, i) 
                for i in range(num_workers)
            ]
            
            results = []
            for future in as_completed(futures):
                results.append(future.result())
        
        # Analyze results
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        duplicate_key_errors = [
            r for r in failed 
            if ("Duplicate key" in r["error"] or 
                "violates primary key constraint" in r["error"] or
                "already exists" in r["error"])
        ]
        
        print(f"\nExperiment linking test results:")
        print(f"  Total workers: {num_workers}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Race condition errors: {len(duplicate_key_errors)}")
        
        if failed:
            print("\nLinking failure details:")
            for f in failed:
                print(f"  Worker {f['worker_id']}: {f['error']}")
        
        # This will FAIL with current implementation but PASS once fixed
        assert len(duplicate_key_errors) == 0, (
            f"Experiment linking race condition detected: {len(duplicate_key_errors)} errors occurred."
        )

    @pytest.mark.slow
    def test_high_concurrency_stress_test(self, temp_db_path, temp_traces_dir):
        """
        Stress test with high concurrency to maximize chance of hitting race condition.
        
        This test uses a higher number of concurrent operations to stress-test
        the database insertion logic.
        """
        num_workers = 20  # High concurrency
        session_id_base = f"stress_test_{uuid.uuid4()}"
        
        def stress_worker(worker_id: int):
            """High-stress worker that creates multiple sessions rapidly."""
            results = []
            
            for session_num in range(3):  # Each worker creates 3 sessions
                try:
                    tracer = SessionTracer(
                        traces_dir=temp_traces_dir,
                        duckdb_path=temp_db_path
                    )
                    
                    # Force collision by having multiple workers use same session IDs
                    session_id = f"{session_id_base}_{(worker_id * 3 + session_num) % 5}"
                    
                    tracer.start_session(session_id)
                    tracer.start_timestep(f"stress_step_{worker_id}_{session_num}")
                    
                    # Quick data insertion
                    event = RuntimeEvent(
                        system_instance_id=f"stress_system_{worker_id}",
                        metadata={"worker": worker_id, "session": session_num}
                    )
                    tracer.record_event(event)
                    
                    # Rapid-fire ending
                    tracer.end_session()
                    tracer.close()
                    
                    results.append({
                        "worker_id": worker_id, 
                        "session_num": session_num,
                        "success": True, 
                        "error": None
                    })
                    
                except Exception as e:
                    results.append({
                        "worker_id": worker_id,
                        "session_num": session_num, 
                        "success": False, 
                        "error": str(e)
                    })
            
            return results
        
        # Run high-concurrency stress test
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(num_workers)]
            
            all_results = []
            for future in as_completed(futures):
                worker_results = future.result()
                all_results.extend(worker_results)
        
        # Analyze stress test results
        total_operations = len(all_results)
        successful = [r for r in all_results if r["success"]]
        failed = [r for r in all_results if not r["success"]]
        race_errors = [
            r for r in failed 
            if "Duplicate key" in r["error"] or "constraint" in r["error"]
        ]
        
        print(f"\nStress test results:")
        print(f"  Total operations: {total_operations}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Race condition errors: {len(race_errors)}")
        print(f"  Success rate: {len(successful)/total_operations*100:.1f}%")
        
        # This assertion will FAIL with current implementation
        assert len(race_errors) == 0, (
            f"High-concurrency stress test detected {len(race_errors)} race condition errors "
            f"out of {total_operations} total operations. Database insertion is not thread-safe."
        )


if __name__ == "__main__":
    """Run the race condition tests directly."""
    print("Running DuckDB race condition tests...")
    print("These tests will FAIL with the current implementation due to race conditions.")
    print("They should PASS once the database insertion race condition is fixed.\n")
    
    pytest.main([__file__, "-v", "-s"]) 