#!/usr/bin/env python3
"""
Tests for concurrent operations in Turso/sqld.

These tests verify that the Turso implementation properly handles
concurrent operations that would cause race conditions in DuckDB.
These tests should PASS with Turso's multi-writer MVCC.
"""

import asyncio
import os
import shutil
import tempfile
import time
import uuid

import pytest
import pytest_asyncio

from ..abstractions import (
    EnvironmentEvent,
    LMCAISEvent,
    RuntimeEvent,
    TimeRecord,
)
from ..config import CONFIG
from ..session_tracer import SessionTracer
from ..turso.daemon import SqldDaemon
from ..turso.native_manager import NativeLibsqlTraceManager


if shutil.which(CONFIG.sqld_binary) is None and shutil.which("libsql-server") is None:
    pytest.skip(
        "sqld binary not available; install Turso sqld or set SQLD_BINARY to skip these tests",
        allow_module_level=True,
    )

# Proactively verify that sqld can start in this environment. Some sandboxed CI
# environments block the process from binding to localhost, resulting in
# `Operation not permitted` errors when the daemon launches. If we detect that
# condition, skip the module instead of failing all tests.
with tempfile.TemporaryDirectory(prefix="sqld_probing_") as _probe_dir:
    _probe_daemon = SqldDaemon(db_path=os.path.join(_probe_dir, "probe.db"), http_port=0)
    try:
        _probe_daemon.start()
    except RuntimeError as exc:  # pragma: no cover - environment dependent
        if "Operation not permitted" in str(exc) or "Permission denied" in str(exc):
            pytest.skip(
                "sqld daemon cannot start in this environment (Operation not permitted)",
                allow_module_level=True,
            )
        raise
    finally:
        try:
            _probe_daemon.stop()
        except Exception:
            pass


@pytest.mark.asyncio
class TestConcurrentOperations:
    """Test concurrent operations with Turso/sqld."""

    @pytest_asyncio.fixture
    async def sqld_daemon(self):
        """Start sqld daemon for concurrent tests."""
        # Create a unique database path
        db_path = os.path.join(tempfile.gettempdir(), f"test_sqld_{uuid.uuid4().hex}.db")
        http_port = 9100 + int(uuid.uuid4().hex[:4], 16) % 500  # Random port 9100-9600

        daemon = SqldDaemon(db_path=db_path, http_port=http_port)
        daemon.start()

        # Wait for daemon to fully initialize
        await asyncio.sleep(2)

        # Initialize schema once before tests
        actual_db_file = os.path.join(db_path, "dbs", "default", "data")
        db_url = f"sqlite+aiosqlite:///{actual_db_file}"

        manager = NativeLibsqlTraceManager(db_url=db_url)
        await manager.initialize()
        await manager.close()

        yield daemon

        daemon.stop()
        # Wait a bit for daemon to fully stop before cleanup
        await asyncio.sleep(0.5)
        # Clean up the database file if it exists
        if os.path.exists(db_path):
            try:
                os.unlink(db_path)
            except PermissionError:
                pass  # File might still be locked on some systems

    @pytest_asyncio.fixture
    async def db_url(self, sqld_daemon):
        """Get the database URL for the running daemon."""
        # Return the actual SQLite file path inside sqld's directory structure
        # sqld creates: {db_path}/dbs/default/data
        actual_db_file = os.path.join(sqld_daemon.db_path, "dbs", "default", "data")
        return f"sqlite+aiosqlite:///{actual_db_file}"

    @pytest.mark.fast
    async def test_concurrent_session_insertion_no_race_condition(self, db_url):
        """
        Test that concurrent session insertions work without race conditions.

        This test should PASS with Turso's multi-writer support.
        """
        num_concurrent_sessions = 10

        async def create_and_insert_session(worker_id: int):
            """Create a session tracer and insert a session trace."""
            try:
                # Each worker creates its own tracer
                tracer = SessionTracer(db_url=db_url, auto_save=True)

                # Use unique session IDs
                session_id = f"concurrent_session_{worker_id}_{uuid.uuid4().hex[:8]}"

                # Start session and add some data
                await tracer.start_session(session_id)

                async with tracer.timestep(f"step_{worker_id}"):
                    # Add a message
                    await tracer.record_message(
                        content=f"Test message from worker {worker_id}", message_type="user"
                    )

                    # Add an event
                    event = RuntimeEvent(
                        system_instance_id=f"test_system_{worker_id}",
                        time_record=TimeRecord(event_time=time.time()),
                        actions=[worker_id],
                        metadata={"worker_id": worker_id},
                    )
                    await tracer.record_event(event)

                # End session (auto-saves to DB)
                await tracer.end_session()
                await tracer.close()

                return {"worker_id": worker_id, "success": True, "error": None}

            except Exception as e:
                import traceback

                return {
                    "worker_id": worker_id,
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }

        # Run concurrent tasks
        tasks = [create_and_insert_session(i) for i in range(num_concurrent_sessions)]
        results = await asyncio.gather(*tasks)

        # Check results
        failures = [r for r in results if not r["success"]]
        if failures:
            print("Failures:", failures)
            # Print first traceback for debugging
            if failures and "traceback" in failures[0]:
                print("\nFirst failure traceback:")
                print(failures[0]["traceback"])

        # All should succeed with Turso
        assert all(r["success"] for r in results), f"Some insertions failed: {failures}"

        # Verify all sessions were inserted
        manager = NativeLibsqlTraceManager(db_url=db_url)
        await manager.initialize()

        df = await manager.query_traces("SELECT COUNT(*) as count FROM session_traces")
        assert df.iloc[0]["count"] == num_concurrent_sessions

        await manager.close()

    @pytest.mark.fast
    async def test_async_concurrent_operations(self, db_url):
        """Test various concurrent async operations."""
        num_workers = 20

        async def worker_task(worker_id: int):
            """Perform various database operations."""
            tracer = SessionTracer(db_url=db_url, auto_save=True)

            # Create session
            session_id = f"async_session_{worker_id}"
            await tracer.start_session(session_id)

            # Add multiple timesteps
            for step in range(3):
                async with tracer.timestep(f"step_{step}"):
                    # Add different event types
                    if step % 3 == 0:
                        event = LMCAISEvent(
                            system_instance_id=f"llm_{worker_id}",
                            time_record=TimeRecord(event_time=time.time()),
                            model_name="gpt-4",
                            input_tokens=100,
                            output_tokens=50,
                            total_tokens=150,
                            cost_usd=0.003,
                        )
                    elif step % 3 == 1:
                        event = EnvironmentEvent(
                            system_instance_id=f"env_{worker_id}",
                            time_record=TimeRecord(event_time=time.time()),
                            reward=0.5,
                            terminated=False,
                        )
                    else:
                        event = RuntimeEvent(
                            system_instance_id=f"runtime_{worker_id}",
                            time_record=TimeRecord(event_time=time.time()),
                            actions=[step, worker_id],
                        )

                    await tracer.record_event(event)

                    # Add message
                    await tracer.record_message(
                        content=f"Message {step} from worker {worker_id}", message_type="assistant"
                    )

                    # Small delay to increase chance of contention
                    await asyncio.sleep(0.01)

            await tracer.end_session()
            await tracer.close()
            return worker_id

        # Run all workers concurrently
        tasks = [worker_task(i) for i in range(num_workers)]
        completed_ids = await asyncio.gather(*tasks)

        assert len(completed_ids) == num_workers
        assert set(completed_ids) == set(range(num_workers))

        # Verify data integrity
        manager = NativeLibsqlTraceManager(db_url=db_url)
        await manager.initialize()

        # Check session count
        df = await manager.query_traces("SELECT COUNT(*) as count FROM session_traces")
        assert df.iloc[0]["count"] == num_workers

        # Check event distribution
        df = await manager.query_traces(
            "SELECT event_type, COUNT(*) as count FROM events GROUP BY event_type"
        )

        # Each worker creates 3 events, one of each type
        for event_type in ["cais", "environment", "runtime"]:
            count = df[df["event_type"] == event_type]["count"].iloc[0]
            assert count == num_workers, f"Expected {num_workers} {event_type} events, got {count}"

        await manager.close()

    @pytest.mark.fast
    async def test_experiment_linking_concurrent(self, db_url):
        """Test concurrent experiment creation and session linking."""
        num_experiments = 5
        sessions_per_experiment = 10

        async def create_experiment_with_sessions(exp_num: int):
            """Create an experiment and link multiple sessions."""
            manager = NativeLibsqlTraceManager(db_url=db_url)
            await manager.initialize()

            # Create experiment
            exp_id = f"exp_{exp_num}_{uuid.uuid4().hex[:8]}"
            await manager.create_experiment(
                experiment_id=exp_id,
                name=f"Experiment {exp_num}",
                description=f"Test experiment {exp_num}",
            )

            # Create sessions for this experiment
            session_tasks = []

            async def create_session(session_num: int):
                tracer = SessionTracer(db_url=db_url, auto_save=True)
                session_id = f"exp{exp_num}_session{session_num}"

                await tracer.start_session(session_id)
                async with tracer.timestep("step1"):
                    await tracer.record_message(
                        content=f"Session {session_num} in experiment {exp_num}",
                        message_type="system",
                    )
                await tracer.end_session()
                await tracer.close()

                # Link to experiment
                await manager.link_session_to_experiment(session_id, exp_id)
                return session_id

            # Create all sessions for this experiment concurrently
            for s in range(sessions_per_experiment):
                session_tasks.append(create_session(s))

            session_ids = await asyncio.gather(*session_tasks)
            await manager.close()

            return exp_id, session_ids

        # Create all experiments concurrently
        exp_tasks = [create_experiment_with_sessions(i) for i in range(num_experiments)]
        exp_results = await asyncio.gather(*exp_tasks)

        # Verify all experiments and sessions were created
        manager = NativeLibsqlTraceManager(db_url=db_url)
        await manager.initialize()

        # Check experiment count
        df = await manager.query_traces("SELECT COUNT(*) as count FROM experiments")
        assert df.iloc[0]["count"] == num_experiments

        # Check total session count
        df = await manager.query_traces("SELECT COUNT(*) as count FROM session_traces")
        assert df.iloc[0]["count"] == num_experiments * sessions_per_experiment

        # Check that each experiment has correct number of sessions
        for exp_id, session_ids in exp_results:
            sessions = await manager.get_sessions_by_experiment(exp_id)
            assert len(sessions) == sessions_per_experiment

            # Verify session IDs match
            retrieved_ids = {s["session_id"] for s in sessions}
            expected_ids = set(session_ids)
            assert retrieved_ids == expected_ids

        await manager.close()

    @pytest.mark.fast
    async def test_high_concurrency_stress(self, db_url):
        """Stress test with high concurrency."""
        num_workers = 50
        operations_per_worker = 20

        async def stress_worker(worker_id: int):
            """Perform many rapid operations."""
            tracer = SessionTracer(db_url=db_url, auto_save=True)

            for op in range(operations_per_worker):
                session_id = f"stress_{worker_id}_{op}"
                await tracer.start_session(session_id)

                # Quick operations
                async with tracer.timestep("step"):
                    event = RuntimeEvent(
                        system_instance_id=f"stress_{worker_id}",
                        time_record=TimeRecord(event_time=time.time()),
                        actions=[op],
                    )
                    await tracer.record_event(event)

                await tracer.end_session()

                # No delay - maximum stress

            await tracer.close()
            return worker_id

        # Run stress test
        start_time = time.time()
        tasks = [stress_worker(i) for i in range(num_workers)]
        completed = await asyncio.gather(*tasks)
        duration = time.time() - start_time

        assert len(completed) == num_workers

        # Verify all operations completed
        manager = NativeLibsqlTraceManager(db_url=db_url)
        await manager.initialize()

        df = await manager.query_traces("SELECT COUNT(*) as count FROM session_traces")
        total_sessions = df.iloc[0]["count"]
        expected_sessions = num_workers * operations_per_worker

        assert total_sessions == expected_sessions, (
            f"Expected {expected_sessions} sessions, got {total_sessions}"
        )

        # Calculate throughput
        ops_per_second = total_sessions / duration
        print(f"\nStress test completed: {total_sessions} sessions in {duration:.2f}s")
        print(f"Throughput: {ops_per_second:.0f} sessions/second")

        await manager.close()

    @pytest.mark.fast
    async def test_duplicate_session_handling(self, db_url):
        """Test handling of duplicate session IDs (should use OR IGNORE)."""
        session_id = "duplicate_test_session"
        num_attempts = 10

        async def try_insert_session(attempt: int):
            """Try to insert a session with the same ID."""
            tracer = SessionTracer(db_url=db_url, auto_save=True)

            await tracer.start_session(session_id, metadata={"attempt": attempt})
            async with tracer.timestep(f"step_{attempt}"):
                await tracer.record_message(content=f"Attempt {attempt}", message_type="user")
            await tracer.end_session()
            await tracer.close()

            return attempt

        # Try to insert the same session ID multiple times
        tasks = [try_insert_session(i) for i in range(num_attempts)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Should not raise exceptions (uses OR IGNORE)
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Got exceptions: {exceptions}"

        # Verify only one session was inserted
        manager = NativeLibsqlTraceManager(db_url=db_url)
        await manager.initialize()

        df = await manager.query_traces(
            "SELECT COUNT(*) as count FROM session_traces WHERE session_id = :sid",
            {"sid": session_id},
        )
        assert df.iloc[0]["count"] == 1

        await manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
