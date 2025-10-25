#!/usr/bin/env python3
"""
Unit tests for Turso/sqld manager conversion utilities and core functionality.
Async version of test_duckdb_manager.py for tracing v3.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime

import pytest
import pytest_asyncio

from ..abstractions import (
    EnvironmentEvent,
    LMCAISEvent,
    RuntimeEvent,
    TimeRecord,
)
from ..config import CONFIG
from ..db_config import DatabaseConfig, set_default_db_config
from ..session_tracer import SessionTracer

# Import the utilities and components we want to test
from ..turso.native_manager import NativeLibsqlTraceManager
from ..utils import calculate_cost, detect_provider, json_dumps


if shutil.which(CONFIG.sqld_binary) is None and shutil.which("libsql-server") is None:
    pytest.skip(
        "sqld binary not available; install Turso sqld or set SQLD_BINARY to skip these tests",
        allow_module_level=True,
    )

from ..turso.daemon import SqldDaemon

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


@pytest.fixture(autouse=True)
def _isolate_db_config(tmp_path, monkeypatch):
    """Ensure tests run against isolated SQLite config without sqld daemon."""
    monkeypatch.setenv("SYNTH_AI_V3_USE_SQLD", "false")
    monkeypatch.delenv("SQLD_HTTP_PORT", raising=False)
    db_path = tmp_path / "fixture.db"
    config = DatabaseConfig(db_path=str(db_path), use_sqld=False)
    set_default_db_config(config)
    yield
    config.stop_daemon()
    set_default_db_config(DatabaseConfig(use_sqld=False))


@pytest.mark.asyncio
class TestConversionUtilities:
    """Test the conversion utilities for Turso/SQLite compatibility."""

    async def test_json_dumps_datetime(self):
        """Test JSON serialization with datetime objects."""
        dt = datetime(2023, 1, 15, 12, 30, 45, 123456)
        data = {"timestamp": dt, "value": "test"}
        result = json_dumps(data)

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["value"] == "test"
        assert parsed["timestamp"] == "2023-01-15 12:30:45.123456"

    async def test_json_dumps_complex_object(self):
        """Test JSON serialization with complex objects."""
        data = {
            "datetime": datetime(2023, 1, 15, 12, 30, 45),
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "string": "test",
        }
        result = json_dumps(data)

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["string"] == "test"
        assert parsed["list"] == [1, 2, 3]
        assert parsed["dict"]["nested"] == "value"
        assert "2023-01-15" in parsed["datetime"]

    async def test_json_dumps_none(self):
        """Test JSON serialization with None."""
        result = json_dumps(None)
        assert result == "null"

    async def test_json_dumps_empty_dict(self):
        """Test JSON serialization with empty dict."""
        result = json_dumps({})
        assert result == "{}"

    async def test_detect_provider(self):
        """Test LLM provider detection."""
        assert detect_provider("gpt-4") == "openai"
        assert detect_provider("claude-3-opus") == "anthropic"
        assert detect_provider("gemini-pro") == "google"
        assert detect_provider("llama-2-70b") == "local"
        assert detect_provider("unknown-model") == "unknown"
        assert detect_provider(None) == "unknown"

    async def test_calculate_cost(self):
        """Test cost calculation."""
        # Test known model
        cost = calculate_cost("gpt-4", 1000, 500)
        assert cost is not None
        assert cost > 0

        # Test unknown model
        cost = calculate_cost("unknown-model", 1000, 500)
        assert cost is None


@pytest.mark.asyncio
class TestNativeTraceManager:
    """Test the native libsql trace manager functionality."""

    @pytest_asyncio.fixture
    async def sqld_daemon(self):
        """Use the centralized database configuration."""
        from ..db_config import get_default_db_config

        config = get_default_db_config()

        # If sqld is already running (via serve.sh), just use the existing database
        if not config.use_sqld:
            yield None  # No daemon to manage
        else:
            # Start a new daemon for isolated testing
            daemon, _ = config.get_daemon_and_url()
            yield daemon
            config.stop_daemon()

    @pytest_asyncio.fixture
    async def db_manager(self, sqld_daemon):
        """Create a NativeLibsqlTraceManager instance using centralized config."""

        from ..db_config import get_default_db_config

        config = get_default_db_config()
        db_url = config.database_url

        manager = NativeLibsqlTraceManager(db_url=db_url)
        await manager.initialize()

        yield manager
        await manager.close()

    @pytest.mark.fast
    async def test_init_schema(self, db_manager):
        """Test schema initialization."""
        # Schema should already be initialized by fixture
        # Try a simple query to verify
        df = await db_manager.query_traces("SELECT name FROM sqlite_master WHERE type='table'")

        table_names = df["name"].tolist()
        expected_tables = ["session_traces", "session_timesteps", "events", "messages"]

        for table in expected_tables:
            assert table in table_names

    @pytest.mark.fast
    async def test_insert_session_trace_basic(self, db_manager):
        """Test basic session trace insertion."""
        # Create a simple session trace
        session_tracer = SessionTracer(db_url="sqlite+aiosqlite:///:memory:", auto_save=False)
        session_id = await session_tracer.start_session()

        # Add some data
        await session_tracer.start_timestep("step_1")
        await session_tracer.record_message(
            content="test message", message_type="user", event_time=datetime.now().timestamp()
        )

        # End session
        trace = await session_tracer.end_session(save=False)

        # Insert into database
        await db_manager.insert_session_trace(trace)

        # Verify insertion
        df = await db_manager.query_traces(
            "SELECT session_id FROM session_traces WHERE session_id = :sid", {"sid": session_id}
        )

        assert len(df) == 1
        assert df.iloc[0]["session_id"] == session_id

    @pytest.mark.fast
    async def test_insert_session_trace_with_events(self, db_manager):
        """Test session trace insertion with events."""
        # Create session tracer
        session_tracer = SessionTracer(db_url="sqlite+aiosqlite:///:memory:", auto_save=False)
        session_id = await session_tracer.start_session()

        # Add timestep and events
        await session_tracer.start_timestep("step_1")

        # Add runtime event
        runtime_event = RuntimeEvent(
            system_instance_id="test_system",
            time_record=TimeRecord(event_time=datetime.now().timestamp(), message_time=1),
            actions=[1, 2, 3],
            metadata={"test": "value"},
        )
        await session_tracer.record_event(runtime_event)

        # Add environment event
        env_event = EnvironmentEvent(
            system_instance_id="test_env",
            time_record=TimeRecord(event_time=datetime.now().timestamp(), message_time=1),
            reward=1.0,
            terminated=False,
        )
        await session_tracer.record_event(env_event)

        # End session
        trace = await session_tracer.end_session(save=False)

        # Insert into database
        await db_manager.insert_session_trace(trace)

        # Verify events were inserted
        df = await db_manager.query_traces(
            "SELECT event_type FROM events WHERE session_id = :sid", {"sid": session_id}
        )

        assert len(df) == 2
        event_types = df["event_type"].tolist()
        assert "runtime" in event_types
        assert "environment" in event_types

    @pytest.mark.fast
    async def test_insert_session_trace_with_lm_events(self, db_manager):
        """Test session trace insertion with LM events."""
        # Create session tracer
        session_tracer = SessionTracer(db_url="sqlite+aiosqlite:///:memory:", auto_save=False)
        session_id = await session_tracer.start_session()

        # Add timestep and LM event
        await session_tracer.start_timestep("step_1")

        # Add LM event
        lm_event = LMCAISEvent(
            system_instance_id="test_lm",
            time_record=TimeRecord(event_time=datetime.now().timestamp(), message_time=1),
            model_name="gpt-4",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
            latency_ms=500,
        )
        await session_tracer.record_event(lm_event)

        # End session
        trace = await session_tracer.end_session(save=False)

        # Insert into database
        await db_manager.insert_session_trace(trace)

        # Verify LM event was inserted
        df = await db_manager.query_traces(
            "SELECT model_name, input_tokens, output_tokens, cost_usd FROM events WHERE session_id = :sid",
            {"sid": session_id},
        )

        assert len(df) == 1
        row = df.iloc[0]
        assert row["model_name"] == "gpt-4"
        assert row["input_tokens"] == 100
        assert row["output_tokens"] == 50
        assert row["cost_usd"] == 1  # Stored as cents (0.01 USD = 1 cent)

    @pytest.mark.fast
    async def test_batch_insert_sessions(self, db_manager):
        """Test batch insert functionality."""
        # Create multiple session traces
        traces = []
        session_ids = []

        for i in range(3):
            session_tracer = SessionTracer(db_url="sqlite+aiosqlite:///:memory:", auto_save=False)
            session_id = await session_tracer.start_session()
            session_ids.append(session_id)

            # Add some data
            await session_tracer.start_timestep("step_1")
            await session_tracer.record_message(content=f"test message {i}", message_type="user")

            trace = await session_tracer.end_session(save=False)
            traces.append(trace)

        # Batch insert
        inserted_ids = await db_manager.batch_insert_sessions(traces)

        # Verify all sessions were inserted
        for session_id in session_ids:
            df = await db_manager.query_traces(
                "SELECT session_id FROM session_traces WHERE session_id = :sid", {"sid": session_id}
            )
            assert len(df) == 1
            assert df.iloc[0]["session_id"] == session_id

    @pytest.mark.fast
    async def test_query_traces(self, db_manager):
        """Test query functionality."""
        # Insert some test data
        session_tracer = SessionTracer(db_url="sqlite+aiosqlite:///:memory:", auto_save=False)
        session_id = await session_tracer.start_session()

        await session_tracer.start_timestep("step_1")
        await session_tracer.record_message(content="test message", message_type="user")

        trace = await session_tracer.end_session(save=False)
        await db_manager.insert_session_trace(trace)

        # Test query
        df = await db_manager.query_traces(
            "SELECT session_id FROM session_traces WHERE session_id = :sid", {"sid": session_id}
        )

        assert len(df) == 1
        assert df.iloc[0]["session_id"] == session_id

    @pytest.mark.fast
    async def test_get_model_usage(self, db_manager):
        """Test model usage statistics."""
        # Insert test data with LM events
        session_tracer = SessionTracer(db_url="sqlite+aiosqlite:///:memory:", auto_save=False)
        session_id = await session_tracer.start_session()

        await session_tracer.start_timestep("step_1")
        lm_event = LMCAISEvent(
            system_instance_id="test_lm",
            time_record=TimeRecord(event_time=datetime.now().timestamp(), message_time=1),
            model_name="gpt-4",
            provider="openai",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
        )
        await session_tracer.record_event(lm_event)

        trace = await session_tracer.end_session(save=False)
        await db_manager.insert_session_trace(trace)

        # Test model usage query
        df = await db_manager.get_model_usage()

        # Should have at least one row
        assert len(df) >= 1
        # Check if our model is in the results
        gpt4_rows = df[df["model_name"] == "gpt-4"]
        if len(gpt4_rows) > 0:
            assert gpt4_rows.iloc[0]["total_input_tokens"] >= 100

    @pytest.mark.fast
    async def test_get_session_trace(self, db_manager):
        """Test retrieving a specific session."""
        # Insert test data
        session_tracer = SessionTracer(db_url="sqlite+aiosqlite:///:memory:", auto_save=False)
        session_id = await session_tracer.start_session(metadata={"test": "value"})

        await session_tracer.start_timestep("step_1")
        await session_tracer.record_message(content="test message", message_type="user")

        trace = await session_tracer.end_session(save=False)
        await db_manager.insert_session_trace(trace)

        # Retrieve session
        session_data = await db_manager.get_session_trace(session_id)

        assert session_data is not None
        assert session_data["session_id"] == session_id
        assert session_data["metadata"] == {"test": "value"}
        assert len(session_data["timesteps"]) == 1

    @pytest.mark.fast
    async def test_delete_session(self, db_manager):
        """Test session deletion."""
        # Insert test data
        session_tracer = SessionTracer(db_url="sqlite+aiosqlite:///:memory:", auto_save=False)
        session_id = await session_tracer.start_session()

        await session_tracer.start_timestep("step_1")
        trace = await session_tracer.end_session(save=False)
        await db_manager.insert_session_trace(trace)

        # Verify it exists
        session_data = await db_manager.get_session_trace(session_id)
        assert session_data is not None

        # Delete it
        deleted = await db_manager.delete_session(session_id)
        assert deleted is True

        # Verify it's gone
        session_data = await db_manager.get_session_trace(session_id)
        assert session_data is None

    @pytest.mark.fast
    async def test_experiment_management(self, db_manager):
        """Test experiment creation and linking."""
        # Use a unique experiment ID to avoid conflicts
        import uuid

        exp_id_suffix = str(uuid.uuid4())[:8]
        exp_id = f"exp_test_{exp_id_suffix}"

        # Create experiment
        created_id = await db_manager.create_experiment(
            experiment_id=exp_id,
            name="Test Experiment",
            description="Test description",
            configuration={"param1": "value1"},
        )
        assert created_id == exp_id

        # Create and link session
        session_tracer = SessionTracer(db_url="sqlite+aiosqlite:///:memory:", auto_save=False)
        session_id = await session_tracer.start_session()
        trace = await session_tracer.end_session(save=False)
        await db_manager.insert_session_trace(trace)

        await db_manager.link_session_to_experiment(session_id, exp_id)

        # Get sessions by experiment
        sessions = await db_manager.get_sessions_by_experiment(exp_id)
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == session_id


@pytest.mark.asyncio
class TestIntegrationScenarios:
    """Test integration scenarios with real data."""

    @pytest_asyncio.fixture
    async def sqld_daemon(self):
        """Use the centralized database configuration for integration tests."""
        from ..db_config import get_default_db_config

        config = get_default_db_config()

        # If sqld is already running (via serve.sh), just use the existing database
        if not config.use_sqld:
            yield None  # No daemon to manage
        else:
            # Start a new daemon for isolated testing
            daemon, _ = config.get_daemon_and_url()
            yield daemon
            config.stop_daemon()

    @pytest_asyncio.fixture
    async def db_manager(self, sqld_daemon):
        """Create a NativeLibsqlTraceManager instance using centralized config."""

        from ..db_config import get_default_db_config

        config = get_default_db_config()
        db_url = config.database_url

        manager = NativeLibsqlTraceManager(db_url=db_url)
        await manager.initialize()

        yield manager
        await manager.close()

    @pytest.mark.fast
    async def test_crafter_session_trace(self, db_manager):
        """Test inserting a realistic Crafter session trace."""
        # Create a realistic session trace
        session_tracer = SessionTracer(db_url="sqlite+aiosqlite:///:memory:", auto_save=False)
        session_id = await session_tracer.start_session()

        # Simulate a few steps
        for step in range(3):
            await session_tracer.start_timestep(f"step_{step}", turn_number=step)

            # Add observation message
            await session_tracer.record_message(
                content=json.dumps(
                    {
                        "inventory": {"wood": 2, "stone": 1},
                        "nearby": ["tree", "stone"],
                        "status": {"health": 9, "food": 8},
                    }
                ),
                message_type="system",
            )

            # Add action message
            await session_tracer.record_message(content="collect_wood", message_type="assistant")

            # Add environment event
            env_event = EnvironmentEvent(
                system_instance_id="crafter_env",
                time_record=TimeRecord(event_time=datetime.now().timestamp(), message_time=step),
                reward=1.0,
                terminated=False,
                system_state_before={"observation": {"inventory": {"wood": 1}}},
                system_state_after={"observation": {"inventory": {"wood": 2}}},
            )
            await session_tracer.record_event(env_event)

            # Add runtime event
            runtime_event = RuntimeEvent(
                system_instance_id="crafter_runtime",
                time_record=TimeRecord(event_time=datetime.now().timestamp(), message_time=step),
                actions=[5],  # "do" action
                metadata={"action_name": "collect_wood"},
            )
            await session_tracer.record_event(runtime_event)

        # End session
        trace = await session_tracer.end_session(save=False)

        # Insert into database
        await db_manager.insert_session_trace(trace)

        # Verify the data was inserted correctly
        session_df = await db_manager.query_traces(
            "SELECT session_id, num_timesteps, num_events, num_messages FROM session_traces WHERE session_id = :sid",
            {"sid": session_id},
        )

        assert len(session_df) == 1
        row = session_df.iloc[0]
        assert row["num_timesteps"] == 3
        assert row["num_events"] == 6  # 3 env + 3 runtime
        assert row["num_messages"] == 6  # 3 obs + 3 actions

        # Verify timesteps
        timesteps_df = await db_manager.query_traces(
            "SELECT step_id FROM session_timesteps WHERE session_id = :sid ORDER BY step_index",
            {"sid": session_id},
        )

        assert len(timesteps_df) == 3
        step_ids = timesteps_df["step_id"].tolist()
        assert step_ids == ["step_0", "step_1", "step_2"]

        # Verify events
        events_df = await db_manager.query_traces(
            "SELECT event_type FROM events WHERE session_id = :sid", {"sid": session_id}
        )

        event_types = events_df["event_type"].tolist()
        assert event_types.count("environment") == 3
        assert event_types.count("runtime") == 3

    @pytest.mark.fast
    async def test_large_session_trace(self, db_manager):
        """Test handling of large session traces."""
        # Create a large session trace
        session_tracer = SessionTracer(db_url="sqlite+aiosqlite:///:memory:", auto_save=False)
        session_id = await session_tracer.start_session()

        # Add many steps
        for step in range(100):
            await session_tracer.start_timestep(f"step_{step}")

            # Add complex observation
            await session_tracer.record_message(
                content=json.dumps(
                    {
                        "inventory": {f"item_{i}": i for i in range(10)},
                        "nearby": [f"object_{i}" for i in range(5)],
                        "status": {"health": 9, "food": 8, "energy": 7},
                        "achievements": {f"achievement_{i}": False for i in range(20)},
                    }
                ),
                message_type="system",
            )

            # Add LM event
            lm_event = LMCAISEvent(
                system_instance_id="test_lm",
                time_record=TimeRecord(event_time=datetime.now().timestamp(), message_time=step),
                model_name="gpt-4",
                input_tokens=100 + step,
                output_tokens=50 + step,
                total_tokens=150 + step * 2,
                cost_usd=0.01 + step * 0.001,
                latency_ms=500 + step,
            )
            await session_tracer.record_event(lm_event)

        # End session
        trace = await session_tracer.end_session(save=False)

        # Insert into database
        await db_manager.insert_session_trace(trace)

        # Verify the large session was inserted
        session_df = await db_manager.query_traces(
            "SELECT num_timesteps, num_events, num_messages FROM session_traces WHERE session_id = :sid",
            {"sid": session_id},
        )

        assert len(session_df) == 1
        row = session_df.iloc[0]
        assert row["num_timesteps"] == 100
        assert row["num_events"] == 100
        assert row["num_messages"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
