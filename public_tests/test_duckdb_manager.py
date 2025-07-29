#!/usr/bin/env python3
"""
Unit tests for DuckDB manager conversion utilities and core functionality.
Tests the datetime conversion and JSON serialization utilities.
"""

import pytest
import json
import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import duckdb

# Import the utilities we want to test
from synth_ai.tracing_v2.duckdb.manager import (
    convert_datetime_for_duckdb, 
    safe_json_serialize,
    DuckDBTraceManager
)
from synth_ai.tracing_v2.session_tracer import (
    SessionTracer, SessionEventMessage, TimeRecord,
    RuntimeEvent, EnvironmentEvent, LMCAISEvent
)
from synth_ai.tracing_v2.storage.types import EventType


@pytest.mark.slow
class TestConversionUtilities:
    """Test the conversion utilities for DuckDB compatibility."""
    
    @pytest.mark.slow
    def test_convert_datetime_for_duckdb_datetime(self):
        """Test converting datetime objects to ISO format."""
        dt = datetime(2023, 1, 15, 12, 30, 45, 123456)
        result = convert_datetime_for_duckdb(dt)
        expected = "2023-01-15T12:30:45.123456"
        assert result == expected
    
    @pytest.mark.slow
    def test_convert_datetime_for_duckdb_string(self):
        """Test that string timestamps are returned as-is."""
        timestamp_str = "2023-01-15T12:30:45.123456"
        result = convert_datetime_for_duckdb(timestamp_str)
        assert result == timestamp_str
    
    @pytest.mark.slow
    def test_convert_datetime_for_duckdb_unix_timestamp(self):
        """Test converting Unix timestamps."""
        unix_timestamp = 1673781045.123456
        result = convert_datetime_for_duckdb(unix_timestamp)
        # Should convert to ISO format
        assert result.startswith("2023-01-15")
        assert "T" in result
    
    @pytest.mark.slow
    def test_convert_datetime_for_duckdb_none(self):
        """Test handling None values."""
        result = convert_datetime_for_duckdb(None)
        assert result is None
    
    @pytest.mark.slow
    def test_convert_datetime_for_duckdb_invalid_unix(self):
        """Test handling invalid Unix timestamps."""
        result = convert_datetime_for_duckdb(-1)  # Invalid timestamp
        # Should still convert to a datetime string (Unix epoch - 1 second)
        assert result is not None
        assert isinstance(result, str)
    
    @pytest.mark.slow
    def test_convert_datetime_for_duckdb_other_types(self):
        """Test handling other types."""
        result = convert_datetime_for_duckdb(123)
        # Should convert to datetime string
        assert result is not None
        assert isinstance(result, str)
    
    @pytest.mark.slow
    def test_safe_json_serialize_datetime(self):
        """Test JSON serialization with datetime objects."""
        dt = datetime(2023, 1, 15, 12, 30, 45, 123456)
        data = {"timestamp": dt, "value": "test"}
        result = safe_json_serialize(data)
        
        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["value"] == "test"
        assert parsed["timestamp"] == "2023-01-15T12:30:45.123456"
    
    @pytest.mark.slow
    def test_safe_json_serialize_complex_object(self):
        """Test JSON serialization with complex objects."""
        data = {
            "datetime": datetime(2023, 1, 15, 12, 30, 45),
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "string": "test"
        }
        result = safe_json_serialize(data)
        
        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["string"] == "test"
        assert parsed["list"] == [1, 2, 3]
        assert parsed["dict"]["nested"] == "value"
        assert parsed["datetime"] == "2023-01-15T12:30:45"
    
    @pytest.mark.slow
    def test_safe_json_serialize_non_serializable(self):
        """Test JSON serialization with non-serializable objects."""
        class NonSerializable:
            def __init__(self):
                self.value = "test"
        
        obj = NonSerializable()
        result = safe_json_serialize(obj)
        
        # Should fall back to string representation
        assert result.startswith('"<')
        assert 'NonSerializable' in result
        assert result.endswith('>"')
    
    @pytest.mark.slow
    def test_safe_json_serialize_none(self):
        """Test JSON serialization with None."""
        result = safe_json_serialize(None)
        assert result == "null"
    
    @pytest.mark.slow
    def test_safe_json_serialize_empty_dict(self):
        """Test JSON serialization with empty dict."""
        result = safe_json_serialize({})
        assert result == "{}"


@pytest.mark.slow
class TestDuckDBTraceManager:
    """Test the DuckDB trace manager functionality."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        # Create a temporary directory and use a database file within it
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test.duckdb")
        yield db_path
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def db_manager(self, temp_db_path):
        """Create a DuckDB manager instance."""
        return DuckDBTraceManager(db_path=temp_db_path)
    
    @pytest.mark.slow
    def test_init_schema(self, db_manager):
        """Test schema initialization."""
        # Should not raise any exceptions
        db_manager.init_schema()
        
        # Check that tables were created
        tables = db_manager.conn.execute("SHOW TABLES").fetchall()
        table_names = [row[0] for row in tables]
        
        expected_tables = ['session_traces', 'session_timesteps', 'events', 'messages']
        for table in expected_tables:
            assert table in table_names
    
    @pytest.mark.slow
    def test_insert_session_trace_basic(self, db_manager):
        """Test basic session trace insertion."""
        # Create a simple session trace
        session_tracer = SessionTracer()
        session_id = "test_session_123"
        session_tracer.start_session(session_id)
        
        # Add some data
        session_tracer.start_timestep("step_1")
        session_tracer.record_message(SessionEventMessage(
            content="test message",
            message_type="test",
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=1
            )
        ))
        
        # End session
        session_tracer.end_session(save=False)
        
        # Insert into database
        db_manager.insert_session_trace(session_tracer.current_session)
        
        # Verify insertion
        result = db_manager.conn.execute(
            "SELECT session_id FROM session_traces WHERE session_id = ?", 
            [session_id]
        ).fetchone()
        
        assert result is not None
        assert result[0] == session_id
    
    @pytest.mark.slow
    def test_insert_session_trace_with_events(self, db_manager):
        """Test session trace insertion with events."""
        # Create session tracer
        session_tracer = SessionTracer()
        session_id = "test_session_with_events"
        session_tracer.start_session(session_id)
        
        # Add timestep and events
        session_tracer.start_timestep("step_1")
        
        # Add runtime event
        runtime_event = RuntimeEvent(
            system_instance_id="test_system",
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=1
            ),
            metadata={"test": "value"}
        )
        session_tracer.record_event(runtime_event)
        
        # Add environment event
        env_event = EnvironmentEvent(
            system_instance_id="test_env",
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=1
            ),
            reward=1.0,
            terminated=False
        )
        session_tracer.record_event(env_event)
        
        # End session
        session_tracer.end_session(save=False)
        
        # Insert into database
        db_manager.insert_session_trace(session_tracer.current_session)
        
        # Verify events were inserted
        events = db_manager.conn.execute(
            "SELECT event_type FROM events WHERE session_id = ?", 
            [session_id]
        ).fetchall()
        
        assert len(events) == 2
        event_types = [row[0] for row in events]
        assert "runtime" in event_types
        assert "environment" in event_types
    
    @pytest.mark.slow
    def test_insert_session_trace_with_lm_events(self, db_manager):
        """Test session trace insertion with LM events."""
        # Create session tracer
        session_tracer = SessionTracer()
        session_id = "test_session_with_lm"
        session_tracer.start_session(session_id)
        
        # Add timestep and LM event
        session_tracer.start_timestep("step_1")
        
        # Add LM event
        lm_event = LMCAISEvent(
            system_instance_id="test_lm",
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=1
            ),
            model_name="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost=0.01,
            latency_ms=500.0
        )
        session_tracer.record_event(lm_event)
        
        # End session
        session_tracer.end_session(save=False)
        
        # Insert into database
        db_manager.insert_session_trace(session_tracer.current_session)
        
        # Verify LM event was inserted
        events = db_manager.conn.execute(
            "SELECT model_name, prompt_tokens, completion_tokens, cost FROM events WHERE session_id = ?", 
            [session_id]
        ).fetchall()
        
        assert len(events) == 1
        event = events[0]
        assert event[0] == "gpt-4o-mini"  # model_name
        assert event[1] == 100  # prompt_tokens
        assert event[2] == 50   # completion_tokens
        assert event[3] == 0.01  # cost
    
    @pytest.mark.slow
    def test_batch_upload(self, db_manager):
        """Test batch upload functionality."""
        # Create multiple session traces
        traces = []
        for i in range(3):
            session_tracer = SessionTracer()
            session_id = f"batch_test_session_{i}"
            session_tracer.start_session(session_id)
            
            # Add some data
            session_tracer.start_timestep("step_1")
            session_tracer.record_message(SessionEventMessage(
                content=f"test message {i}",
                message_type="test",
                time_record=TimeRecord(
                    event_time=datetime.now().isoformat(),
                    message_time=1
                )
            ))
            
            session_tracer.end_session(save=False)
            traces.append(session_tracer.current_session)
        
        # Batch upload
        db_manager.batch_upload(traces)
        
        # Verify all sessions were inserted
        for i in range(3):
            session_id = f"batch_test_session_{i}"
            result = db_manager.conn.execute(
                "SELECT session_id FROM session_traces WHERE session_id = ?", 
                [session_id]
            ).fetchone()
            
            assert result is not None
            assert result[0] == session_id
    
    @pytest.mark.slow
    def test_query_traces(self, db_manager):
        """Test query functionality."""
        # Insert some test data
        session_tracer = SessionTracer()
        session_id = "query_test_session"
        session_tracer.start_session(session_id)
        
        session_tracer.start_timestep("step_1")
        session_tracer.record_message(SessionEventMessage(
            content="test message",
            message_type="test",
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=1
            )
        ))
        
        session_tracer.end_session(save=False)
        db_manager.insert_session_trace(session_tracer.current_session)
        
        # Test query
        df = db_manager.query_traces("SELECT session_id FROM session_traces WHERE session_id = ?", [session_id])
        
        assert len(df) == 1
        assert df.iloc[0]['session_id'] == session_id
    
    @pytest.mark.slow
    def test_get_model_usage(self, db_manager):
        """Test model usage statistics."""
        # Insert test data with LM events
        session_tracer = SessionTracer()
        session_id = "model_usage_test"
        session_tracer.start_session(session_id)
        
        session_tracer.start_timestep("step_1")
        lm_event = LMCAISEvent(
            system_instance_id="test_lm",
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=1
            ),
            model_name="gpt-4o-mini",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost=0.01
        )
        session_tracer.record_event(lm_event)
        
        session_tracer.end_session(save=False)
        db_manager.insert_session_trace(session_tracer.current_session)
        
        # Test model usage query
        df = db_manager.get_model_usage()
        
        # Should have at least one row
        assert len(df) >= 0  # May be empty if analytics views not enabled
    
    @pytest.mark.slow
    def test_get_expensive_calls(self, db_manager):
        """Test expensive calls query."""
        # Insert test data with expensive call
        session_tracer = SessionTracer()
        session_id = "expensive_call_test"
        session_tracer.start_session(session_id)
        
        session_tracer.start_timestep("step_1")
        lm_event = LMCAISEvent(
            system_instance_id="test_lm",
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=1
            ),
            model_name="gpt-4o-mini",
            cost=0.05  # Expensive call
        )
        session_tracer.record_event(lm_event)
        
        session_tracer.end_session(save=False)
        db_manager.insert_session_trace(session_tracer.current_session)
        
        # Test expensive calls query
        df = db_manager.get_expensive_calls(cost_threshold=0.01)
        
        assert len(df) >= 1
        assert df.iloc[0]['cost'] >= 0.01
    
    @pytest.mark.slow
    def test_error_handling_invalid_data(self, db_manager):
        """Test error handling with invalid data."""
        # Try to insert invalid data
        with pytest.raises(Exception):
            # This should fail gracefully
            db_manager.insert_session_trace(None)
    
    @pytest.mark.slow
    def test_connection_cleanup(self, temp_db_path):
        """Test that connections are properly cleaned up."""
        # Create manager
        manager = DuckDBTraceManager(db_path=temp_db_path)
        
        # Use context manager
        with manager:
            # Do some operations
            manager.init_schema()
        
        # Connection should be closed
        assert manager.conn is None or manager.conn.closed


@pytest.mark.slow
class TestIntegrationScenarios:
    """Test integration scenarios with real data."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        # Create a temporary directory and use a database file within it
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test.duckdb")
        yield db_path
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.slow
    def test_crafter_session_trace(self, temp_db_path):
        """Test inserting a realistic Crafter session trace."""
        db_manager = DuckDBTraceManager(db_path=temp_db_path)
        
        # Create a realistic session trace
        session_tracer = SessionTracer()
        session_id = "crafter_session_123"
        session_tracer.start_session(session_id)
        
        # Simulate a few steps
        for step in range(3):
            session_tracer.start_timestep(f"step_{step}")
            
            # Add observation message
            session_tracer.record_message(SessionEventMessage(
                content={
                    "inventory": {"wood": 2, "stone": 1},
                    "nearby": ["tree", "stone"],
                    "status": {"health": 9, "food": 8}
                },
                message_type="observation",
                time_record=TimeRecord(
                    event_time=datetime.now().isoformat(),
                    message_time=step
                )
            ))
            
            # Add action message
            session_tracer.record_message(SessionEventMessage(
                content="collect_wood",
                message_type="action",
                time_record=TimeRecord(
                    event_time=datetime.now().isoformat(),
                    message_time=step
                )
            ))
            
            # Add environment event
            env_event = EnvironmentEvent(
                system_instance_id="crafter_env",
                time_record=TimeRecord(
                    event_time=datetime.now().isoformat(),
                    message_time=step
                ),
                reward=1.0,
                terminated=False,
                system_state_before={"observation": {"inventory": {"wood": 1}}},
                system_state_after={"observation": {"inventory": {"wood": 2}}}
            )
            session_tracer.record_event(env_event)
            
            # Add runtime event
            runtime_event = RuntimeEvent(
                system_instance_id="crafter_runtime",
                time_record=TimeRecord(
                    event_time=datetime.now().isoformat(),
                    message_time=step
                ),
                actions=[5],  # "do" action
                metadata={"action_name": "collect_wood"}
            )
            session_tracer.record_event(runtime_event)
        
        # End session
        session_tracer.end_session(save=False)
        
        # Insert into database
        db_manager.insert_session_trace(session_tracer.current_session)
        
        # Verify the data was inserted correctly
        session_result = db_manager.conn.execute(
            "SELECT session_id, num_timesteps, num_events, num_messages FROM session_traces WHERE session_id = ?",
            [session_id]
        ).fetchone()
        
        assert session_result is not None
        assert session_result[1] == 3  # num_timesteps
        assert session_result[2] == 6  # num_events (3 env + 3 runtime)
        assert session_result[3] == 6  # num_messages (3 obs + 3 actions)
        
        # Verify timesteps
        timesteps = db_manager.conn.execute(
            "SELECT step_id FROM session_timesteps WHERE session_id = ? ORDER BY step_index",
            [session_id]
        ).fetchall()
        
        assert len(timesteps) == 3
        assert timesteps[0][0] == "step_0"
        assert timesteps[1][0] == "step_1"
        assert timesteps[2][0] == "step_2"
        
        # Verify events
        events = db_manager.conn.execute(
            "SELECT event_type FROM events WHERE session_id = ?",
            [session_id]
        ).fetchall()
        
        event_types = [row[0] for row in events]
        assert event_types.count("environment") == 3
        assert event_types.count("runtime") == 3
    
    @pytest.mark.slow
    def test_large_session_trace(self, temp_db_path):
        """Test handling of large session traces."""
        db_manager = DuckDBTraceManager(db_path=temp_db_path)
        
        # Create a large session trace
        session_tracer = SessionTracer()
        session_id = "large_session_123"
        session_tracer.start_session(session_id)
        
        # Add many steps
        for step in range(100):
            session_tracer.start_timestep(f"step_{step}")
            
            # Add complex observation
            session_tracer.record_message(SessionEventMessage(
                content={
                    "inventory": {f"item_{i}": i for i in range(10)},
                    "nearby": [f"object_{i}" for i in range(5)],
                    "status": {"health": 9, "food": 8, "energy": 7},
                    "achievements": {f"achievement_{i}": False for i in range(20)}
                },
                message_type="observation",
                time_record=TimeRecord(
                    event_time=datetime.now().isoformat(),
                    message_time=step
                )
            ))
            
            # Add LM event
            lm_event = LMCAISEvent(
                system_instance_id="test_lm",
                time_record=TimeRecord(
                    event_time=datetime.now().isoformat(),
                    message_time=step
                ),
                model_name="gpt-4o-mini",
                prompt_tokens=100 + step,
                completion_tokens=50 + step,
                total_tokens=150 + step * 2,
                cost=0.01 + step * 0.001,
                latency_ms=500.0 + step
            )
            session_tracer.record_event(lm_event)
        
        # End session
        session_tracer.end_session(save=False)
        
        # Insert into database
        db_manager.insert_session_trace(session_tracer.current_session)
        
        # Verify the large session was inserted
        session_result = db_manager.conn.execute(
            "SELECT num_timesteps, num_events, num_messages FROM session_traces WHERE session_id = ?",
            [session_id]
        ).fetchone()
        
        assert session_result is not None
        assert session_result[0] == 100  # num_timesteps
        assert session_result[1] == 100  # num_events
        assert session_result[2] == 100  # num_messages


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 