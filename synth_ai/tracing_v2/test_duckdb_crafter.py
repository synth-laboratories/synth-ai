#!/usr/bin/env python3
"""
Quick test to verify DuckDB integration with tracing_v2
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synth_ai.tracing_v2.session_tracer import SessionTracer, SessionEventMessage, TimeRecord, CAISEvent
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
from synth_ai.tracing_v2.config import LOCAL_SYNTH, DUCKDB_CONFIG

def test_integration():
    """Test basic DuckDB integration."""
    print(f"LOCAL_SYNTH: {LOCAL_SYNTH}")
    print(f"DUCKDB_CONFIG: {DUCKDB_CONFIG}")
    
    # Create tracer with DuckDB
    tracer = SessionTracer(
        traces_dir="test_traces",
        duckdb_path="test_traces.duckdb"
    )
    
    # Start a test session
    session = tracer.start_session("test-session-001")
    print(f"Started session: {session.session_id}")
    
    # Add a timestep
    timestep = tracer.start_timestep("step-1")
    
    # Add a test message
    msg = SessionEventMessage(
        content={"text": "Test message"},
        message_type="test",
        time_record=TimeRecord(event_time="2024-01-01T00:00:00", message_time=0)
    )
    tracer.record_message(msg)
    
    # Add a test event
    event = CAISEvent(
        system_instance_id="test-system",
        model_name="gpt-3.5-turbo",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )
    tracer.record_event(event)
    
    # End session (should upload to DuckDB)
    print("Ending session...")
    tracer.end_session(save=True, upload_to_db=True)
    
    # Check DuckDB data
    print("\nChecking DuckDB data...")
    with DuckDBTraceManager("test_traces.duckdb") as db:
        sessions = db.conn.execute("SELECT * FROM session_traces").df()
        print(f"Sessions in DB: {len(sessions)}")
        
        events = db.conn.execute("SELECT * FROM events").df()
        print(f"Events in DB: {len(events)}")
        
        if not events.empty:
            print(f"First event type: {events.iloc[0]['event_type']}")
            print(f"Model name: {events.iloc[0]['model_name']}")
    
    print("\n✅ Integration test completed!")

if __name__ == "__main__":
    test_integration()