#!/usr/bin/env python3
"""Test script to verify duplicate session handling works."""

import tempfile
import os
from datetime import datetime
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
from synth_ai.tracing_v2.session_tracer import SessionTrace, SessionTimeStep, SessionEventMessage

def create_test_session_trace(session_id: str):
    """Create a minimal test session trace."""
    trace = SessionTrace(session_id=session_id, created_at=datetime.now())
    
    # Add a timestep
    timestep = SessionTimeStep(
        step_id="test_step_1",
        timestamp=datetime.now(),
        events=[],
        step_messages=[]
    )
    trace.session_time_steps = [timestep]
    trace.event_history = []
    trace.message_history = []
    
    return trace

def test_duplicate_session_handling():
    """Test that duplicate sessions are handled properly."""
    # Create a temporary database
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, 'test.duckdb')
    
    try:
        # Create the database and insert a session
        with DuckDBTraceManager(db_path) as db:
            session_id = "test_session_123"
            trace = create_test_session_trace(session_id)
            
            print(f"Inserting session {session_id} first time...")
            db.insert_session_trace(trace)
            print("✅ First insertion successful")
            
            print(f"Inserting session {session_id} second time (should be handled gracefully)...")
            db.insert_session_trace(trace)
            print("✅ Second insertion handled gracefully")
            
            # Verify only one session exists
            result = db.conn.execute("SELECT COUNT(*) FROM session_traces WHERE session_id = ?", [session_id]).fetchone()
            count = result[0]
            print(f"Sessions in database: {count}")
            
            if count == 1:
                print("✅ Duplicate handling working correctly!")
            else:
                print(f"❌ Expected 1 session, found {count}")
                
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        import shutil
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_duplicate_session_handling()