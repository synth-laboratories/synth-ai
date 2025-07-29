#!/usr/bin/env python3
"""Debug the exact session insertion flow to find where duplicates come from."""

import tempfile
import os
import shutil
from datetime import datetime
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
from synth_ai.tracing_v2.session_tracer import SessionTrace, SessionTimeStep

def create_test_trace(session_id: str):
    """Create a test trace."""
    trace = SessionTrace(session_id=session_id, created_at=datetime.now())
    
    # Create enough timesteps to trigger bulk insert (>10)
    timesteps = []
    for i in range(15):
        timestep = SessionTimeStep(
            step_id=f"turn_{i}",
            timestamp=datetime.now(),
            events=[],
            step_messages=[]
        )
        timesteps.append(timestep)
    
    trace.session_time_steps = timesteps
    trace.event_history = []
    trace.message_history = []
    
    return trace

def debug_session_insertion():
    """Debug where duplicate sessions are coming from."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, 'debug.duckdb')
    
    try:
        print("=== Debugging Session Insertion Flow ===")
        
        session_id = "debug_session_test"
        trace = create_test_trace(session_id)
        
        print(f"\n1. Testing insertion of session: {session_id}")
        
        with DuckDBTraceManager(db_path) as db:
            # Check initial state
            count = db.conn.execute("SELECT COUNT(*) FROM session_traces").fetchone()[0]
            print(f"   Initial database count: {count}")
            
            # First insertion
            print(f"   Attempting first insertion...")
            try:
                db.insert_session_trace(trace)
                print(f"   ✅ First insertion succeeded")
            except Exception as e:
                print(f"   ❌ First insertion failed: {e}")
            
            # Check state after first insertion
            count = db.conn.execute("SELECT COUNT(*) FROM session_traces").fetchone()[0]
            print(f"   Database count after first insertion: {count}")
            
            # Check if session exists
            existing = db.conn.execute(
                "SELECT session_id FROM session_traces WHERE session_id = ?", 
                [session_id]
            ).fetchone()
            print(f"   Session exists check: {'Yes' if existing else 'No'}")
            
            # Second insertion - should be skipped
            print(f"   Attempting second insertion...")
            try:
                db.insert_session_trace(trace)
                print(f"   ✅ Second insertion succeeded (should have been skipped)")
            except Exception as e:
                print(f"   ❌ Second insertion failed: {e}")
            
            # Final state
            count = db.conn.execute("SELECT COUNT(*) FROM session_traces").fetchone()[0]
            print(f"   Final database count: {count}")
            
        print("\n2. Testing with different DuckDBTraceManager instances...")
        
        # Try with a new DuckDBTraceManager instance to simulate script behavior
        with DuckDBTraceManager(db_path) as db2:
            # Check what this connection sees
            count = db2.conn.execute("SELECT COUNT(*) FROM session_traces").fetchone()[0]
            print(f"   Second connection sees {count} sessions")
            
            existing = db2.conn.execute(
                "SELECT session_id FROM session_traces WHERE session_id = ?", 
                [session_id]
            ).fetchone()
            print(f"   Second connection existence check: {'Yes' if existing else 'No'}")
            
            # Try inserting again
            print(f"   Attempting insertion from second connection...")
            try:
                db2.insert_session_trace(trace)
                print(f"   ✅ Second connection insertion succeeded")
            except Exception as e:
                print(f"   ❌ Second connection insertion failed: {e}")
            
            # Final count
            count = db2.conn.execute("SELECT COUNT(*) FROM session_traces").fetchone()[0]
            print(f"   Final count from second connection: {count}")
                
    except Exception as e:
        print(f"❌ Debug failed with exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    debug_session_insertion()