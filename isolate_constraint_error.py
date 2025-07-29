#!/usr/bin/env python3
"""Isolate exactly where the constraint violation is happening."""

import tempfile
import os
import shutil
from datetime import datetime
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
from synth_ai.tracing_v2.session_tracer import SessionTrace, SessionTimeStep
import traceback

def create_simple_trace(session_id: str):
    """Create a trace with multiple timesteps to trigger the issue."""
    trace = SessionTrace(session_id=session_id, created_at=datetime.now())
    
    # Create multiple timesteps with the same session_id but different step_ids
    timesteps = []
    for i in range(5):  # Just 5 to keep it simple
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

def isolate_constraint_error():
    """Isolate exactly where the constraint violation happens."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, 'isolate.duckdb')
    
    try:
        print("=== Isolating Constraint Violation ===")
        
        session_id = "test_constraint_isolation"
        trace = create_simple_trace(session_id)
        
        print(f"\\nTesting insertion of session with ID: {session_id}")
        print(f"Session has {len(trace.session_time_steps)} timesteps")
        
        with DuckDBTraceManager(db_path) as db:
            try:
                print("\\nAttempting insertion...")
                db.insert_session_trace(trace)
                print("✅ Insertion completed successfully")
                
                # Check what was actually inserted
                session_count = db.conn.execute("SELECT COUNT(*) FROM session_traces").fetchone()[0]
                timestep_count = db.conn.execute("SELECT COUNT(*) FROM session_timesteps").fetchone()[0]
                print(f"Database contains {session_count} sessions and {timestep_count} timesteps")
                
            except Exception as e:
                print(f"❌ Insertion failed with error: {e}")
                print(f"Error type: {type(e).__name__}")
                
                # Print the full stack trace to see exactly where it fails
                print("\\nFull stack trace:")
                traceback.print_exc()
                
                # Check what was partially inserted
                try:
                    session_count = db.conn.execute("SELECT COUNT(*) FROM session_traces").fetchone()[0]
                    timestep_count = db.conn.execute("SELECT COUNT(*) FROM session_timesteps").fetchone()[0]
                    print(f"\\nPartial insertion - {session_count} sessions and {timestep_count} timesteps")
                except:
                    print("\\nCould not check partial insertion state")
                
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
    finally:
        # Clean up
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    isolate_constraint_error()