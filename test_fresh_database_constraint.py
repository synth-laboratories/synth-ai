#!/usr/bin/env python3
"""Test constraint violations with fresh database to replicate the issue."""

import tempfile
import os
import shutil
from datetime import datetime
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
from synth_ai.tracing_v2.session_tracer import SessionTrace, SessionTimeStep, SessionEventMessage

def create_realistic_crafter_trace(session_id: str):
    """Create a trace that mimics what Crafter produces."""
    trace = SessionTrace(session_id=session_id, created_at=datetime.now())
    
    # Create 15+ timesteps to trigger bulk insert path
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

def test_fresh_database_constraint():
    """Test the exact scenario from the Crafter script."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, 'fresh_test.duckdb')
    
    try:
        print("=== Testing Fresh Database Constraint Issue ===")
        
        # Step 1: Create fresh database and verify it's empty
        print("\n1. Creating fresh database...")
        with DuckDBTraceManager(db_path) as db:
            result = db.conn.execute("SELECT COUNT(*) FROM session_traces").fetchone()
            print(f"   Database contains {result[0]} sessions")
        
        # Step 2: Try inserting sessions like the Crafter script does
        print("\n2. Testing session insertion...")
        
        # Simulate the _SESSIONS collection
        sessions = {}
        for i in range(2):
            session_id = f"episode_{i}_test_model_123456_{i}"
            trace = create_realistic_crafter_trace(session_id)
            sessions[session_id] = ("test_experiment", trace)
            print(f"   Created session: {session_id}")
        
        # Step 3: Mimic the bulk upload logic
        print("\n3. Mimicking bulk upload logic...")
        with DuckDBTraceManager(db_path) as db:
            # Check existing sessions (like the script does)
            existing_sessions = db.conn.execute("SELECT session_id FROM session_traces").fetchall()
            existing_ids = {row[0] for row in existing_sessions}
            print(f"   Database check shows {len(existing_ids)} existing sessions")
            
            # Try inserting each session
            uploaded_count = 0
            skipped_count = 0
            
            for session_id, (experiment_id, trace) in sessions.items():
                try:
                    print(f"   Inserting {session_id}...")
                    db.insert_session_trace(trace)
                    
                    # Update experiment_id like the script does
                    db.conn.execute(
                        "UPDATE session_traces SET experiment_id = ? "
                        "WHERE session_id = ? AND (experiment_id IS NULL OR experiment_id = '')",
                        [experiment_id, session_id]
                    )
                    uploaded_count += 1
                    print(f"   ✅ Successfully inserted {session_id}")
                    
                except Exception as e:
                    print(f"   ❌ Failed to insert {session_id}: {e}")
                    skipped_count += 1
            
            print(f"\n   Summary: {uploaded_count} uploaded, {skipped_count} skipped")
            
            # Verify final state
            final_count = db.conn.execute("SELECT COUNT(*) FROM session_traces").fetchone()[0]
            print(f"   Final database session count: {final_count}")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_fresh_database_constraint()