#!/usr/bin/env python3
"""Debug script to understand why constraint violations are still happening."""

import tempfile
import os
import shutil
from datetime import datetime
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
from synth_ai.tracing_v2.session_tracer import SessionTrace, SessionTimeStep, SessionEventMessage

def create_realistic_session_trace(session_id: str):
    """Create a realistic session trace that mimics Crafter output."""
    trace = SessionTrace(session_id=session_id, created_at=datetime.now())
    
    # Add multiple timesteps like Crafter would
    timesteps = []
    for i in range(15):  # More than 10 to trigger bulk insert path
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

def test_constraint_violation_scenarios():
    """Test various scenarios that might cause constraint violations."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, 'test.duckdb')
    
    try:
        print("=== Testing Constraint Violation Scenarios ===")
        
        # Test 1: Basic duplicate session handling
        print("\n1. Testing basic duplicate session handling...")
        with DuckDBTraceManager(db_path) as db:
            session_id = "test_session_duplicate"
            trace = create_realistic_session_trace(session_id)
            
            print(f"   Inserting session {session_id} first time...")
            db.insert_session_trace(trace)
            print("   ✅ First insertion successful")
            
            print(f"   Inserting session {session_id} second time...")
            try:
                db.insert_session_trace(trace)
                print("   ✅ Second insertion handled gracefully")
            except Exception as e:
                print(f"   ❌ Second insertion failed: {e}")
        
        # Test 2: Manual transaction management (like in Crafter script)
        print("\n2. Testing manual transaction management...")
        with DuckDBTraceManager(db_path) as db:
            session_id = "test_session_manual_tx"
            trace = create_realistic_session_trace(session_id)
            
            # Simulate the Crafter script's transaction management
            print("   Starting manual transaction...")
            # Don't call db.conn.begin() as the script does it wrong
            
            print(f"   Inserting session {session_id} first time...")
            db.insert_session_trace(trace)
            print("   ✅ First insertion successful")
            
            print(f"   Inserting session {session_id} second time...")
            try:
                db.insert_session_trace(trace)
                print("   ✅ Second insertion handled gracefully")
            except Exception as e:
                print(f"   ❌ Second insertion failed: {e}")
        
        # Test 3: Check what happens with the exact schema query
        print("\n3. Testing direct schema INSERT query...")
        with DuckDBTraceManager(db_path) as db:
            session_id = "test_direct_insert"
            created_at = datetime.now()
            
            # First insert
            print("   Executing direct INSERT with ON CONFLICT first time...")
            cursor = db.conn.execute(
                """
                INSERT INTO session_traces (session_id, created_at, num_timesteps, num_events, num_messages, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                [session_id, created_at, 10, 0, 0, '{}']
            )
            print(f"   First insert rowcount: {cursor.rowcount}")
            
            # Second insert
            print("   Executing direct INSERT with ON CONFLICT second time...")
            try:
                cursor = db.conn.execute(
                    """
                    INSERT INTO session_traces (session_id, created_at, num_timesteps, num_events, num_messages, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO NOTHING
                    """,
                    [session_id, created_at, 10, 0, 0, '{}']
                )
                print(f"   Second insert rowcount: {cursor.rowcount}")
                print("   ✅ Direct INSERT with ON CONFLICT works correctly")
            except Exception as e:
                print(f"   ❌ Direct INSERT failed: {e}")
        
        # Test 4: Check if the issue is in bulk vs individual paths
        print("\n4. Testing bulk vs individual insert paths...")
        
        # Small trace (should use individual path)
        print("   Testing small trace (individual path)...")
        with DuckDBTraceManager(db_path) as db:
            session_id = "test_small_trace"
            trace = SessionTrace(session_id=session_id, created_at=datetime.now())
            trace.session_time_steps = [SessionTimeStep(step_id="step_1", timestamp=datetime.now(), events=[], step_messages=[])]
            trace.event_history = []
            trace.message_history = []
            
            try:
                db.insert_session_trace(trace)  # First time
                db.insert_session_trace(trace)  # Second time
                print("   ✅ Small trace duplicate handling works")
            except Exception as e:
                print(f"   ❌ Small trace failed: {e}")
        
        # Large trace (should use bulk path)
        print("   Testing large trace (bulk path)...")
        with DuckDBTraceManager(db_path) as db:
            session_id = "test_large_trace"
            trace = create_realistic_session_trace(session_id)  # 15 timesteps = bulk path
            
            try:
                db.insert_session_trace(trace)  # First time
                db.insert_session_trace(trace)  # Second time
                print("   ✅ Large trace duplicate handling works")
            except Exception as e:
                print(f"   ❌ Large trace failed: {e}")
                
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_constraint_violation_scenarios()