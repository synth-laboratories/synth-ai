#!/usr/bin/env python3
"""Debug script to reproduce the exact constraint violation from rollout script."""

import tempfile
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add the synth_ai path
sys.path.insert(0, str(Path(__file__).parent))

from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
from synth_ai.tracing_v2.session_tracer import SessionTrace, SessionTimeStep, SessionEventMessage, RuntimeEvent, TimeRecord

def create_realistic_trace(session_id: str, num_steps: int = 5):
    """Create a trace that mimics what the rollout script generates."""
    trace = SessionTrace(session_id=session_id, created_at=datetime.now())
    
    # Create timesteps similar to how the rollout script does
    for i in range(num_steps):
        timestep = SessionTimeStep(
            step_id=f"turn_{i}",  # This is how the rollout script creates step IDs
            timestamp=datetime.now(),
            events=[],
            step_messages=[],
            step_metadata={"turn": i}
        )
        
        # Add some realistic events and messages like the rollout script
        event = RuntimeEvent(
            system_instance_id="crafter_agent",
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=i
            ),
            actions=[f"action_{i}"],
            metadata={"step": i}
        )
        timestep.events.append(event)
        
        message = SessionEventMessage(
            content=f"Step {i} content",
            message_type="agent_action",
            time_record=TimeRecord(
                event_time=datetime.now().isoformat(),
                message_time=i
            )
        )
        timestep.step_messages.append(message)
        
        trace.session_time_steps.append(timestep)
    
    # Add to global histories like SessionTracer does
    for timestep in trace.session_time_steps:
        for event in timestep.events:
            trace.event_history.append(event)
        for message in timestep.step_messages:
            trace.message_history.append(message)
    
    return trace

def test_multiple_similar_sessions():
    """Test inserting multiple sessions that might have similar structure."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, 'test_constraint.duckdb')
    
    try:
        print("=== Testing Multiple Session Insertion ===")
        
        # Create sessions with similar structure but unique IDs
        sessions = []
        for i in range(3):
            session_id = f"episode_{i}_gpt-4o-mini_12345_{i}"
            sessions.append((session_id, create_realistic_trace(session_id)))
        
        print(f"Created {len(sessions)} sessions to test")
        
        with DuckDBTraceManager(db_path) as db:
            for session_id, trace in sessions:
                try:
                    print(f"\\nInserting session: {session_id}")
                    print(f"  - {len(trace.session_time_steps)} timesteps")
                    print(f"  - {len(trace.event_history)} events")
                    print(f"  - {len(trace.message_history)} messages")
                    
                    # Show the step_ids that will be inserted
                    step_ids = [ts.step_id for ts in trace.session_time_steps]
                    print(f"  - step_ids: {step_ids}")
                    
                    db.insert_session_trace(trace)
                    print(f"✅ Successfully inserted {session_id}")
                    
                except Exception as e:
                    print(f"❌ Failed to insert {session_id}: {e}")
                    print(f"Error type: {type(e).__name__}")
                    
                    # Print detailed debug info
                    if "Duplicate key" in str(e):
                        print(f"🔍 Constraint violation details:")
                        print(f"   Error message: {str(e)}")
                        
                        # Check what's in the database
                        try:
                            sessions_in_db = db.conn.execute("SELECT session_id FROM session_traces").fetchall()
                            print(f"   Sessions in DB: {[s[0] for s in sessions_in_db]}")
                            
                            timesteps_in_db = db.conn.execute("SELECT session_id, step_id FROM session_timesteps").fetchall()
                            print(f"   Timesteps in DB: {timesteps_in_db}")
                        except Exception as check_e:
                            print(f"   Could not check DB state: {check_e}")
                    
                    import traceback
                    traceback.print_exc()
                    break
            
            # Final state check
            session_count = db.conn.execute("SELECT COUNT(*) FROM session_traces").fetchone()[0]
            timestep_count = db.conn.execute("SELECT COUNT(*) FROM session_timesteps").fetchone()[0]
            print(f"\\nFinal database state: {session_count} sessions, {timestep_count} timesteps")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

def test_session_with_duplicate_step_ids():
    """Test what happens if a session has duplicate step_ids."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, 'test_duplicate_steps.duckdb')
    
    try:
        print("\\n=== Testing Session with Duplicate Step IDs ===")
        
        session_id = "test_duplicate_steps"
        trace = SessionTrace(session_id=session_id, created_at=datetime.now())
        
        # Create timesteps with duplicate step_ids (this should cause constraint violation)
        for i in range(3):
            timestep = SessionTimeStep(
                step_id="turn_0",  # Same step_id for all timesteps - this should fail!
                timestamp=datetime.now(),
                events=[],
                step_messages=[],
                step_metadata={"turn": i}
            )
            trace.session_time_steps.append(timestep)
        
        print(f"Created session with {len(trace.session_time_steps)} timesteps")
        step_ids = [ts.step_id for ts in trace.session_time_steps]
        print(f"Step IDs: {step_ids} (should show duplicates)")
        
        with DuckDBTraceManager(db_path) as db:
            try:
                db.insert_session_trace(trace)
                print("✅ Insertion succeeded (unexpected)")
            except Exception as e:
                print(f"❌ Insertion failed as expected: {e}")
                if "Duplicate key" in str(e) and "step_id" in str(e):
                    print("✅ Confirmed: Duplicate step_id causes constraint violation")
                else:
                    print("❓ Unexpected error type")
                
    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_multiple_similar_sessions()
    test_session_with_duplicate_step_ids()