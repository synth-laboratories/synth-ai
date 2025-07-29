#!/usr/bin/env python3
"""Test to identify the exact exception type for DuckDB constraint violations."""

import tempfile
import os
import shutil
from datetime import datetime
from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
import duckdb

def test_exception_type():
    """Test what exception type is thrown for constraint violations."""
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, 'exception_test.duckdb')
    
    try:
        print("=== Testing DuckDB Constraint Violation Exception Type ===")
        
        with DuckDBTraceManager(db_path) as db:
            session_id = "test_constraint_exception"
            created_at = datetime.now()
            
            # First insert - should succeed
            print("\n1. First insert (should succeed)...")
            try:
                cursor = db.conn.execute(
                    """
                    INSERT INTO session_traces (session_id, created_at, num_timesteps, num_events, num_messages, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [session_id, created_at, 10, 0, 0, '{}']
                )
                print(f"   ✅ First insert succeeded, rowcount: {cursor.rowcount}")
            except Exception as e:
                print(f"   ❌ First insert failed: {type(e).__name__}: {e}")
            
            # Second insert - should fail with constraint violation
            print("\n2. Second insert without ON CONFLICT (should fail)...")
            try:
                cursor = db.conn.execute(
                    """
                    INSERT INTO session_traces (session_id, created_at, num_timesteps, num_events, num_messages, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [session_id, created_at, 10, 0, 0, '{}']
                )
                print(f"   ❌ Second insert unexpectedly succeeded")
            except Exception as e:
                print(f"   ✅ Second insert failed as expected")
                print(f"   Exception type: {type(e).__name__}")
                print(f"   Exception message: {e}")
                print(f"   Exception module: {type(e).__module__}")
            
            # Third insert with ON CONFLICT - behavior depends on DuckDB version
            print("\n3. Third insert with ON CONFLICT DO NOTHING...")
            try:
                cursor = db.conn.execute(
                    """
                    INSERT INTO session_traces (session_id, created_at, num_timesteps, num_events, num_messages, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO NOTHING
                    """,
                    [session_id, created_at, 10, 0, 0, '{}']
                )
                print(f"   ✅ ON CONFLICT insert succeeded, rowcount: {cursor.rowcount}")
            except Exception as e:
                print(f"   ❌ ON CONFLICT insert failed")
                print(f"   Exception type: {type(e).__name__}")  
                print(f"   Exception message: {e}")
                print(f"   Is it a DuckDB exception? {isinstance(e, duckdb.Error)}")
                if hasattr(e, 'args'):
                    print(f"   Exception args: {e.args}")
                
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    test_exception_type()