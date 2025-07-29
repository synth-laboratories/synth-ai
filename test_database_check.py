#!/usr/bin/env python3
"""Test the database checking logic from the Crafter script."""

from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager

def test_database_check():
    """Test database session counting."""
    db_path = "/Users/joshuapurtell/Documents/GitHub/synth-ai/synth_ai/traces/crafter_multi_model_traces.duckdb"
    
    print("Testing database session check...")
    
    # Test 1: Basic connection and count
    print("\n1. Basic connection test:")
    with DuckDBTraceManager(db_path) as db:
        result = db.conn.execute("SELECT COUNT(*) FROM session_traces").fetchone()
        print(f"   Session count: {result[0]}")
        
        if result[0] > 0:
            sample = db.conn.execute("SELECT session_id FROM session_traces LIMIT 3").fetchall()
            print("   Sample sessions:")
            for row in sample:
                print(f"     - {row[0]}")
    
    # Test 2: Mimic the exact script logic
    print("\n2. Mimicking script logic:")
    with DuckDBTraceManager(db_path) as db:
        existing_sessions = db.conn.execute("SELECT session_id FROM session_traces").fetchall()
        existing_ids = {row[0] for row in existing_sessions}
        print(f"   🔍 Database already contains {len(existing_ids)} sessions")
        
        # Show some examples
        if existing_ids:
            sample_ids = list(existing_ids)[:3]
            print("   Sample existing IDs:")
            for sid in sample_ids:
                print(f"     - {sid}")

if __name__ == "__main__":
    test_database_check()