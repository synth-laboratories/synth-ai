#!/usr/bin/env python3
"""Check DuckDB schema to understand table structure."""

import duckdb
from pathlib import Path

def check_schema(db_path: str):
    """Check database schema."""
    conn = duckdb.connect(db_path, read_only=True)
    
    print("🔍 Checking database schema...\n")
    
    # Get all tables
    tables = conn.execute("SHOW TABLES").fetchall()
    print("📋 Tables in database:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # Check schema of key tables
    key_tables = ['session_traces', 'events', 'messages', 'session_timesteps']
    
    for table_name in key_tables:
        if any(t[0] == table_name for t in tables):
            print(f"\n📊 Schema for {table_name}:")
            print("-" * 50)
            schema = conn.execute(f"DESCRIBE {table_name}").fetchall()
            for col_name, col_type, _, _, _, _ in schema:
                print(f"  {col_name}: {col_type}")
            
            # Show sample data
            print(f"\n📄 Sample data from {table_name} (first 2 rows):")
            sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 2").fetchall()
            if sample:
                # Get column names
                cols = [desc[0] for desc in conn.execute(f"SELECT * FROM {table_name} LIMIT 0").description]
                print(f"  Columns: {cols}")
                for row in sample:
                    print(f"  {row}")
            else:
                print("  (No data)")
    
    conn.close()

if __name__ == "__main__":
    db_path = "./traces_v2_synth/traces.duckdb"
    if Path(db_path).exists():
        check_schema(db_path)
    else:
        print(f"❌ Database not found at {db_path}")