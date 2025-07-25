#!/usr/bin/env python3
"""
Minimal test to verify DuckDB integration imports work correctly.
"""
import sys
from pathlib import Path

# Add synth_ai to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

try:
    print("Testing imports...")
    
    # Test basic imports
    from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
    print("✅ DuckDBTraceManager imported successfully")
    
    from synth_ai.tracing_v2.duckdb.ft_utils import FinetuningDataExtractor
    print("✅ FinetuningDataExtractor imported successfully")
    
    from synth_ai.tracing_v2.session_tracer import SessionTracer
    print("✅ SessionTracer imported successfully")
    
    # Test creating a temporary database
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".duckdb") as tmp:
        print(f"\nCreating test database: {tmp.name}")
        
        # Test DuckDBTraceManager
        with DuckDBTraceManager(tmp.name) as db:
            # Check schema was created
            tables = db.conn.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'main'
            """).df()
            print(f"✅ Schema created with tables: {list(tables['table_name'])}")
        
        # Test FinetuningDataExtractor
        with FinetuningDataExtractor(tmp.name) as extractor:
            sessions = extractor.get_successful_sessions()
            print(f"✅ FinetuningDataExtractor works (found {len(sessions)} sessions)")
    
    print("\n✅ All imports and basic functionality verified!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)