#!/usr/bin/env python3
"""
Test script to verify DuckDB filtering works correctly.
"""
import sys
from pathlib import Path

# Add synth_ai to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
from synth_ai.tracing_v2.duckdb.ft_utils import FinetuningDataExtractor


def test_duckdb_filter():
    """Test DuckDB filtering functionality."""
    db_path = "crafter_traces.duckdb"
    
    # Check if database exists
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        print("Please run test_crafter_react_agent_openai.py first to generate traces.")
        return
    
    print(f"✅ Found database: {db_path}")
    
    # Test basic connection
    try:
        with DuckDBTraceManager(db_path) as db:
            # Check tables
            tables = db.conn.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'main'
            """).df()
            print(f"\nTables in database: {list(tables['table_name'])}")
            
            # Count sessions
            sessions = db.conn.execute("SELECT COUNT(*) as count FROM session_traces").fetchone()
            print(f"Total sessions: {sessions[0]}")
            
            # Count events
            events = db.conn.execute("SELECT COUNT(*) as count FROM events").fetchone()
            print(f"Total events: {events[0]}")
            
            # Count messages
            messages = db.conn.execute("SELECT COUNT(*) as count FROM messages").fetchone()
            print(f"Total messages: {messages[0]}")
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return
    
    # Test filtering utilities
    print("\n" + "="*50)
    print("Testing FinetuningDataExtractor")
    print("="*50)
    
    try:
        with FinetuningDataExtractor(db_path) as extractor:
            # Get successful sessions
            successful = extractor.get_successful_sessions(min_reward=0.0)
            print(f"\nSuccessful sessions (reward > 0): {len(successful)}")
            
            if not successful.empty:
                print("\nTop 5 sessions by reward:")
                for idx, row in successful.head(5).iterrows():
                    print(f"  - {row['session_id']}: reward={row['total_reward']:.2f}")
                
                # Test conversation extraction
                test_session = successful.iloc[0]['session_id']
                print(f"\nTesting conversation extraction for: {test_session}")
                
                conversations = extractor.get_session_conversations(test_session)
                print(f"Found {len(conversations)} messages")
                
                if conversations:
                    print("\nFirst 3 messages:")
                    for i, conv in enumerate(conversations[:3]):
                        print(f"  [{conv['role']}]: {conv['content'][:100]}...")
                
                # Test OpenAI format extraction
                print("\nTesting OpenAI format extraction...")
                openai_data = extractor.extract_openai_format(
                    session_ids=[test_session],
                    min_reward=0.0
                )
                
                if openai_data:
                    print(f"✅ Successfully extracted {len(openai_data)} training examples")
                    example = openai_data[0]
                    print(f"   Messages in example: {len(example['messages'])}")
                    print(f"   Roles: {[m['role'] for m in example['messages']]}")
                else:
                    print("❌ No training data extracted")
                
                # Test metrics
                print("\nTesting session metrics...")
                metrics = extractor.get_session_metrics(test_session)
                print(f"Session metrics:")
                for key, value in metrics.items():
                    print(f"  - {key}: {value}")
                
    except Exception as e:
        print(f"❌ Error testing extractor: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    test_duckdb_filter()