#!/usr/bin/env python3
"""
Check where hook results are being stored in the database.
"""

import duckdb
import json

def check_hook_storage():
    conn = duckdb.connect("crafter_traces.duckdb")
    
    # Get recent experiment
    result = conn.execute("SELECT id, name FROM experiments ORDER BY created_at DESC LIMIT 1").fetchall()
    exp_id = result[0][0]
    exp_name = result[0][1]
    
    print(f"🔍 Checking hook storage for experiment: {exp_name} ({exp_id})")
    print("=" * 60)
    
    # Check session metadata
    print("\n📋 SESSION METADATA ANALYSIS:")
    result = conn.execute("SELECT session_id, metadata FROM session_traces WHERE experiment_id = ?", [exp_id]).fetchall()
    
    for row in result:
        session_id, metadata = row
        metadata_list = json.loads(metadata) if isinstance(metadata, str) else metadata
        
        print(f"\nSession: {session_id}")
        for i, item in enumerate(metadata_list):
            metadata_type = item.get('metadata_type', 'unknown')
            data = item.get('data', {})
            print(f"  Item {i}: {metadata_type}")
            print(f"    Keys: {list(data.keys())}")
            
            # Check for hook-related data
            if 'achievements' in data:
                achievements = data['achievements']
                unlocked = [k for k, v in achievements.items() if v]
                print(f"    Achievements: {unlocked}")
            
            if 'num_achievements' in data:
                print(f"    Num achievements: {data['num_achievements']}")
    
    # Check if there are any hook events
    print(f"\n🔍 HOOK EVENTS CHECK:")
    result = conn.execute("""
        SELECT COUNT(*) 
        FROM events e 
        JOIN session_traces st ON e.session_id = st.session_id 
        WHERE st.experiment_id = ? AND e.event_type = 'hook'
    """, [exp_id]).fetchall()
    
    hook_count = result[0][0]
    print(f"Hook events found: {hook_count}")
    
    if hook_count > 0:
        print("Sample hook events:")
        result = conn.execute("""
            SELECT e.session_id, e.metadata 
            FROM events e 
            JOIN session_traces st ON e.session_id = st.session_id 
            WHERE st.experiment_id = ? AND e.event_type = 'hook'
            LIMIT 3
        """, [exp_id]).fetchall()
        
        for row in result:
            session_id, metadata = row
            print(f"  Session {session_id}: {metadata}")
    
    # Check for any other hook-related data
    print(f"\n🔍 OTHER HOOK DATA:")
    result = conn.execute("""
        SELECT e.event_type, COUNT(*) 
        FROM events e 
        JOIN session_traces st ON e.session_id = st.session_id 
        WHERE st.experiment_id = ? 
        GROUP BY e.event_type
    """, [exp_id]).fetchall()
    
    print("Event types in experiment:")
    for event_type, count in result:
        print(f"  {event_type}: {count}")
    
    conn.close()

if __name__ == "__main__":
    check_hook_storage() 