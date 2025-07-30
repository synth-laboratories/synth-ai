#!/usr/bin/env python3
"""
Analyze how hooks are attached to events as metadata.
"""

import duckdb
import json

def analyze_hook_events(experiment_id: str):
    """Analyze how hooks are attached to events."""
    conn = duckdb.connect("crafter_traces.duckdb")
    
    print(f"🔍 HOOK EVENT ATTACHMENT ANALYSIS")
    print("=" * 80)
    print(f"Experiment ID: {experiment_id}")
    print()
    
    # Get events with hook metadata
    result = conn.execute("""
        SELECT e.session_id, e.event_type, e.event_metadata, e.metadata
        FROM events e 
        JOIN session_traces st ON e.session_id = st.session_id 
        WHERE st.experiment_id = ? AND e.event_metadata IS NOT NULL
        ORDER BY e.event_time
    """, [experiment_id]).fetchall()
    
    print(f"📊 Events with hook metadata: {len(result)}")
    print()
    
    hook_types = {
        'easy_achievement': 0,
        'medium_achievement': 0,
        'hard_achievement': 0,
        'invalid_action': 0,
        'inventory_increase': 0
    }
    
    for i, row in enumerate(result):
        session_id, event_type, event_metadata, metadata = row
        
        print(f"Event {i+1}:")
        print(f"  Session: {session_id}")
        print(f"  Type: {event_type}")
        print(f"  Base Metadata: {metadata}")
        print(f"  Hook Metadata: {event_metadata}")
        
        # Parse hook metadata
        if event_metadata:
            try:
                hook_data = json.loads(event_metadata) if isinstance(event_metadata, str) else event_metadata
                if isinstance(hook_data, list):
                    for hook in hook_data:
                        if isinstance(hook, str):
                            hook = json.loads(hook)
                        hook_name = hook.get('hook_name', 'unknown')
                        hook_types[hook_name] = hook_types.get(hook_name, 0) + 1
                        print(f"    Hook: {hook_name} - {hook.get('description', 'No description')}")
                else:
                    hook_name = hook_data.get('hook_name', 'unknown')
                    hook_types[hook_name] = hook_types.get(hook_name, 0) + 1
                    print(f"    Hook: {hook_name} - {hook_data.get('description', 'No description')}")
            except Exception as e:
                print(f"    Error parsing hook metadata: {e}")
        
        print()
    
    # Summary
    print("📈 HOOK SUMMARY")
    print("-" * 50)
    for hook_type, count in hook_types.items():
        if count > 0:
            print(f"  {hook_type}: {count} events")
    
    # Check event types that have hooks
    print(f"\n🔍 EVENT TYPES WITH HOOKS:")
    result = conn.execute("""
        SELECT e.event_type, COUNT(*) 
        FROM events e 
        JOIN session_traces st ON e.session_id = st.session_id 
        WHERE st.experiment_id = ? AND e.event_metadata IS NOT NULL
        GROUP BY e.event_type
    """, [experiment_id]).fetchall()
    
    for event_type, count in result:
        print(f"  {event_type}: {count} events with hooks")
    
    conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        experiment_id = sys.argv[1]
        analyze_hook_events(experiment_id)
    else:
        print("Usage: python analyze_hook_events.py <experiment_id>")
        print("Example: python analyze_hook_events.py 77022cce-4bda-4415-9bce-0095e4ef2237") 