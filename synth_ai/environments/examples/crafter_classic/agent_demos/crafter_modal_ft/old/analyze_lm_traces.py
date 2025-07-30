#!/usr/bin/env python3
"""Analyze LM traces to see what the agent is actually doing."""

import duckdb
import json
from pathlib import Path
from collections import Counter, defaultdict

def analyze_lm_traces(db_path: str):
    """Analyze LM traces to understand agent behavior."""
    conn = duckdb.connect(db_path, read_only=True)
    
    print("🔍 Analyzing LM traces after fix...\n")
    
    # First, check if we have any LM events
    print("📊 CHECKING FOR LM EVENTS")
    print("=" * 50)
    
    lm_events = conn.execute("""
        SELECT COUNT(*) 
        FROM events 
        WHERE event_type = 'lm' OR event_type = 'generation'
    """).fetchone()[0]
    
    print(f"LM/Generation events found: {lm_events}")
    
    # Get all unique event types
    event_types = conn.execute("""
        SELECT DISTINCT event_type, COUNT(*) as count
        FROM events
        GROUP BY event_type
        ORDER BY count DESC
    """).fetchall()
    
    print("\nAll event types:")
    for event_type, count in event_types:
        print(f"  {event_type}: {count}")
    
    # Analyze sessions
    print("\n📊 SESSION ANALYSIS")
    print("=" * 50)
    
    sessions = conn.execute("""
        SELECT DISTINCT session_id
        FROM events
        ORDER BY session_id
        LIMIT 5
    """).fetchall()
    
    for i, (session_id,) in enumerate(sessions):
        print(f"\n--- Session {i+1}: {session_id} ---")
        
        # Get all events for this session
        events = conn.execute("""
            SELECT event_type, metadata, system_state_after
            FROM events
            WHERE session_id = ?
            ORDER BY id
        """, [session_id]).fetchall()
        
        print(f"Total events: {len(events)}")
        
        # Count event types
        event_type_counts = Counter()
        actions_taken = []
        achievements = set()
        
        for event_type, metadata_str, state_after_str in events:
            event_type_counts[event_type] += 1
            
            if metadata_str:
                metadata = json.loads(metadata_str)
                
                # Track actions
                if 'action_name' in metadata:
                    actions_taken.append(metadata['action_name'])
                elif 'action' in metadata:
                    actions_taken.append(metadata['action'])
            
            # Check for achievements
            if state_after_str:
                state_after = json.loads(state_after_str)
                if 'public_state' in state_after:
                    public_state = state_after['public_state']
                    if 'achievements_status' in public_state:
                        for ach, unlocked in public_state['achievements_status'].items():
                            if unlocked:
                                achievements.add(ach)
        
        print(f"Event type distribution: {dict(event_type_counts)}")
        print(f"Total actions: {len(actions_taken)}")
        if actions_taken:
            action_counts = Counter(actions_taken)
            print(f"Top actions: {action_counts.most_common(5)}")
        print(f"Achievements unlocked: {achievements if achievements else 'None'}")
    
    # Look for any LM-related metadata
    print("\n📊 LM METADATA ANALYSIS")
    print("=" * 50)
    
    # Check for model information in metadata
    model_events = conn.execute("""
        SELECT metadata
        FROM events
        WHERE metadata LIKE '%model%'
        LIMIT 10
    """).fetchall()
    
    if model_events:
        print(f"Found {len(model_events)} events with model metadata")
        for i, (metadata_str,) in enumerate(model_events[:3]):
            metadata = json.loads(metadata_str)
            print(f"\nEvent {i+1} metadata keys: {list(metadata.keys())}")
            if 'model' in metadata:
                print(f"  Model: {metadata['model']}")
    else:
        print("No events with model metadata found")
    
    # Check messages table
    print("\n📊 MESSAGES ANALYSIS")
    print("=" * 50)
    
    message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    print(f"Total messages: {message_count}")
    
    if message_count > 0:
        messages = conn.execute("""
            SELECT message_type, content
            FROM messages
            LIMIT 10
        """).fetchall()
        
        for i, (msg_type, content_str) in enumerate(messages[:5]):
            content = json.loads(content_str)
            print(f"\nMessage {i+1} ({msg_type}):")
            if isinstance(content, dict):
                print(f"  Keys: {list(content.keys())}")
                if 'payload' in content:
                    payload = content['payload']
                    if isinstance(payload, dict) and 'inventory' in payload:
                        # Show non-zero inventory items
                        inv = payload['inventory']
                        non_zero = {k: v for k, v in inv.items() if v > 0}
                        if non_zero:
                            print(f"  Non-zero inventory: {non_zero}")
    
    # Check for tool calls or function calls
    print("\n📊 TOOL CALL ANALYSIS")
    print("=" * 50)
    
    tool_events = conn.execute("""
        SELECT metadata
        FROM events
        WHERE metadata LIKE '%tool%' OR metadata LIKE '%function%'
        LIMIT 20
    """).fetchall()
    
    if tool_events:
        print(f"Found {len(tool_events)} events with tool/function mentions")
        tool_names = Counter()
        
        for (metadata_str,) in tool_events:
            metadata = json.loads(metadata_str)
            if 'tool_name' in metadata:
                tool_names[metadata['tool_name']] += 1
            elif 'function' in metadata:
                tool_names[metadata['function']] += 1
        
        if tool_names:
            print("Tool usage:")
            for tool, count in tool_names.most_common():
                print(f"  {tool}: {count}")
    else:
        print("No tool/function events found")
    
    conn.close()

if __name__ == "__main__":
    db_path = "./traces_v2_synth/traces.duckdb"
    if Path(db_path).exists():
        analyze_lm_traces(db_path)
    else:
        print(f"❌ Database not found at {db_path}")