#!/usr/bin/env python3
"""Analyze trace data to understand why no achievements were unlocked."""

import duckdb
import json
from pathlib import Path
from collections import defaultdict, Counter

def analyze_traces(db_path: str):
    """Analyze trace data to identify issues."""
    conn = duckdb.connect(db_path, read_only=True)
    
    print("🔍 Analyzing trace data...\n")
    
    # 1. Check basic statistics
    print("📊 BASIC STATISTICS")
    print("=" * 50)
    
    # First, show available tables
    tables = conn.execute("SHOW TABLES").fetchall()
    print("Available tables:")
    for table in tables:
        print(f"  - {table[0]}")
    
    # Count sessions
    session_count = conn.execute("SELECT COUNT(DISTINCT session_id) FROM session_traces").fetchone()[0]
    print(f"\nTotal sessions: {session_count}")
    
    # Count events
    event_count = conn.execute("SELECT COUNT(*) FROM session_traces").fetchone()[0]
    print(f"Total events: {event_count}")
    
    # Check event types
    print("\n📋 EVENT TYPE DISTRIBUTION")
    print("-" * 30)
    event_types = conn.execute("""
        SELECT event_type, COUNT(*) as count 
        FROM session_traces 
        GROUP BY event_type 
        ORDER BY count DESC
    """).fetchall()
    
    for event_type, count in event_types:
        print(f"{event_type}: {count}")
    
    # 2. Analyze agent decisions
    print("\n🤖 AGENT DECISIONS ANALYSIS")
    print("=" * 50)
    
    # Get all generation completion events
    completions = conn.execute("""
        SELECT event_data 
        FROM session_traces 
        WHERE event_type = 'generation_completion'
        LIMIT 50
    """).fetchall()
    
    if completions:
        print(f"Found {len(completions)} generation completions (showing first 50)")
        
        # Analyze first few completions
        for i, (event_data,) in enumerate(completions[:5]):
            data = json.loads(event_data)
            print(f"\n--- Completion {i+1} ---")
            
            # Extract response
            if 'response' in data and data['response']:
                response = data['response']
                if 'content' in response:
                    print(f"Content preview: {response['content'][:200]}...")
                if 'tool_calls' in response:
                    print(f"Tool calls: {response['tool_calls']}")
            else:
                print("No response found in event data")
    else:
        print("❌ No generation completion events found!")
    
    # 3. Analyze runtime events (actions taken)
    print("\n🎮 RUNTIME EVENTS (ACTIONS)")
    print("=" * 50)
    
    runtime_events = conn.execute("""
        SELECT event_data 
        FROM session_traces 
        WHERE event_type = 'runtime_event'
        LIMIT 100
    """).fetchall()
    
    if runtime_events:
        action_counter = Counter()
        
        for (event_data,) in runtime_events:
            data = json.loads(event_data)
            if 'metadata' in data and 'action_name' in data['metadata']:
                action_counter[data['metadata']['action_name']] += 1
        
        print(f"Found {len(runtime_events)} runtime events")
        print("\nAction distribution:")
        for action, count in action_counter.most_common():
            print(f"  {action}: {count}")
    else:
        print("❌ No runtime events found!")
    
    # 4. Analyze environment events (results)
    print("\n🌍 ENVIRONMENT EVENTS")
    print("=" * 50)
    
    env_events = conn.execute("""
        SELECT event_data 
        FROM session_traces 
        WHERE event_type = 'environment_event'
        LIMIT 100
    """).fetchall()
    
    if env_events:
        reward_sum = 0
        achievements_found = []
        
        for (event_data,) in env_events:
            data = json.loads(event_data)
            
            # Check rewards
            if 'reward' in data:
                reward_sum += data['reward'] or 0
            
            # Check for achievements in state
            if 'system_state_after' in data:
                state = data['system_state_after']
                if 'public_state' in state and 'achievements' in state['public_state']:
                    achievements = state['public_state']['achievements']
                    for ach, unlocked in achievements.items():
                        if unlocked:
                            achievements_found.append(ach)
        
        print(f"Found {len(env_events)} environment events")
        print(f"Total reward across all events: {reward_sum}")
        print(f"Achievements found: {set(achievements_found) if achievements_found else 'None'}")
    else:
        print("❌ No environment events found!")
    
    # 5. Check for errors
    print("\n⚠️ ERROR CHECK")
    print("=" * 50)
    
    # Look for error messages in events
    error_events = conn.execute("""
        SELECT event_type, event_data 
        FROM session_traces 
        WHERE event_data LIKE '%error%' OR event_data LIKE '%Error%'
        LIMIT 10
    """).fetchall()
    
    if error_events:
        print(f"Found {len(error_events)} events with potential errors:")
        for event_type, event_data in error_events[:3]:
            print(f"\n{event_type}:")
            data = json.loads(event_data)
            print(json.dumps(data, indent=2)[:500])
    else:
        print("No obvious errors found in events")
    
    # 6. Sample a full episode flow
    print("\n📖 SAMPLE EPISODE FLOW")
    print("=" * 50)
    
    # Get events from first session
    first_session = conn.execute("SELECT DISTINCT session_id FROM session_traces LIMIT 1").fetchone()
    if first_session:
        session_id = first_session[0]
        print(f"Analyzing session: {session_id}")
        
        session_events = conn.execute("""
            SELECT event_type, event_data, created_at
            FROM session_traces 
            WHERE session_id = ?
            ORDER BY created_at
            LIMIT 20
        """, [session_id]).fetchall()
        
        print(f"\nFirst 20 events in session:")
        for i, (event_type, event_data, created_at) in enumerate(session_events):
            data = json.loads(event_data)
            print(f"\n{i+1}. {event_type} at {created_at}")
            
            # Show relevant info based on event type
            if event_type == 'generation_completion':
                if 'response' in data and 'tool_calls' in data['response']:
                    print(f"   Tool calls: {data['response']['tool_calls']}")
            elif event_type == 'runtime_event':
                if 'metadata' in data:
                    print(f"   Action: {data['metadata'].get('action_name', 'Unknown')}")
            elif event_type == 'environment_event':
                if 'reward' in data:
                    print(f"   Reward: {data['reward']}")
    
    conn.close()

if __name__ == "__main__":
    db_path = "./traces_v2_synth/traces.duckdb"
    if Path(db_path).exists():
        analyze_traces(db_path)
    else:
        print(f"❌ Database not found at {db_path}")
        print("Available databases:")
        for db in Path(".").glob("**/traces.duckdb"):
            print(f"  - {db}")