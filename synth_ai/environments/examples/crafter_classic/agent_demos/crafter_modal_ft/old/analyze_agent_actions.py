#!/usr/bin/env python3
"""Analyze agent actions and model responses from trace data."""

import duckdb
import json
from pathlib import Path
from collections import Counter

def analyze_agent_actions(db_path: str):
    """Analyze agent actions and responses."""
    conn = duckdb.connect(db_path, read_only=True)
    
    print("🔍 Analyzing agent actions and responses...\n")
    
    # Get all events
    all_events = conn.execute("""
        SELECT 
            e.session_id,
            e.event_type,
            e.event_time,
            e.metadata,
            e.system_state_after,
            e.system_state_before,
            e.reward,
            e.terminated
        FROM events e
        ORDER BY e.session_id, e.event_time
    """).fetchall()
    
    print(f"Total events found: {len(all_events)}\n")
    
    # Group events by session
    sessions = {}
    for event in all_events:
        session_id = event[0]
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(event)
    
    print(f"Total sessions: {len(sessions)}\n")
    
    # Analyze each session
    for i, (session_id, events) in enumerate(sessions.items()):
        if i >= 3:  # Only analyze first 3 sessions
            break
            
        print(f"\n{'='*60}")
        print(f"SESSION {i+1}: {session_id}")
        print(f"{'='*60}")
        print(f"Total events in session: {len(events)}")
        
        # Track actions and achievements
        actions_taken = []
        achievements_unlocked = set()
        total_reward = 0
        
        for event in events:
            event_type = event[1]
            metadata = json.loads(event[3]) if event[3] else {}
            state_after = json.loads(event[4]) if event[4] else {}
            state_before = json.loads(event[5]) if event[5] else {}
            reward = event[6] or 0
            terminated = event[7]
            
            total_reward += reward
            
            # Track runtime events (actions)
            if event_type == 'runtime' and 'action_name' in metadata:
                action = metadata['action_name']
                actions_taken.append(action)
                
            # Check for achievements
            if 'public_state' in state_after:
                public_state = state_after['public_state']
                if 'achievements_status' in public_state:
                    for ach, unlocked in public_state['achievements_status'].items():
                        if unlocked:
                            achievements_unlocked.add(ach)
            
            # Look for generation responses
            if event_type == 'generation' and metadata:
                print(f"\n--- Generation Event ---")
                if 'model' in metadata:
                    print(f"Model: {metadata['model']}")
                if 'response' in metadata:
                    response = metadata['response']
                    if isinstance(response, str):
                        print(f"Response preview: {response[:200]}")
                    elif isinstance(response, dict):
                        if 'content' in response:
                            print(f"Content: {response['content'][:200]}")
                        if 'tool_calls' in response:
                            print(f"Tool calls: {response['tool_calls']}")
        
        # Summary for this session
        print(f"\n--- Session Summary ---")
        print(f"Total reward: {total_reward}")
        print(f"Achievements unlocked: {achievements_unlocked if achievements_unlocked else 'None'}")
        print(f"Total actions taken: {len(actions_taken)}")
        
        if actions_taken:
            action_counts = Counter(actions_taken)
            print(f"\nAction distribution:")
            for action, count in action_counts.most_common(10):
                print(f"  {action}: {count}")
    
    # Overall analysis
    print(f"\n\n{'='*60}")
    print("OVERALL ANALYSIS")
    print(f"{'='*60}")
    
    # Check for any achievements across all sessions
    all_achievements = conn.execute("""
        SELECT DISTINCT
            json_extract_string(system_state_after, '$.public_state.achievements_status') as achievements
        FROM events
        WHERE system_state_after IS NOT NULL
        AND json_extract_string(system_state_after, '$.public_state.achievements_status') IS NOT NULL
        LIMIT 10
    """).fetchall()
    
    print(f"\nChecking achievement states...")
    total_unlocked = 0
    for ach_json, in all_achievements[:3]:
        if ach_json:
            achievements = json.loads(ach_json)
            unlocked = [k for k, v in achievements.items() if v]
            if unlocked:
                total_unlocked += len(unlocked)
                print(f"Found unlocked: {unlocked}")
    
    if total_unlocked == 0:
        print("❌ No achievements were unlocked in any session!")
    
    # Check for specific issues
    print(f"\n\n{'='*60}")
    print("POTENTIAL ISSUES")
    print(f"{'='*60}")
    
    # Look for model responses
    model_responses = conn.execute("""
        SELECT COUNT(*) 
        FROM events 
        WHERE event_type = 'generation'
    """).fetchone()[0]
    
    print(f"Generation events found: {model_responses}")
    
    if model_responses == 0:
        print("❌ No generation events found - agent may not be responding")
    
    # Look for specific metadata patterns
    print("\nChecking event metadata patterns...")
    metadata_samples = conn.execute("""
        SELECT event_type, metadata
        FROM events
        WHERE metadata IS NOT NULL
        AND metadata != '{}'
        AND metadata != '[]'
        LIMIT 20
    """).fetchall()
    
    for event_type, metadata_str in metadata_samples[:5]:
        metadata = json.loads(metadata_str)
        print(f"\n{event_type}: {list(metadata.keys())}")
        if 'action_name' in metadata:
            print(f"  Action: {metadata['action_name']}")
        if 'model' in metadata:
            print(f"  Model: {metadata['model']}")
    
    conn.close()

if __name__ == "__main__":
    db_path = "./traces_v2_synth/traces.duckdb"
    if Path(db_path).exists():
        analyze_agent_actions(db_path)
    else:
        print(f"❌ Database not found at {db_path}")