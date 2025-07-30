#!/usr/bin/env python3
"""Analyze the latest run to understand why no achievements were unlocked."""

import duckdb
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

def analyze_latest_run(db_path: str):
    """Analyze the most recent run."""
    conn = duckdb.connect(db_path, read_only=True)
    
    print("🔍 Analyzing latest run with 97 steps and 0 achievements...\n")
    
    # Get the latest experiment
    latest_experiment = conn.execute("""
        SELECT experiment_id, experiment_name, created_at
        FROM experiments
        ORDER BY created_at DESC
        LIMIT 1
    """).fetchone()
    
    if latest_experiment:
        exp_id, exp_name, created_at = latest_experiment
        print(f"📊 Latest Experiment: {exp_name}")
        print(f"   ID: {exp_id}")
        print(f"   Created: {created_at}")
    
    # Get all sessions from latest experiment
    sessions = conn.execute("""
        SELECT DISTINCT s.session_id, s.num_timesteps, s.num_events
        FROM session_traces s
        JOIN events e ON s.session_id = e.session_id
        WHERE e.timestep_id IN (
            SELECT id FROM session_timesteps 
            WHERE experiment_id = ?
        )
        ORDER BY s.created_at DESC
    """, [exp_id]).fetchall()
    
    print(f"\n📊 Sessions in latest experiment: {len(sessions)}")
    
    for session_id, num_timesteps, num_events in sessions:
        print(f"\n{'='*60}")
        print(f"SESSION: {session_id}")
        print(f"Timesteps: {num_timesteps}, Events: {num_events}")
        print(f"{'='*60}")
        
        # Get all events for this session
        events = conn.execute("""
            SELECT event_type, metadata, system_state_after, reward
            FROM events
            WHERE session_id = ?
            ORDER BY id
        """, [session_id]).fetchall()
        
        # Analyze actions taken
        actions_taken = []
        action_sequences = []
        current_sequence = []
        total_reward = 0
        achievements_over_time = []
        
        for event_type, metadata_str, state_after_str, reward in events:
            if reward:
                total_reward += reward
                
            if metadata_str:
                metadata = json.loads(metadata_str)
                
                # Track runtime actions
                if event_type == 'runtime' and 'action_name' in metadata:
                    action = metadata['action_name']
                    actions_taken.append(action)
                    current_sequence.append(action)
                    
                    # Group actions into sequences (reset on certain actions)
                    if action in ['do', 'make_wood_pickaxe', 'place_table', 'sleep']:
                        if len(current_sequence) > 1:
                            action_sequences.append(current_sequence[:-1])
                        current_sequence = [action]
            
            # Check achievements
            if state_after_str:
                state_after = json.loads(state_after_str)
                if 'public_state' in state_after:
                    ps = state_after['public_state']
                    if 'achievements_status' in ps:
                        unlocked = [k for k, v in ps['achievements_status'].items() if v]
                        achievements_over_time.append(len(unlocked))
                    
                    # Check inventory
                    if 'inventory' in ps and len(actions_taken) % 20 == 0:  # Sample every 20 actions
                        inv = ps['inventory']
                        non_zero = {k: v for k, v in inv.items() if v > 0}
                        if non_zero:
                            print(f"\n📦 Inventory at action {len(actions_taken)}: {non_zero}")
        
        # Action analysis
        print(f"\n📊 ACTION ANALYSIS")
        print(f"Total actions taken: {len(actions_taken)}")
        
        if actions_taken:
            action_counts = Counter(actions_taken)
            print(f"\nAction distribution:")
            for action, count in action_counts.most_common():
                percentage = (count / len(actions_taken)) * 100
                print(f"  {action:20} {count:4} ({percentage:5.1f}%)")
            
            # Check for repetitive patterns
            print(f"\n🔄 REPETITIVE PATTERNS")
            # Find consecutive repeated actions
            consecutive_repeats = []
            if actions_taken:
                current_action = actions_taken[0]
                repeat_count = 1
                
                for action in actions_taken[1:]:
                    if action == current_action:
                        repeat_count += 1
                    else:
                        if repeat_count > 3:
                            consecutive_repeats.append((current_action, repeat_count))
                        current_action = action
                        repeat_count = 1
                
                if repeat_count > 3:
                    consecutive_repeats.append((current_action, repeat_count))
            
            if consecutive_repeats:
                print("Found repetitive sequences:")
                for action, count in consecutive_repeats[:5]:
                    print(f"  {action} repeated {count} times consecutively")
            else:
                print("No significant repetitive patterns found")
            
            # Check action sequences
            if action_sequences:
                print(f"\n🎯 ACTION SEQUENCES (movement -> action):")
                seq_counter = Counter([' → '.join(seq) for seq in action_sequences if len(seq) <= 5])
                for seq, count in seq_counter.most_common(5):
                    print(f"  {seq}: {count} times")
        
        # Achievement progress
        print(f"\n🏆 ACHIEVEMENT PROGRESS")
        print(f"Total reward: {total_reward}")
        if achievements_over_time:
            max_achievements = max(achievements_over_time)
            print(f"Max achievements reached: {max_achievements}")
            
            # Find when achievements were unlocked
            achievement_unlocks = []
            prev_count = 0
            for i, count in enumerate(achievements_over_time):
                if count > prev_count:
                    achievement_unlocks.append((i, count))
                    prev_count = count
            
            if achievement_unlocks:
                print("Achievement unlock timeline:")
                for step, count in achievement_unlocks:
                    print(f"  Step {step}: {count} achievements")
        else:
            print("No achievement data found")
        
        # Check final state
        final_event = events[-1] if events else None
        if final_event and final_event[2]:  # state_after
            final_state = json.loads(final_event[2])
            if 'public_state' in final_state:
                ps = final_state['public_state']
                
                print(f"\n📊 FINAL STATE")
                
                # Final inventory
                if 'inventory' in ps:
                    inv = ps['inventory']
                    non_zero = {k: v for k, v in inv.items() if v > 0}
                    print(f"Final inventory: {non_zero if non_zero else 'Empty'}")
                
                # Final achievements
                if 'achievements_status' in ps:
                    unlocked = [k for k, v in ps['achievements_status'].items() if v]
                    print(f"Final achievements: {unlocked if unlocked else 'None'}")
                
                # Player stats
                if 'health' in ps:
                    print(f"Final stats: Health={ps.get('health', '?')}, Food={ps.get('food', '?')}, Energy={ps.get('energy', '?')}")
    
    # Check for any generation/LM events
    print(f"\n{'='*60}")
    print("📊 LM/GENERATION EVENT CHECK")
    print(f"{'='*60}")
    
    lm_events = conn.execute("""
        SELECT COUNT(*) 
        FROM events 
        WHERE session_id IN (
            SELECT DISTINCT s.session_id
            FROM session_traces s
            JOIN events e ON s.session_id = e.session_id
            WHERE e.timestep_id IN (
                SELECT id FROM session_timesteps 
                WHERE experiment_id = ?
            )
        ) AND (event_type = 'lm' OR event_type = 'generation')
    """, [exp_id]).fetchone()[0]
    
    print(f"LM/Generation events in this experiment: {lm_events}")
    
    if lm_events == 0:
        print("⚠️  No LM events found - the agent may not be generating proper decisions")
    
    conn.close()

if __name__ == "__main__":
    db_path = "./traces_v2_synth/traces.duckdb"
    if Path(db_path).exists():
        analyze_latest_run(db_path)
    else:
        print(f"❌ Database not found at {db_path}")