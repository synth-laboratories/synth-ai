#!/usr/bin/env python3
"""Analyze why there are no rewards or achievements."""

import duckdb
import json
from pathlib import Path
from collections import Counter, defaultdict

def analyze_no_rewards(db_path: str):
    """Analyze why there are no rewards despite actions being taken."""
    conn = duckdb.connect(db_path, read_only=True)
    
    print("🔍 Analyzing why there are no rewards or achievements...\n")
    
    # Get the latest experiment
    latest_exp = conn.execute("""
        SELECT experiment_id, experiment_name
        FROM experiments
        ORDER BY created_at DESC
        LIMIT 1
    """).fetchone()
    
    if not latest_exp:
        print("No experiments found")
        return
        
    exp_id, exp_name = latest_exp
    print(f"📊 Latest Experiment: {exp_name}")
    print(f"   ID: {exp_id}\n")
    
    # Get all sessions from latest experiment
    sessions = conn.execute("""
        SELECT DISTINCT e.session_id
        FROM events e
        JOIN session_timesteps st ON e.timestep_id = st.id
        WHERE st.experiment_id = ?
        ORDER BY e.session_id
    """, [exp_id]).fetchall()
    
    print(f"Found {len(sessions)} sessions\n")
    
    # Analyze each session
    for i, (session_id,) in enumerate(sessions[:3]):  # First 3 sessions
        print(f"\n{'='*60}")
        print(f"SESSION {i}: {session_id}")
        print(f"{'='*60}")
        
        # Get all events for this session
        events = conn.execute("""
            SELECT event_type, metadata, system_state_after, reward
            FROM events
            WHERE session_id = ?
            ORDER BY id
        """, [session_id]).fetchall()
        
        # Track actions and their results
        action_results = []
        total_reward = 0
        achievements_timeline = []
        inventory_timeline = []
        
        for event_type, metadata_str, state_after_str, reward in events:
            if event_type == 'runtime' and metadata_str:
                metadata = json.loads(metadata_str)
                action_name = metadata.get('action_name', 'unknown')
                
                # Look for the corresponding environment event
                result = {
                    'action': action_name,
                    'reward': reward or 0,
                    'achievements_unlocked': []
                }
                
                if state_after_str:
                    state_after = json.loads(state_after_str)
                    if 'public_state' in state_after:
                        ps = state_after['public_state']
                        
                        # Check achievements
                        if 'achievements_status' in ps:
                            unlocked = [k for k, v in ps['achievements_status'].items() if v]
                            achievements_timeline.append((action_name, unlocked))
                            if len(unlocked) > len(result['achievements_unlocked']):
                                result['achievements_unlocked'] = unlocked
                        
                        # Check inventory
                        if 'inventory' in ps:
                            inv = ps['inventory']
                            non_zero = {k: v for k, v in inv.items() if v > 0 and k not in ['health', 'food', 'drink', 'energy']}
                            if non_zero:
                                inventory_timeline.append((action_name, non_zero))
                
                action_results.append(result)
                if reward:
                    total_reward += reward
        
        # Analyze action effectiveness
        print(f"\n📊 ACTION ANALYSIS")
        print(f"Total actions: {len(action_results)}")
        print(f"Total reward: {total_reward}")
        
        # Count actions by type
        action_counts = Counter([r['action'] for r in action_results])
        print(f"\nAction distribution:")
        for action, count in action_counts.most_common(10):
            print(f"  {action:20} {count:3}")
        
        # Check for successful resource collection
        print(f"\n📦 RESOURCE COLLECTION")
        successful_collections = []
        for i, (action, inv) in enumerate(inventory_timeline):
            if i > 0 and inv != inventory_timeline[i-1][1]:
                # Inventory changed
                prev_inv = inventory_timeline[i-1][1] if i > 0 else {}
                new_items = {k: v for k, v in inv.items() if v > prev_inv.get(k, 0)}
                if new_items:
                    successful_collections.append((action, new_items))
        
        if successful_collections:
            print("Successful collections:")
            for action, items in successful_collections[:5]:
                print(f"  After '{action}': gained {items}")
        else:
            print("No successful resource collections detected!")
        
        # Check specific action sequences
        print(f"\n🔍 ACTION SEQUENCE ANALYSIS")
        # Look for 'do' actions and their context
        do_actions = []
        for i, result in enumerate(action_results):
            if result['action'] == 'do':
                context = {
                    'prev_action': action_results[i-1]['action'] if i > 0 else 'start',
                    'next_action': action_results[i+1]['action'] if i < len(action_results)-1 else 'end',
                    'reward': result['reward']
                }
                do_actions.append(context)
        
        if do_actions:
            print(f"Found {len(do_actions)} 'do' actions")
            # Check what happened before 'do' actions
            prev_action_counts = Counter([d['prev_action'] for d in do_actions])
            print("Actions before 'do':")
            for action, count in prev_action_counts.most_common(5):
                print(f"  {action}: {count}")
        
        # Check for make_wood_pickaxe attempts
        pickaxe_attempts = [r for r in action_results if r['action'] == 'make_wood_pickaxe']
        if pickaxe_attempts:
            print(f"\n🔨 PICKAXE CRAFTING")
            print(f"Attempted to make wood pickaxe {len(pickaxe_attempts)} times")
        
        # Final inventory check
        if inventory_timeline:
            final_inv = inventory_timeline[-1][1]
            print(f"\n📦 FINAL INVENTORY: {final_inv if final_inv else 'Empty'}")
        
        # Achievement check
        if achievements_timeline:
            final_achievements = achievements_timeline[-1][1]
            print(f"🏆 FINAL ACHIEVEMENTS: {final_achievements if final_achievements else 'None'}")
    
    # Check for any rewards across all sessions
    print(f"\n\n{'='*60}")
    print("📊 OVERALL REWARD ANALYSIS")
    print(f"{'='*60}")
    
    total_rewards = conn.execute("""
        SELECT SUM(reward) as total, COUNT(*) as count
        FROM events
        WHERE reward IS NOT NULL AND reward != 0
        AND session_id IN (
            SELECT DISTINCT e.session_id
            FROM events e
            JOIN session_timesteps st ON e.timestep_id = st.id
            WHERE st.experiment_id = ?
        )
    """, [exp_id]).fetchone()
    
    if total_rewards:
        total, count = total_rewards
        print(f"Total non-zero rewards: {total or 0}")
        print(f"Number of reward events: {count or 0}")
    
    # Check if rewards are being recorded at all
    print("\n🔍 REWARD RECORDING CHECK")
    sample_rewards = conn.execute("""
        SELECT reward, metadata
        FROM events
        WHERE event_type = 'environment'
        AND session_id IN (
            SELECT DISTINCT e.session_id
            FROM events e
            JOIN session_timesteps st ON e.timestep_id = st.id
            WHERE st.experiment_id = ?
        )
        LIMIT 20
    """, [exp_id]).fetchall()
    
    reward_values = [r for r, _ in sample_rewards if r is not None]
    print(f"Sample reward values: {reward_values[:10]}")
    
    conn.close()

if __name__ == "__main__":
    db_path = "./traces_v2_synth/traces.duckdb"
    if Path(db_path).exists():
        analyze_no_rewards(db_path)
    else:
        print(f"❌ Database not found at {db_path}")