#!/usr/bin/env python3
"""
Plot the frequency of achievements and invalid actions over time (by step number).
Terminal-only version.
"""

import duckdb
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple

def extract_step_from_metadata(metadata: str) -> int:
    """Extract step number from event metadata."""
    try:
        metadata_dict = json.loads(metadata) if isinstance(metadata, str) else metadata
        return metadata_dict.get('turn', 0)
    except:
        return 0

def parse_hook_metadata(event_metadata: str) -> List[Dict]:
    """Parse hook metadata from event_metadata string with better error handling."""
    hooks = []
    try:
        # The metadata is stored as a list of strings, each containing a hook dict
        hook_list = json.loads(event_metadata) if isinstance(event_metadata, str) else event_metadata
        
        for hook_str in hook_list:
            if isinstance(hook_str, str):
                # Use regex to extract hook_name more reliably
                hook_name_match = re.search(r"'hook_name':\s*'([^']+)'", hook_str)
                if hook_name_match:
                    hook_name = hook_name_match.group(1)
                    hooks.append({'hook_name': hook_name})
            else:
                hooks.append(hook_str)
    except Exception as e:
        # Try alternative parsing if JSON fails
        try:
            # Look for hook_name patterns in the string
            hook_names = re.findall(r"'hook_name':\s*'([^']+)'", event_metadata)
            for hook_name in hook_names:
                hooks.append({'hook_name': hook_name})
        except:
            pass
    
    return hooks

def analyze_hook_frequency(experiment_id: str):
    """Analyze hook frequency over time."""
    conn = duckdb.connect("crafter_traces.duckdb")
    
    print(f"📊 ANALYZING HOOK FREQUENCY OVER TIME")
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
    
    # Track hook frequency by step
    step_achievements = defaultdict(int)
    step_invalid_actions = defaultdict(int)
    step_inventory_increases = defaultdict(int)
    
    # Track by session for more detailed analysis
    session_data = defaultdict(lambda: {
        'achievements': defaultdict(int),
        'invalid_actions': defaultdict(int),
        'inventory_increases': defaultdict(int)
    })
    
    for row in result:
        session_id, event_type, event_metadata, metadata = row
        
        # Extract step number
        step = extract_step_from_metadata(metadata)
        
        # Parse hook metadata
        hooks = parse_hook_metadata(event_metadata)
        
        for hook in hooks:
            hook_name = hook.get('hook_name', 'unknown')
            
            if hook_name == 'easy_achievement' or hook_name == 'medium_achievement' or hook_name == 'hard_achievement':
                step_achievements[step] += 1
                session_data[session_id]['achievements'][step] += 1
            elif hook_name == 'invalid_action':
                step_invalid_actions[step] += 1
                session_data[session_id]['invalid_actions'][step] += 1
            elif hook_name == 'inventory_increase':
                step_inventory_increases[step] += 1
                session_data[session_id]['inventory_increases'][step] += 1
    
    # Prepare data for plotting
    max_step = max(
        max(step_achievements.keys()) if step_achievements else 0,
        max(step_invalid_actions.keys()) if step_invalid_actions else 0,
        max(step_inventory_increases.keys()) if step_inventory_increases else 0
    )
    
    steps = list(range(max_step + 1))
    achievement_freq = [step_achievements[step] for step in steps]
    invalid_action_freq = [step_invalid_actions[step] for step in steps]
    inventory_freq = [step_inventory_increases[step] for step in steps]
    
    # Print summary statistics
    print("📈 SUMMARY STATISTICS")
    print("-" * 50)
    print(f"Total steps analyzed: {max_step + 1}")
    print(f"Total achievements: {sum(achievement_freq)}")
    print(f"Total invalid actions: {sum(invalid_action_freq)}")
    print(f"Total inventory increases: {sum(inventory_freq)}")
    print()
    
    print("🏆 ACHIEVEMENT ANALYSIS")
    print("-" * 50)
    achievement_steps = [step for step, freq in step_achievements.items() if freq > 0]
    if achievement_steps:
        print(f"Achievements occur at steps: {sorted(achievement_steps)}")
        print(f"Most common achievement step: {max(step_achievements.items(), key=lambda x: x[1])}")
    else:
        print("No achievements found")
    print()
    
    print("❌ INVALID ACTION ANALYSIS")
    print("-" * 50)
    invalid_steps = [step for step, freq in step_invalid_actions.items() if freq > 0]
    if invalid_steps:
        print(f"Invalid actions occur at steps: {sorted(invalid_steps)}")
        print(f"Most common invalid action step: {max(step_invalid_actions.items(), key=lambda x: x[1])}")
    else:
        print("No invalid actions found")
    print()
    
    print("📦 INVENTORY ANALYSIS")
    print("-" * 50)
    inventory_steps = [step for step, freq in step_inventory_increases.items() if freq > 0]
    if inventory_steps:
        print(f"Inventory increases occur at steps: {sorted(inventory_steps)}")
        print(f"Most common inventory increase step: {max(step_inventory_increases.items(), key=lambda x: x[1])}")
    else:
        print("No inventory increases found")
    
    # Create ASCII chart
    print("\n📊 ASCII FREQUENCY CHART")
    print("=" * 80)
    print("Step | Achievements | Invalid Actions | Inventory")
    print("-" * 80)
    
    for step in steps:
        achievements = step_achievements[step]
        invalid_actions = step_invalid_actions[step]
        inventory = step_inventory_increases[step]
        
        if achievements > 0 or invalid_actions > 0 or inventory > 0:
            print(f"{step:4d} | {achievements:11d} | {invalid_actions:14d} | {inventory:9d}")
    
    # Session-by-session breakdown
    print("\n📋 SESSION-BY-SESSION BREAKDOWN")
    print("-" * 50)
    for session_id, data in session_data.items():
        print(f"\nSession: {session_id}")
        if data['achievements']:
            print(f"  Achievements: {dict(data['achievements'])}")
        if data['invalid_actions']:
            print(f"  Invalid actions: {dict(data['invalid_actions'])}")
        if data['inventory_increases']:
            print(f"  Inventory increases: {dict(data['inventory_increases'])}")
    
    conn.close()
    
    return {
        'steps': steps,
        'achievement_freq': achievement_freq,
        'invalid_action_freq': invalid_action_freq,
        'inventory_freq': inventory_freq,
        'session_data': session_data
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        experiment_id = sys.argv[1]
        analyze_hook_frequency(experiment_id)
    else:
        print("Usage: python plot_hook_frequency.py <experiment_id>")
        print("Example: python plot_hook_frequency.py 77022cce-4bda-4415-9bce-0095e4ef2237") 