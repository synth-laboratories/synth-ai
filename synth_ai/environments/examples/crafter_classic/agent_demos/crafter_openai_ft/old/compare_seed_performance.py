#!/usr/bin/env python3
"""
Compare which seeds/instances performed better in nano vs mini.
Shows detailed performance breakdown by instance.
"""

import duckdb
import json
from typing import Dict, List, Any

# Experiment IDs
EXPERIMENTS = {
    "gpt-4.1-nano": "194a3cd2-ecd3-4081-b46d-a7883e4a86f9",
    "gpt-4.1-mini": "da74a769-b33d-4b60-ae2a-52a4b67b3f35"
}

def get_instance_performance(conn, experiment_id: str) -> Dict[int, Dict[str, Any]]:
    """Get performance data for each instance in an experiment."""
    query = """
    SELECT session_id, metadata 
    FROM session_traces 
    WHERE experiment_id = ?
    ORDER BY session_id
    """
    
    results = conn.execute(query, [experiment_id]).fetchall()
    
    instance_data = {}
    
    for session_id, metadata in results:
        if metadata:
            try:
                metadata_list = json.loads(metadata) if isinstance(metadata, str) else metadata
                
                # Look for instance_num in first metadata item
                instance_num = None
                for meta_item in metadata_list:
                    if isinstance(meta_item, dict) and meta_item.get('metadata_type') == 'SessionMetadum':
                        data = meta_item.get('data', {})
                        if 'instance_num' in data:
                            instance_num = data['instance_num']
                            break
                
                # Look for achievement data in any metadata item
                achievements = {}
                num_achievements = 0
                total_reward = 0.0
                rollout_length = 0
                terminated = False
                
                for meta_item in metadata_list:
                    if isinstance(meta_item, dict) and meta_item.get('metadata_type') == 'SessionMetadum':
                        data = meta_item.get('data', {})
                        
                        if 'achievements' in data:
                            achievements = data.get('achievements', {})
                            num_achievements = data.get('num_achievements', 0)
                            total_reward = data.get('total_reward', 0.0)
                            rollout_length = data.get('rollout_length', 0)
                            terminated = data.get('terminated', False)
                            break
                
                if instance_num is not None:
                    unlocked_achievements = [ach for ach, unlocked in achievements.items() if unlocked]
                    
                    instance_data[instance_num] = {
                        'session_id': session_id,
                        'num_achievements': num_achievements,
                        'unlocked_achievements': unlocked_achievements,
                        'total_reward': total_reward,
                        'rollout_length': rollout_length,
                        'terminated': terminated,
                        'all_achievements': achievements
                    }
            except Exception as e:
                print(f"Error parsing metadata for {session_id}: {e}")
    
    return instance_data

def compare_instance_performance():
    """Compare performance between nano and mini for each instance."""
    conn = duckdb.connect("crafter_traces.duckdb")
    
    # Get performance data for both experiments
    nano_data = get_instance_performance(conn, EXPERIMENTS["gpt-4.1-nano"])
    mini_data = get_instance_performance(conn, EXPERIMENTS["gpt-4.1-mini"])
    
    print("🔍 INSTANCE-BY-INSTANCE PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Compare each instance
    nano_wins = []
    mini_wins = []
    ties = []
    
    for instance_num in range(1, 11):  # Instances 1-10
        nano_perf = nano_data.get(instance_num, {})
        mini_perf = mini_data.get(instance_num, {})
        
        nano_achievements = nano_perf.get('num_achievements', 0)
        mini_achievements = mini_perf.get('num_achievements', 0)
        
        nano_unlocked = nano_perf.get('unlocked_achievements', [])
        mini_unlocked = mini_perf.get('unlocked_achievements', [])
        
        print(f"\n📊 Instance {instance_num} (Seed {42 + instance_num}):")
        print(f"  GPT-4.1-NANO:  {nano_achievements} achievements - {nano_unlocked}")
        print(f"  GPT-4.1-MINI:  {mini_achievements} achievements - {mini_unlocked}")
        
        if nano_achievements > mini_achievements:
            nano_wins.append(instance_num)
            print(f"  🏆 NANO WINS")
        elif mini_achievements > nano_achievements:
            mini_wins.append(instance_num)
            print(f"  🏆 MINI WINS")
        else:
            ties.append(instance_num)
            print(f"  🤝 TIE")
    
    # Summary statistics
    print(f"\n📈 PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"NANO wins: {len(nano_wins)} instances - {nano_wins}")
    print(f"MINI wins: {len(mini_wins)} instances - {mini_wins}")
    print(f"Ties: {len(ties)} instances - {ties}")
    
    # Calculate total achievements by model
    total_nano_achievements = sum(nano_data[i].get('num_achievements', 0) for i in range(1, 11))
    total_mini_achievements = sum(mini_data[i].get('num_achievements', 0) for i in range(1, 11))
    
    print(f"\n🏆 TOTAL ACHIEVEMENTS:")
    print(f"  GPT-4.1-NANO:  {total_nano_achievements}")
    print(f"  GPT-4.1-MINI:  {total_mini_achievements}")
    
    # Show which instances each model dominated
    print(f"\n🎯 INSTANCE DOMINANCE:")
    if nano_wins:
        print(f"  NANO dominated: {nano_wins}")
    if mini_wins:
        print(f"  MINI dominated: {mini_wins}")
    if ties:
        print(f"  Tied instances: {ties}")
    
    # Detailed breakdown by achievement type
    print(f"\n📊 ACHIEVEMENT BREAKDOWN BY INSTANCE:")
    print("-" * 60)
    
    all_achievements = set()
    for data in [nano_data, mini_data]:
        for instance_data in data.values():
            all_achievements.update(instance_data.get('all_achievements', {}).keys())
    
    achievement_types = sorted(all_achievements)
    
    for achievement in achievement_types:
        nano_count = sum(1 for data in nano_data.values() 
                        if data.get('all_achievements', {}).get(achievement, False))
        mini_count = sum(1 for data in mini_data.values() 
                        if data.get('all_achievements', {}).get(achievement, False))
        
        print(f"{achievement:20} | NANO: {nano_count:2d} | MINI: {mini_count:2d} | {'MINI' if mini_count > nano_count else 'NANO' if nano_count > mini_count else 'TIE'}")
    
    conn.close()

def show_detailed_instance_analysis():
    """Show detailed analysis of each instance."""
    conn = duckdb.connect("crafter_traces.duckdb")
    
    nano_data = get_instance_performance(conn, EXPERIMENTS["gpt-4.1-nano"])
    mini_data = get_instance_performance(conn, EXPERIMENTS["gpt-4.1-mini"])
    
    print(f"\n🔍 DETAILED INSTANCE ANALYSIS")
    print("=" * 80)
    
    for instance_num in range(1, 11):
        nano_perf = nano_data.get(instance_num, {})
        mini_perf = mini_data.get(instance_num, {})
        
        print(f"\n📋 Instance {instance_num} (Seed {42 + instance_num}):")
        print(f"  NANO: {nano_perf.get('num_achievements', 0)} achievements, reward: {nano_perf.get('total_reward', 0.0):.2f}, length: {nano_perf.get('rollout_length', 0)}")
        print(f"  MINI: {mini_perf.get('num_achievements', 0)} achievements, reward: {mini_perf.get('total_reward', 0.0):.2f}, length: {mini_perf.get('rollout_length', 0)}")
        
        # Show specific achievements
        nano_achievements = nano_perf.get('unlocked_achievements', [])
        mini_achievements = mini_perf.get('unlocked_achievements', [])
        
        if nano_achievements or mini_achievements:
            print(f"  NANO unlocked: {nano_achievements}")
            print(f"  MINI unlocked: {mini_achievements}")
    
    conn.close()

if __name__ == "__main__":
    compare_instance_performance()
    show_detailed_instance_analysis() 