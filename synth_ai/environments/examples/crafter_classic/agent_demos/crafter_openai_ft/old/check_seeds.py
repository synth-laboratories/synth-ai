#!/usr/bin/env python3
"""
Check what seeds are being used in the two experiments.
"""

import duckdb
import json

# Experiment IDs
EXPERIMENTS = {
    "gpt-4.1-nano": "194a3cd2-ecd3-4081-b46d-a7883e4a86f9",
    "gpt-4.1-mini": "da74a769-b33d-4b60-ae2a-52a4b67b3f35"
}

def check_seeds():
    """Check seeds for both experiments."""
    conn = duckdb.connect("crafter_traces.duckdb")
    
    for model_name, exp_id in EXPERIMENTS.items():
        print(f"\n🔍 {model_name.upper()} EXPERIMENT SEEDS")
        print("-" * 50)
        
        # Get all sessions for this experiment
        query = """
        SELECT session_id, metadata 
        FROM session_traces 
        WHERE experiment_id = ?
        ORDER BY session_id
        """
        
        results = conn.execute(query, [exp_id]).fetchall()
        
        seeds = []
        for session_id, metadata in results:
            if metadata:
                try:
                    metadata_list = json.loads(metadata) if isinstance(metadata, str) else metadata
                    
                    for meta_item in metadata_list:
                        if isinstance(meta_item, dict) and meta_item.get('metadata_type') == 'SessionMetadum':
                            data = meta_item.get('data', {})
                            
                            # Look for seed information
                            if 'seed' in data:
                                seeds.append({
                                    'session_id': session_id,
                                    'seed': data['seed']
                                })
                            elif 'instance_num' in data:
                                # Sometimes seed is derived from instance_num
                                seeds.append({
                                    'session_id': session_id,
                                    'instance_num': data['instance_num']
                                })
                except Exception as e:
                    print(f"Error parsing metadata for {session_id}: {e}")
        
        if seeds:
            print(f"Found {len(seeds)} sessions with seed info:")
            for seed_info in seeds:
                print(f"  {seed_info}")
        else:
            print("No explicit seed information found in metadata")
            print("Checking for instance numbers...")
            
            # Check for instance numbers as a proxy for seeds
            instance_nums = []
            for session_id, metadata in results:
                if metadata:
                    try:
                        metadata_list = json.loads(metadata) if isinstance(metadata, str) else metadata
                        for meta_item in metadata_list:
                            if isinstance(meta_item, dict) and meta_item.get('metadata_type') == 'SessionMetadum':
                                data = meta_item.get('data', {})
                                if 'instance_num' in data:
                                    instance_nums.append(data['instance_num'])
                    except:
                        pass
            
            if instance_nums:
                print(f"Instance numbers found: {sorted(instance_nums)}")
            else:
                print("No instance numbers found either")
    
    conn.close()

if __name__ == "__main__":
    check_seeds() 