#!/usr/bin/env python3
"""
Analyze DuckDB traces with enhanced hooks: achievements, invalid actions, and inventory increases.
"""

import duckdb
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

def connect_to_db(db_path: str = "crafter_traces.duckdb"):
    """Connect to DuckDB database."""
    return duckdb.connect(db_path)

def get_experiment_info(conn, experiment_id: str) -> Dict[str, Any]:
    """Get experiment information."""
    query = """
    SELECT 
        e.id,
        e.name,
        e.description,
        e.created_at,
        sv.branch,
        sv.commit
    FROM experiments e
    LEFT JOIN experimental_systems es ON e.id = es.experiment_id
    LEFT JOIN system_versions sv ON es.system_version_id = sv.id
    WHERE e.id = ?
    """
    
    result = conn.execute(query, [experiment_id]).fetchone()
    if result:
        return {
            'id': result[0],
            'name': result[1],
            'description': result[2],
            'created_at': result[3],
            'branch': result[4],
            'commit': result[5]
        }
    return {}

def get_hook_events(conn, experiment_id: str) -> List[Dict[str, Any]]:
    """Get all hook events for an experiment."""
    query = """
    SELECT 
        e.session_id,
        e.event_type,
        e.event_metadata,
        e.metadata,
        e.event_time
    FROM events e
    JOIN session_traces st ON e.session_id = st.session_id
    WHERE st.experiment_id = ?
    AND e.event_type = 'hook'
    ORDER BY e.event_time
    """
    
    results = conn.execute(query, [experiment_id]).fetchall()
    events = []
    
    for row in results:
        session_id, event_type, event_metadata, metadata, timestamp = row
        
        # Parse metadata
        hook_data = {}
        if metadata:
            try:
                hook_data = json.loads(metadata) if isinstance(metadata, str) else metadata
            except:
                hook_data = {}
        
        events.append({
            'session_id': session_id,
            'event_type': event_type,
            'event_metadata': event_metadata,
            'hook_data': hook_data,
            'event_time': timestamp
        })
    
    return events

def analyze_achievement_hooks(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze achievement hook events."""
    achievement_events = [e for e in events if e['hook_data'].get('hook_name', '').endswith('achievement')]
    
    analysis = {
        'total_achievement_events': len(achievement_events),
        'easy_achievements': [],
        'medium_achievements': [],
        'hard_achievements': [],
        'achievement_by_session': {},
        'achievement_frequency': {}
    }
    
    for event in achievement_events:
        hook_data = event['hook_data']
        hook_name = hook_data.get('hook_name', '')
        achievements = hook_data.get('data', {}).get('achievements', [])
        session_id = event['session_id']
        
        # Categorize achievements
        if 'easy' in hook_name:
            analysis['easy_achievements'].extend(achievements)
        elif 'medium' in hook_name:
            analysis['medium_achievements'].extend(achievements)
        elif 'hard' in hook_name:
            analysis['hard_achievements'].extend(achievements)
        
        # Track by session
        if session_id not in analysis['achievement_by_session']:
            analysis['achievement_by_session'][session_id] = []
        analysis['achievement_by_session'][session_id].extend(achievements)
        
        # Track frequency
        for achievement in achievements:
            analysis['achievement_frequency'][achievement] = analysis['achievement_frequency'].get(achievement, 0) + 1
    
    return analysis

def analyze_invalid_action_hooks(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze invalid action hook events."""
    invalid_events = [e for e in events if e['hook_data'].get('hook_name') == 'invalid_action']
    
    analysis = {
        'total_invalid_events': len(invalid_events),
        'invalid_actions_by_type': {},
        'invalid_actions_by_session': {},
        'reasons': {}
    }
    
    for event in invalid_events:
        hook_data = event['hook_data']
        action = hook_data.get('data', {}).get('action', 'unknown')
        reason = hook_data.get('data', {}).get('reason', 'unknown')
        session_id = event['session_id']
        
        # Track by action type
        analysis['invalid_actions_by_type'][action] = analysis['invalid_actions_by_type'].get(action, 0) + 1
        
        # Track by session
        if session_id not in analysis['invalid_actions_by_session']:
            analysis['invalid_actions_by_session'][session_id] = []
        analysis['invalid_actions_by_session'][session_id].append(action)
        
        # Track reasons
        analysis['reasons'][reason] = analysis['reasons'].get(reason, 0) + 1
    
    return analysis

def analyze_inventory_hooks(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze inventory increase hook events."""
    inventory_events = [e for e in events if e['hook_data'].get('hook_name') == 'inventory_increase']
    
    analysis = {
        'total_inventory_events': len(inventory_events),
        'inventory_increases_by_item': {},
        'inventory_increases_by_session': {},
        'total_items_collected': 0
    }
    
    for event in inventory_events:
        hook_data = event['hook_data']
        increased_items = hook_data.get('data', {}).get('increased_items', [])
        session_id = event['session_id']
        
        for item_data in increased_items:
            item = item_data.get('item', 'unknown')
            increase = item_data.get('increase', 0)
            
            # Track by item
            if item not in analysis['inventory_increases_by_item']:
                analysis['inventory_increases_by_item'][item] = {'count': 0, 'total_increase': 0}
            analysis['inventory_increases_by_item'][item]['count'] += 1
            analysis['inventory_increases_by_item'][item]['total_increase'] += increase
            
            # Track by session
            if session_id not in analysis['inventory_increases_by_session']:
                analysis['inventory_increases_by_session'][session_id] = {}
            analysis['inventory_increases_by_session'][session_id][item] = analysis['inventory_increases_by_session'][session_id].get(item, 0) + increase
            
            analysis['total_items_collected'] += increase
    
    return analysis

def print_hook_analysis(experiment_id: str, db_path: str = "crafter_traces.duckdb"):
    """Print comprehensive hook analysis."""
    conn = connect_to_db(db_path)
    
    # Get experiment info
    exp_info = get_experiment_info(conn, experiment_id)
    if not exp_info:
        print(f"❌ Experiment {experiment_id} not found")
        return
    
    print(f"🔍 ENHANCED HOOK ANALYSIS")
    print("=" * 80)
    print(f"🧪 Experiment: {exp_info['name']}")
    print(f"📋 ID: {exp_info['id']}")
    print(f"🌿 Branch: {exp_info['branch']}")
    print(f"📍 Commit: {exp_info['commit']}")
    print(f"📅 Created: {exp_info['created_at']}")
    print()
    
    # Get all hook events
    events = get_hook_events(conn, experiment_id)
    print(f"📊 Total hook events: {len(events)}")
    print()
    
    # Analyze achievements
    achievement_analysis = analyze_achievement_hooks(events)
    print("🏆 ACHIEVEMENT ANALYSIS")
    print("-" * 50)
    print(f"Total achievement events: {achievement_analysis['total_achievement_events']}")
    print(f"Easy achievements: {len(achievement_analysis['easy_achievements'])} - {achievement_analysis['easy_achievements']}")
    print(f"Medium achievements: {len(achievement_analysis['medium_achievements'])} - {achievement_analysis['medium_achievements']}")
    print(f"Hard achievements: {len(achievement_analysis['hard_achievements'])} - {achievement_analysis['hard_achievements']}")
    print()
    
    if achievement_analysis['achievement_frequency']:
        print("Achievement frequency:")
        for achievement, count in sorted(achievement_analysis['achievement_frequency'].items()):
            print(f"  {achievement}: {count} times")
    print()
    
    # Analyze invalid actions
    invalid_analysis = analyze_invalid_action_hooks(events)
    print("❌ INVALID ACTION ANALYSIS")
    print("-" * 50)
    print(f"Total invalid action events: {invalid_analysis['total_invalid_events']}")
    print()
    
    if invalid_analysis['invalid_actions_by_type']:
        print("Invalid actions by type:")
        for action, count in sorted(invalid_analysis['invalid_actions_by_type'].items()):
            print(f"  {action}: {count} times")
        print()
    
    if invalid_analysis['reasons']:
        print("Invalid action reasons:")
        for reason, count in sorted(invalid_analysis['reasons'].items()):
            print(f"  {reason}: {count} times")
        print()
    
    # Analyze inventory increases
    inventory_analysis = analyze_inventory_hooks(events)
    print("📦 INVENTORY INCREASE ANALYSIS")
    print("-" * 50)
    print(f"Total inventory events: {inventory_analysis['total_inventory_events']}")
    print(f"Total items collected: {inventory_analysis['total_items_collected']}")
    print()
    
    if inventory_analysis['inventory_increases_by_item']:
        print("Inventory increases by item:")
        for item, data in sorted(inventory_analysis['inventory_increases_by_item'].items()):
            print(f"  {item}: {data['count']} events, +{data['total_increase']} total")
        print()
    
    # Session-level summary
    print("📋 SESSION-LEVEL SUMMARY")
    print("-" * 50)
    sessions_with_achievements = len([s for s in achievement_analysis['achievement_by_session'].values() if s])
    sessions_with_invalid = len(invalid_analysis['invalid_actions_by_session'])
    sessions_with_inventory = len(inventory_analysis['inventory_increases_by_session'])
    
    print(f"Sessions with achievements: {sessions_with_achievements}")
    print(f"Sessions with invalid actions: {sessions_with_invalid}")
    print(f"Sessions with inventory increases: {sessions_with_inventory}")
    print()
    
    # Hook effectiveness
    total_sessions = len(set(e['session_id'] for e in events))
    print("🎯 HOOK EFFECTIVENESS")
    print("-" * 50)
    print(f"Total sessions: {total_sessions}")
    print(f"Achievement detection rate: {sessions_with_achievements/total_sessions*100:.1f}%")
    print(f"Invalid action detection rate: {sessions_with_invalid/total_sessions*100:.1f}%")
    print(f"Inventory detection rate: {sessions_with_inventory/total_sessions*100:.1f}%")
    
    conn.close()

def list_recent_experiments(db_path: str = "crafter_traces.duckdb"):
    """List recent experiments."""
    conn = connect_to_db(db_path)
    
    query = """
    SELECT 
        e.id,
        e.name,
        e.description,
        e.created_at,
        COUNT(st.session_id) as session_count
    FROM experiments e
    LEFT JOIN session_traces st ON e.id = st.experiment_id
    GROUP BY e.id, e.name, e.description, e.created_at
    ORDER BY e.created_at DESC
    LIMIT 10
    """
    
    results = conn.execute(query).fetchall()
    
    print("📋 RECENT EXPERIMENTS")
    print("=" * 80)
    for row in results:
        exp_id, name, description, created_at, session_count = row
        print(f"🧪 {name}")
        print(f"📋 ID: {exp_id}")
        print(f"📅 Created: {created_at}")
        print(f"📊 Sessions: {session_count}")
        print(f"📝 Description: {description}")
        print("-" * 40)
    
    conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "list":
            list_recent_experiments()
        else:
            # Assume it's an experiment ID
            experiment_id = sys.argv[1]
            print_hook_analysis(experiment_id)
    else:
        print("Usage:")
        print("  python analyze_enhanced_hooks.py list                    # List recent experiments")
        print("  python analyze_enhanced_hooks.py <experiment_id>        # Analyze specific experiment")
        print()
        print("Example:")
        print("  python analyze_enhanced_hooks.py d3f4f503-036e-4a5a-a45e-28ae53ce48a9") 