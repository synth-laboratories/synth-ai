#!/usr/bin/env python3
"""Quick check of recent traces without locking the database."""

import sqlite3
import json
from pathlib import Path
from collections import Counter

# Use SQLite interface which is more permissive with locks
db_path = "./traces_v2_synth/traces.duckdb"

if Path(db_path).exists():
    try:
        # DuckDB files can be read with SQLite in read-only mode
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        
        print("🔍 Quick trace analysis...\n")
        
        # Get recent events
        cursor.execute("""
            SELECT event_type, metadata 
            FROM events 
            ORDER BY id DESC 
            LIMIT 100
        """)
        
        events = cursor.fetchall()
        print(f"Found {len(events)} recent events\n")
        
        # Count event types
        event_types = Counter([e[0] for e in events])
        print("Event type distribution:")
        for etype, count in event_types.most_common():
            print(f"  {etype}: {count}")
        
        # Check for actions
        print("\n🎮 Recent actions:")
        action_count = 0
        action_types = Counter()
        
        for event_type, metadata_str in events:
            if metadata_str and event_type == 'runtime':
                try:
                    metadata = json.loads(metadata_str)
                    if 'action_name' in metadata:
                        action_types[metadata['action_name']] += 1
                        action_count += 1
                        if action_count <= 10:
                            print(f"  - {metadata['action_name']}")
                except:
                    pass
        
        if action_types:
            print(f"\nAction summary:")
            for action, count in action_types.most_common():
                print(f"  {action}: {count}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying alternative analysis...")
        
        # If we can't read the DB, check for any JSON trace files
        trace_files = list(Path("./traces_v2_synth").glob("session_*.json"))
        if trace_files:
            print(f"Found {len(trace_files)} JSON trace files")
            latest = max(trace_files, key=lambda f: f.stat().st_mtime)
            print(f"Latest: {latest.name}")
            
            with open(latest) as f:
                data = json.load(f)
                print(f"Session ID: {data.get('session_id', 'Unknown')}")
                print(f"Events: {len(data.get('events', []))}")
else:
    print("No trace database found")