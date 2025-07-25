#!/usr/bin/env python3
"""Debug DuckDB messages to see what's stored."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from synth_ai.tracing_v2.duckdb.manager import DuckDBTraceManager
import json

with DuckDBTraceManager("crafter_traces.duckdb") as db:
    # Check messages
    messages = db.conn.execute("""
        SELECT session_id, message_type, content, message_time 
        FROM messages 
        ORDER BY message_time 
        LIMIT 10
    """).df()
    
    print("Messages in database:")
    for idx, row in messages.iterrows():
        print(f"\n[{idx}] Type: {row['message_type']}, Time: {row['message_time']}")
        content = json.loads(row['content']) if isinstance(row['content'], str) else row['content']
        print(f"Content: {json.dumps(content, indent=2)[:200]}...")
        
    # Check if messages have actual text content
    print("\n\nChecking for messages with actual content...")
    messages_with_content = db.conn.execute("""
        SELECT COUNT(*) as count 
        FROM messages 
        WHERE content IS NOT NULL 
        AND content != '{}'
        AND json_extract(content, '$.content') IS NOT NULL
    """).fetchone()
    print(f"Messages with content: {messages_with_content[0]}")