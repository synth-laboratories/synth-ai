#!/usr/bin/env python3
"""Simple test to verify v3 fixes work correctly."""

import asyncio
import os
import tempfile
import uuid
import pytest

from synth_ai.tracing_v3.turso.daemon import SqldDaemon
from synth_ai.tracing_v3.turso.manager import AsyncSQLTraceManager
from synth_ai.tracing_v3.session_tracer import SessionTracer


async def main():
    """Test basic functionality with fresh database."""
    
    # Create completely unique path
    test_id = uuid.uuid4().hex
    db_path = os.path.join(tempfile.gettempdir(), f"v3_test_{test_id}.db")
    http_port = 9500 + (hash(test_id) % 100)
    
    print(f"Test ID: {test_id}")
    print(f"DB Path: {db_path}")
    print(f"HTTP Port: {http_port}")
    
    # Start daemon
    daemon = SqldDaemon(db_path=db_path, http_port=http_port)
    daemon.start()
    
    # Wait for startup
    await asyncio.sleep(2)
    
    # Use file URL - sqld creates a directory structure  
    actual_db_file = os.path.join(db_path, "dbs", "default", "data")
    db_url = f"sqlite+aiosqlite:///{actual_db_file}"
    
    try:
        # Create exactly 3 sessions
        for i in range(3):
            tracer = SessionTracer(db_url=db_url, auto_save=True)
            session_id = f"test_session_{i}"
            
            await tracer.start_session(session_id)
            
            async with tracer.timestep("test_step"):
                await tracer.record_message(
                    content=f"Test message {i}",
                    message_type="user"
                )
            
            await tracer.end_session()
            await tracer.close()
            
            print(f"✅ Created session {i}")
        
        # Verify count
        manager = AsyncSQLTraceManager(db_url=db_url)
        await manager.initialize()
        
        df = await manager.query_traces("SELECT session_id FROM session_traces ORDER BY session_id")
        sessions = df['session_id'].tolist()
        
        print(f"\nSessions in database: {sessions}")
        print(f"Expected: ['test_session_0', 'test_session_1', 'test_session_2']")
        
        assert len(sessions) == 3, f"Expected 3 sessions, got {len(sessions)}"
        assert sessions == ['test_session_0', 'test_session_1', 'test_session_2']
        
        print("\n✅ Test passed! Exactly 3 sessions as expected.")
        
        await manager.close()
        
    finally:
        # Clean up
        daemon.stop()
        await asyncio.sleep(0.5)
        
        # Clean up the directory structure
        if os.path.exists(db_path):
            import shutil
            shutil.rmtree(db_path)
            print(f"🧹 Cleaned up {db_path}")


if __name__ == "__main__":
    asyncio.run(main())