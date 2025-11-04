#!/usr/bin/env python3
"""Test both local sqld and Modal SQLite backends."""

import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add synth-ai to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_local_sqld():
    """Test local development with sqld."""
    print("=" * 70)
    print("TEST 1: Local Development with sqld")
    print("=" * 70)
    
    # Clear Modal env vars
    for key in ["MODAL_TASK_ID", "MODAL_IS_REMOTE", "MODAL_ENVIRONMENT", 
                "SYNTH_TRACES_DB", "LIBSQL_URL", "TURSO_DATABASE_URL", 
                "LIBSQL_AUTH_TOKEN", "TURSO_AUTH_TOKEN"]:
        os.environ.pop(key, None)
    
    # Force reload config modules
    import importlib

    import synth_ai.tracing_v3.config
    import synth_ai.tracing_v3.storage.config
    importlib.reload(synth_ai.tracing_v3.config)
    importlib.reload(synth_ai.tracing_v3.storage.config)
    
    from synth_ai.tracing_v3.abstractions import SessionTrace
    from synth_ai.tracing_v3.config import _is_modal_environment, resolve_trace_db_settings
    from synth_ai.tracing_v3.storage.config import StorageBackend, StorageConfig
    from synth_ai.tracing_v3.storage.factory import create_storage
    
    try:
        # 1. Verify Modal NOT detected
        print("\n✓ Checking environment detection...")
        is_modal = _is_modal_environment()
        print(f"  Modal detected: {is_modal}")
        assert not is_modal, "Should NOT detect Modal environment"
        
        # 2. Verify sqld URL
        print("\n✓ Checking database URL...")
        url, token = resolve_trace_db_settings()
        print(f"  URL: {url}")
        print(f"  Auth token: {token}")
        assert url.startswith("http://127.0.0.1:"), f"Expected sqld URL, got {url}"
        
        # 3. Verify backend
        print("\n✓ Checking backend type...")
        config = StorageConfig()
        print(f"  Backend: {config.backend}")
        assert config.backend == StorageBackend.TURSO_NATIVE, f"Expected TURSO_NATIVE, got {config.backend}"
        
        # 4. Test database operations
        print("\n✓ Testing database operations...")
        storage = create_storage(config)
        await storage.initialize()
        print("  ✓ Database initialized")
        
        # Insert test trace
        trace = SessionTrace(
            session_id="test-local-sqld",
            created_at=datetime.now(UTC),
            markov_blanket_message_history=[],
            session_time_steps=[],
        )
        
        session_id = await storage.insert_session_trace(trace)
        print(f"  ✓ Inserted trace: {session_id}")
        
        # Retrieve trace
        retrieved = await storage.get_session_trace(session_id)
        assert retrieved is not None, "Should retrieve trace"
        assert retrieved["session_id"] == session_id, "Session ID should match"
        print(f"  ✓ Retrieved trace: {retrieved['session_id']}")
        
        await storage.close()
        print("  ✓ Database closed")
        
        print("\n" + "=" * 70)
        print("✅ LOCAL SQLD TEST PASSED")
        print("=" * 70)
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ LOCAL SQLD TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


async def test_modal_sqlite():
    """Test Modal environment with SQLite."""
    print("\n\n" + "=" * 70)
    print("TEST 2: Modal Environment with SQLite")
    print("=" * 70)
    
    # Set Modal env vars
    os.environ["MODAL_TASK_ID"] = "test-task-456"
    os.environ["MODAL_IS_REMOTE"] = "1"
    
    # Clear any DB overrides
    for key in ["SYNTH_TRACES_DB", "LIBSQL_URL", "TURSO_DATABASE_URL", 
                "LIBSQL_AUTH_TOKEN", "TURSO_AUTH_TOKEN"]:
        os.environ.pop(key, None)
    
    # Force reload config modules
    import importlib

    import synth_ai.tracing_v3.config
    import synth_ai.tracing_v3.storage.config
    importlib.reload(synth_ai.tracing_v3.config)
    importlib.reload(synth_ai.tracing_v3.storage.config)
    
    from synth_ai.tracing_v3.abstractions import SessionTrace
    from synth_ai.tracing_v3.config import _is_modal_environment, resolve_trace_db_settings
    from synth_ai.tracing_v3.storage.config import StorageBackend, StorageConfig
    from synth_ai.tracing_v3.storage.factory import create_storage
    
    try:
        # 1. Verify Modal IS detected
        print("\n✓ Checking environment detection...")
        is_modal = _is_modal_environment()
        print(f"  Modal detected: {is_modal}")
        assert is_modal, "Should detect Modal environment"
        
        # 2. Verify SQLite file URL
        print("\n✓ Checking database URL...")
        url, token = resolve_trace_db_settings()
        print(f"  URL: {url}")
        print(f"  Auth token: {token}")
        assert url == "file:/tmp/synth_traces.db", f"Expected SQLite file, got {url}"
        assert token is None, "SQLite should not have auth token"
        
        # 3. Verify backend
        print("\n✓ Checking backend type...")
        config = StorageConfig()
        print(f"  Backend: {config.backend}")
        assert config.backend == StorageBackend.SQLITE, f"Expected SQLITE, got {config.backend}"
        
        # 4. Test database operations
        print("\n✓ Testing database operations...")
        storage = create_storage(config)
        await storage.initialize()
        print("  ✓ Database initialized")
        
        # Insert test trace
        trace = SessionTrace(
            session_id="test-modal-sqlite",
            created_at=datetime.now(UTC),
            markov_blanket_message_history=[],
            session_time_steps=[],
        )
        
        session_id = await storage.insert_session_trace(trace)
        print(f"  ✓ Inserted trace: {session_id}")
        
        # Retrieve trace
        retrieved = await storage.get_session_trace(session_id)
        assert retrieved is not None, "Should retrieve trace"
        assert retrieved["session_id"] == session_id, "Session ID should match"
        print(f"  ✓ Retrieved trace: {retrieved['session_id']}")
        
        await storage.close()
        print("  ✓ Database closed")
        
        print("\n" + "=" * 70)
        print("✅ MODAL SQLITE TEST PASSED")
        print("=" * 70)
        return True
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ MODAL SQLITE TEST FAILED: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup Modal env vars
        for key in ["MODAL_TASK_ID", "MODAL_IS_REMOTE"]:
            os.environ.pop(key, None)


async def main():
    """Run all tests."""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  TESTING DUAL BACKEND SUPPORT: LOCAL SQLD + MODAL SQLITE  ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print("\n")
    
    results = []
    
    # Test 1: Local sqld
    result1 = await test_local_sqld()
    results.append(("Local sqld", result1))
    
    # Test 2: Modal SQLite
    result2 = await test_modal_sqlite()
    results.append(("Modal SQLite", result2))
    
    # Summary
    print("\n\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  TEST SUMMARY  ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    print()
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print()
    print("█" * 70)
    
    if all_passed:
        print("\n" + "🎉" * 35)
        print("  ALL TESTS PASSED! Both backends work correctly! ".center(70))
        print("🎉" * 35 + "\n")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED ⚠️\n")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)



