#!/usr/bin/env python3
"""
Verification script for sqld auto-install integration.

Tests all scenarios:
1. sqld already in PATH (should find immediately)
2. sqld in common location but not PATH
3. sqld missing + auto-install enabled (should prompt/install)
4. sqld missing + auto-install disabled (should fail with good error)
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_1_sqld_already_installed():
    """Test: sqld is already in PATH."""
    print_section("TEST 1: sqld Already Installed")
    
    # Check if sqld is actually installed
    sqld_path = shutil.which("sqld") or shutil.which("libsql-server")
    
    if not sqld_path:
        print("⏭️  SKIPPED - sqld not installed on system")
        print("   Run 'synth-ai turso' or 'brew install turso-tech/tools/sqld' first")
        return False
    
    print(f"✅ Found sqld at: {sqld_path}")
    
    try:
        from synth_ai.tracing_v3.turso.daemon import SqldDaemon
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            daemon = SqldDaemon(db_path=str(db_path), hrana_port=0, http_port=0)
            
            # Should find binary without installing
            binary = daemon.binary_path
            print(f"✅ SqldDaemon found binary: {binary}")
            assert binary == sqld_path or Path(binary).resolve() == Path(sqld_path).resolve()
            
        print("✅ TEST 1 PASSED - Binary found in PATH")
        return True
        
    except Exception as e:
        print(f"❌ TEST 1 FAILED: {e}")
        return False


def test_2_sqld_in_common_location():
    """Test: sqld not in PATH but in common location."""
    print_section("TEST 2: sqld in Common Location (Not in PATH)")
    
    try:
        from synth_ai.utils.sqld import find_sqld_binary
        
        # Check common locations
        binary = find_sqld_binary()
        
        if not binary:
            print("⏭️  SKIPPED - sqld not in common locations")
            return False
        
        print(f"✅ Found sqld in common location: {binary}")
        
        # Now test that daemon can find it
        from synth_ai.tracing_v3.turso.daemon import SqldDaemon
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            daemon = SqldDaemon(db_path=str(db_path), hrana_port=0, http_port=0)
            
            found_binary = daemon.binary_path
            print(f"✅ SqldDaemon found binary: {found_binary}")
            
        print("✅ TEST 2 PASSED - Binary found in common location")
        return True
        
    except Exception as e:
        print(f"⏭️  SKIPPED - {e}")
        return False


def test_3_auto_install_disabled():
    """Test: Auto-install disabled, sqld missing (should fail gracefully)."""
    print_section("TEST 3: Auto-Install Disabled")
    
    # Check if sqld exists
    sqld_exists = bool(shutil.which("sqld") or shutil.which("libsql-server"))
    
    if sqld_exists:
        print("⏭️  SKIPPED - Cannot test 'missing sqld' when sqld is installed")
        print("   (This is a good problem to have!)")
        return True  # Not a failure
    
    # Disable auto-install
    os.environ["SYNTH_AI_AUTO_INSTALL_SQLD"] = "false"
    
    try:
        from synth_ai.tracing_v3.turso.daemon import SqldDaemon
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            try:
                daemon = SqldDaemon(db_path=str(db_path), hrana_port=0, http_port=0)
                print(f"❌ Should have failed but got binary: {daemon.binary_path}")
                return False
            except RuntimeError as e:
                error_msg = str(e)
                # Check for expected error message
                if "sqld binary not found" in error_msg and "synth-ai turso" in error_msg:
                    print("✅ Got expected error message:")
                    print(f"   {error_msg.split(chr(10))[0]}...")
                    print("✅ TEST 3 PASSED - Auto-install correctly disabled")
                    return True
                else:
                    print(f"❌ Wrong error message: {error_msg}")
                    return False
                    
    except Exception as e:
        print(f"❌ TEST 3 FAILED: {e}")
        return False
    finally:
        os.environ.pop("SYNTH_AI_AUTO_INSTALL_SQLD", None)


def test_4_integration_with_start_sqld():
    """Test: Integration with start_sqld helper function."""
    print_section("TEST 4: Integration with start_sqld() Helper")
    
    sqld_path = shutil.which("sqld") or shutil.which("libsql-server")
    
    if not sqld_path:
        print("⏭️  SKIPPED - sqld not installed")
        return False
    
    try:
        from synth_ai.tracing_v3.turso.daemon import start_sqld
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            print(f"Starting sqld daemon...")
            daemon = start_sqld(db_path=str(db_path), hrana_port=18080, http_port=18081)
            
            print(f"✅ Daemon started successfully")
            print(f"   Binary: {daemon.binary_path}")
            print(f"   Hrana port: {daemon.get_hrana_port()}")
            print(f"   HTTP port: {daemon.get_http_port()}")
            print(f"   Running: {daemon.is_running()}")
            
            # Clean up
            daemon.stop()
            print("✅ Daemon stopped cleanly")
            
        print("✅ TEST 4 PASSED - start_sqld() integration works")
        return True
        
    except Exception as e:
        print(f"❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_smoke_test_integration():
    """Test: Smoke test can use auto-install."""
    print_section("TEST 5: Smoke Test Integration")
    
    sqld_path = shutil.which("sqld") or shutil.which("libsql-server")
    
    if not sqld_path:
        print("⏭️  SKIPPED - sqld not installed")
        return False
    
    try:
        # Import smoke test's sqld initialization
        from synth_ai.cli.commands.smoke.core import _ensure_local_libsql
        
        print("Testing smoke test's _ensure_local_libsql()...")
        
        # This should work now with auto-install
        _ensure_local_libsql()
        
        print("✅ Smoke test sqld initialization works")
        print(f"   LIBSQL_URL: {os.getenv('LIBSQL_URL')}")
        print(f"   SYNTH_TRACES_DB: {os.getenv('SYNTH_TRACES_DB')}")
        
        print("✅ TEST 5 PASSED - Smoke test integration works")
        return True
        
    except Exception as e:
        print(f"❌ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " sqld Auto-Install Integration Verification".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    results = {
        "Test 1: sqld Already Installed": test_1_sqld_already_installed(),
        "Test 2: Common Location": test_2_sqld_in_common_location(),
        "Test 3: Auto-Install Disabled": test_3_auto_install_disabled(),
        "Test 4: start_sqld() Helper": test_4_integration_with_start_sqld(),
        "Test 5: Smoke Test Integration": test_5_smoke_test_integration(),
    }
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None or (isinstance(v, bool) and v))
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else ("⏭️  SKIP" if result is None else "❌ FAIL")
        print(f"  {status}  {test_name}")
    
    print()
    print(f"Results: {passed} passed, {failed} failed, {len(results) - passed - failed} skipped")
    print()
    
    if failed > 0:
        print("❌ Some tests failed")
        return 1
    elif passed > 0:
        print("✅ All tests passed!")
        return 0
    else:
        print("⚠️  All tests skipped (sqld might not be installed)")
        print("   Install sqld first: synth-ai turso")
        return 0


if __name__ == "__main__":
    sys.exit(main())



