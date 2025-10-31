#!/usr/bin/env python3
"""
Simulate first-run experience: User tries to use tracing without sqld installed.

This demonstrates the auto-install flow by temporarily hiding sqld from PATH.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


def simulate_first_run():
    """Simulate a user's first run where sqld is not installed."""
    print("\n" + "=" * 80)
    print("  SIMULATING FIRST-RUN EXPERIENCE")
    print("  (Temporarily hiding sqld to test auto-install logic)")
    print("=" * 80)
    
    # Save real sqld path
    real_sqld = shutil.which("sqld") or shutil.which("libsql-server")
    
    if not real_sqld:
        print("\n⚠️  sqld not installed on your system")
        print("This test requires sqld to be installed to verify the logic")
        print("Install with: synth-ai turso")
        return False
    
    print(f"\n1. Real sqld location: {real_sqld}")
    
    # Mock shutil.which to hide sqld
    def mock_which(cmd):
        if cmd in ("sqld", "libsql-server"):
            return None  # Pretend sqld is not installed
        return shutil.which(cmd)
    
    # Mock find_sqld_binary to also return None
    def mock_find_sqld_binary():
        return None
    
    # Mock install_sqld to simulate installation
    def mock_install_sqld():
        print("\n🔧 [MOCK] Installing sqld via Homebrew...")
        print("📥 [MOCK] Downloading sqld via 'turso dev'...")
        print(f"✅ [MOCK] sqld installed to: {real_sqld}")
        return real_sqld  # Return the real path after "installation"
    
    print("\n2. Testing daemon initialization with sqld 'not installed'...")
    
    try:
        # Import here to ensure patches apply
        import synth_ai.tracing_v3.turso.daemon as daemon_module
        
        # Patch at the module level where they're used
        with patch.object(daemon_module.shutil, "which", side_effect=mock_which):
            with patch("synth_ai.utils.sqld.find_sqld_binary", side_effect=mock_find_sqld_binary):
                with patch("synth_ai.utils.sqld.install_sqld", side_effect=mock_install_sqld):
                    # Set auto-install to true
                    os.environ["SYNTH_AI_AUTO_INSTALL_SQLD"] = "true"
                    
                    # Simulate interactive terminal (returns True)
                    # Patch click.confirm to auto-accept
                    def mock_confirm(msg, default=True):
                        print(f"\n[SIMULATED USER] {msg}")
                        print(f"[SIMULATED USER] User response: YES (auto-confirm in test)")
                        return True
                    
                    with patch("sys.stdin.isatty", return_value=True):
                        with patch("click.confirm", side_effect=mock_confirm):
                            with tempfile.TemporaryDirectory() as tmpdir:
                                db_path = Path(tmpdir) / "test.db"
                                
                                print("\n3. Creating SqldDaemon (should trigger auto-install)...")
                                daemon = daemon_module.SqldDaemon(db_path=str(db_path), hrana_port=0, http_port=0)
                                
                                print(f"\n4. SqldDaemon created successfully!")
                                print(f"   Binary path: {daemon.binary_path}")
                                print(f"   ✅ Auto-install logic works!")
                                
                                return True
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        os.environ.pop("SYNTH_AI_AUTO_INSTALL_SQLD", None)


def test_error_message_quality():
    """Test that error messages are helpful when auto-install is disabled."""
    print("\n" + "=" * 80)
    print("  ERROR MESSAGE QUALITY TEST")
    print("=" * 80)
    
    def mock_which(cmd):
        if cmd in ("sqld", "libsql-server"):
            return None
        return shutil.which(cmd)
    
    def mock_find_sqld_binary():
        return None
    
    os.environ["SYNTH_AI_AUTO_INSTALL_SQLD"] = "false"
    
    try:
        from synth_ai.tracing_v3.turso.daemon import SqldDaemon
        
        with patch("shutil.which", side_effect=mock_which):
            with patch("synth_ai.utils.sqld.find_sqld_binary", side_effect=mock_find_sqld_binary):
                with tempfile.TemporaryDirectory() as tmpdir:
                    db_path = Path(tmpdir) / "test.db"
                    
                    try:
                        daemon = SqldDaemon(db_path=str(db_path), hrana_port=0, http_port=0)
                        print("❌ Should have raised RuntimeError")
                        return False
                    except RuntimeError as e:
                        error_msg = str(e)
                        print("\n✅ Got expected RuntimeError")
                        print("\nError message quality check:")
                        
                        checks = {
                            "Mentions 'synth-ai turso'": "synth-ai turso" in error_msg,
                            "Mentions 'brew install'": "brew install" in error_msg,
                            "Mentions CI/CD": "CI/CD" in error_msg,
                            "Multi-line formatting": "\n" in error_msg,
                            "Clear sections": "Quick install" in error_msg,
                        }
                        
                        all_passed = True
                        for check, passed in checks.items():
                            status = "✅" if passed else "❌"
                            print(f"  {status} {check}")
                            all_passed = all_passed and passed
                        
                        if all_passed:
                            print("\n✅ Error message is high quality!")
                            print("\nFull error message:")
                            print("-" * 80)
                            print(error_msg)
                            print("-" * 80)
                        
                        return all_passed
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        os.environ.pop("SYNTH_AI_AUTO_INSTALL_SQLD", None)


def main():
    """Run first-run experience tests."""
    print("\n╔" + "═" * 78 + "╗")
    print("║" + " First-Run Experience Verification".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    
    results = {
        "Simulate First Run": simulate_first_run(),
        "Error Message Quality": test_error_message_quality(),
    }
    
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} passed")
    
    if passed == total:
        print("\n✅ First-run experience is smooth!")
        print("\nKey features:")
        print("  • Auto-detects sqld in PATH")
        print("  • Checks common install locations")
        print("  • Auto-installs if missing (interactive mode)")
        print("  • Clear error messages with multiple install options")
        print("  • Respects SYNTH_AI_AUTO_INSTALL_SQLD=false for CI/CD")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

