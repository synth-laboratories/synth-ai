#!/usr/bin/env python3
"""
Intensive integration test for cloudflared tunnel creation and DNS resolution.

This script tests every step of the tunnel creation process with detailed diagnostics.
Run this to diagnose cloudflared issues.
"""

import asyncio
import logging
import os
import platform
import subprocess
import sys
import time

import httpx

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Import synth-ai modules
try:
    from synth_ai.cloudflare import (
        get_cloudflared_path,
        open_quick_tunnel,
        open_quick_tunnel_with_dns_verification,
        require_cloudflared,
        resolve_hostname_with_explicit_resolvers,
    )
except ImportError as e:
    logger.error(f"Failed to import synth_ai.cloudflare: {e}")
    logger.error("Make sure you're in the synth-ai directory and dependencies are installed")
    sys.exit(1)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_test(name: str, status: str = "RUNNING"):
    """Print a test status."""
    status_symbol = {
        "RUNNING": "🔄",
        "PASS": "✅",
        "FAIL": "❌",
        "SKIP": "⏭️",
        "WARN": "⚠️",
    }.get(status, "❓")
    print(f"{status_symbol} {name}")


def test_system_info():
    """Test 1: System information."""
    print_section("TEST 1: System Information")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"PATH: {os.environ.get('PATH', 'NOT SET')[:200]}...")


def test_cloudflared_binary_location():
    """Test 2: Find cloudflared binary."""
    print_section("TEST 2: Cloudflared Binary Location")
    try:
        path = get_cloudflared_path()
        if path:
            print_test("Binary found", "PASS")
            print(f"  Path: {path}")
            print(f"  Exists: {path.exists()}")
            print(f"  Executable: {os.access(path, os.X_OK)}")
            print(f"  Size: {path.stat().st_size} bytes")
            return path
        else:
            print_test("Binary not found", "FAIL")
            print("  cloudflared is not in PATH or common locations")
            return None
    except Exception as e:
        print_test(f"Error finding binary: {e}", "FAIL")
        return None


def test_cloudflared_version():
    """Test 3: Verify cloudflared can run."""
    print_section("TEST 3: Cloudflared Version Check")
    try:
        bin_path = require_cloudflared()
        print(f"  Using binary: {bin_path}")
        
        # Test --version
        proc = subprocess.run(
            [str(bin_path), "--version"],
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        
        if proc.returncode == 0:
            print_test("cloudflared --version", "PASS")
            print(f"  STDOUT:\n{proc.stdout}")
            if proc.stderr:
                print(f"  STDERR:\n{proc.stderr}")
            return True
        else:
            print_test("cloudflared --version", "FAIL")
            print(f"  Exit code: {proc.returncode}")
            print(f"  STDOUT: {proc.stdout}")
            print(f"  STDERR: {proc.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print_test("cloudflared --version", "FAIL")
        print("  Command timed out after 10s")
        return False
    except FileNotFoundError:
        print_test("cloudflared binary not found", "FAIL")
        return False
    except Exception as e:
        print_test(f"Error running cloudflared: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        return False


def find_available_port(start_port: int = 18080) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + 100):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result != 0:
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find available port starting from {start_port}")


def test_port_availability(port: int = 8080):
    """Test 4: Check if port is available."""
    print_section(f"TEST 4: Port {port} Availability")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        
        if result == 0:
            print_test(f"Port {port} is IN USE", "WARN")
            print(f"  Something is already listening on port {port}")
            print("  Will find an available port for testing")
            return False
        else:
            print_test(f"Port {port} is available", "PASS")
            return True
    except Exception as e:
        print_test(f"Error checking port: {e}", "FAIL")
        return False


def test_cloudflared_tunnel_creation(port: int = 8080):
    """Test 5: Create a tunnel (without DNS verification)."""
    print_section(f"TEST 5: Cloudflared Tunnel Creation (port {port})")
    
    # Start a simple HTTP server on the port
    import http.server
    import socketserver
    import threading
    
    server = None
    server_thread = None
    
    try:
        # Start a simple HTTP server - allow reuse of address
        handler = http.server.SimpleHTTPRequestHandler
        server = socketserver.TCPServer(("127.0.0.1", port), handler)
        server.allow_reuse_address = True  # Allow reuse to avoid "Address already in use"
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print(f"  Started test HTTP server on port {port}")
        time.sleep(0.5)  # Give server time to start
        
        # Try to create tunnel
        print("  Attempting to create tunnel...")
        try:
            url, proc = open_quick_tunnel(port, wait_s=15.0)
            print_test("Tunnel created", "PASS")
            print(f"  URL: {url}")
            print(f"  Process PID: {proc.pid}")
            print(f"  Process running: {proc.poll() is None}")
            
            # Clean up
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            
            return url, True
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str and "Too Many Requests" in error_str
            
            if is_rate_limit:
                print_test("Tunnel creation failed", "FAIL")
                print("\n  ⚠️  RATE LIMIT DETECTED")
                print("  " + "=" * 70)
                print("  Cloudflare is blocking quick tunnel creation due to rate limiting.")
                print("  This happens when too many quick tunnels are created in a short time.")
                print("\n  Solutions:")
                print("    1. ⏰ Wait 5-10 minutes for the rate limit to reset")
                print("    2. 🔑 Use managed tunnels (set SYNTH_API_KEY env var)")
                print("    3. ♻️  Reuse existing tunnel (set INTERCEPTOR_TUNNEL_URL env var)")
                print("  " + "=" * 70)
            else:
                print_test("Tunnel creation failed", "FAIL")
                print(f"  Error: {error_str[:500]}")
                if len(error_str) > 500:
                    print("  ... (truncated, full error in traceback below)")
            import traceback
            traceback.print_exc()
            return None, False
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if server_thread:
            server_thread.join(timeout=2.0)


async def test_dns_resolution(hostname: str):
    """Test 6: DNS resolution with explicit resolvers."""
    print_section(f"TEST 6: DNS Resolution for {hostname}")
    try:
        ip = await resolve_hostname_with_explicit_resolvers(hostname)
        print_test("DNS resolution", "PASS")
        print(f"  {hostname} -> {ip}")
        return ip
    except Exception as e:
        print_test("DNS resolution", "FAIL")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_tunnel_with_dns_verification(port: int = 8080):
    """Test 7: Create tunnel with DNS verification."""
    print_section(f"TEST 7: Tunnel Creation with DNS Verification (port {port})")
    
    # Start a simple HTTP server
    import http.server
    import socketserver
    import threading
    
    server = None
    server_thread = None
    
    try:
        handler = http.server.SimpleHTTPRequestHandler
        server = socketserver.TCPServer(("127.0.0.1", port), handler)
        server.allow_reuse_address = True  # Allow reuse to avoid "Address already in use"
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print(f"  Started test HTTP server on port {port}")
        await asyncio.sleep(0.5)
        
        # Try to create tunnel with DNS verification
        print("  Attempting to create tunnel with DNS verification...")
        try:
            url, proc = await open_quick_tunnel_with_dns_verification(
                port,
                wait_s=15.0,
                max_retries=3,
                dns_timeout_s=30.0,
            )
            print_test("Tunnel with DNS verification", "PASS")
            print(f"  URL: {url}")
            print(f"  Process PID: {proc.pid}")
            
            # Clean up
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            
            return url, True
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str and "Too Many Requests" in error_str
            
            if is_rate_limit:
                print_test("Tunnel with DNS verification failed", "FAIL")
                print("\n  ⚠️  RATE LIMIT DETECTED")
                print("  " + "=" * 70)
                print("  Cloudflare is blocking quick tunnel creation due to rate limiting.")
                print("  This happens when too many quick tunnels are created in a short time.")
                print("\n  Solutions:")
                print("    1. ⏰ Wait 5-10 minutes for the rate limit to reset")
                print("    2. 🔑 Use managed tunnels (set SYNTH_API_KEY env var)")
                print("    3. ♻️  Reuse existing tunnel (set INTERCEPTOR_TUNNEL_URL env var)")
                print("  " + "=" * 70)
            else:
                print_test("Tunnel with DNS verification failed", "FAIL")
                print(f"  Error: {error_str[:500]}")
                if len(error_str) > 500:
                    print("  ... (truncated, full error in traceback below)")
            import traceback
            traceback.print_exc()
            return None, False
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if server_thread:
            server_thread.join(timeout=2.0)


async def test_http_connectivity(url: str):
    """Test 8: HTTP connectivity to tunnel URL."""
    print_section(f"TEST 8: HTTP Connectivity to {url}")
    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.get(url)
            print_test("HTTP connectivity", "PASS")
            print(f"  Status: {response.status_code}")
            print(f"  Headers: {dict(response.headers)}")
            return True
    except Exception as e:
        print_test("HTTP connectivity", "FAIL")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_rate_limit_status():
    """Test 8: Check rate limit status."""
    print_section("TEST 8: Rate Limit Status Check")
    try:
        from synth_ai.cloudflare import check_rate_limit_status
        result = await check_rate_limit_status(test_port=19998)
        
        if result["is_rate_limited"]:
            print_test("Rate limit check", "WARN")
            print("\n  ⚠️  RATE LIMIT DETECTED")
            print("  " + "=" * 70)
            print("  Cloudflare is currently blocking quick tunnel creation.")
            print("  This means too many quick tunnels were created recently.")
            print("\n  Solutions:")
            print("    1. ⏰ Wait 5-10 minutes for the rate limit to reset")
            print("    2. 🔑 Use managed tunnels (set SYNTH_API_KEY env var)")
            print("    3. ♻️  Reuse existing tunnel (set INTERCEPTOR_TUNNEL_URL env var)")
            if result["error_message"]:
                print(f"\n  Error details:\n{result['error_message'][:500]}")
            print("  " + "=" * 70)
        else:
            print_test("Rate limit check", "PASS")
            print("  ✅ Not currently rate-limited")
            print("  Quick tunnel creation should work")
        
        return not result["is_rate_limited"]
    except Exception as e:
        print_test(f"Rate limit check failed: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        return False


async def test_manual_cloudflared_command(port: int = 8080):
    """Test 9: Run cloudflared command manually to see raw output."""
    print_section(f"TEST 9: Manual Cloudflared Command (port {port})")
    
    # Start a simple HTTP server
    import http.server
    import socketserver
    import threading
    
    server = None
    server_thread = None
    
    try:
        handler = http.server.SimpleHTTPRequestHandler
        server = socketserver.TCPServer(("127.0.0.1", port), handler)
        server.allow_reuse_address = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        await asyncio.sleep(0.5)
        
        bin_path = require_cloudflared()
        cmd = [str(bin_path), "tunnel", "--url", f"http://127.0.0.1:{port}"]
        print(f"  Command: {' '.join(cmd)}")
        print("  Starting cloudflared (will capture first 10 seconds of output)...")
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        
        output_lines = []
        start = time.time()
        while time.time() - start < 10.0:
            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                print_test("cloudflared exited early", "FAIL")
                print(f"  Exit code: {proc.returncode}")
                
                # Check for rate limiting
                is_rate_limit = stderr and ("429" in stderr and "Too Many Requests" in stderr)
                
                if is_rate_limit:
                    print("\n  ⚠️  RATE LIMIT DETECTED")
                    print("  " + "=" * 70)
                    print("  Cloudflare is blocking quick tunnel creation due to rate limiting.")
                    print("\n  Solutions:")
                    print("    1. ⏰ Wait 5-10 minutes for the rate limit to reset")
                    print("    2. 🔑 Use managed tunnels (set SYNTH_API_KEY env var)")
                    print("    3. ♻️  Reuse existing tunnel (set INTERCEPTOR_TUNNEL_URL env var)")
                    print("  " + "=" * 70)
                
                print(f"\n  STDOUT:\n{stdout}")
                print(f"\n  STDERR:\n{stderr}")
                return False
            
            if proc.stdout:
                line = proc.stdout.readline()
                if line:
                    output_lines.append(line)
                    print(f"  > {line.rstrip()}")
            
            await asyncio.sleep(0.1)
        
        # Kill the process
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        
        if output_lines:
            print_test("cloudflared output captured", "PASS")
            print(f"  Captured {len(output_lines)} lines")
            result = True
        else:
            print_test("No output from cloudflared", "WARN")
            print("  cloudflared started but produced no output in 10 seconds")
            result = False
            
    except Exception as e:
        print_test(f"Error running manual command: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        result = False
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if server_thread:
            server_thread.join(timeout=2.0)
    return result


async def main():
    """Run all integration tests."""
    print("\n" + "=" * 80)
    print("  CLOUDFLARED INTEGRATION TEST SUITE")
    print("=" * 80)
    print("\nThis will test every aspect of cloudflared tunnel creation.")
    print("Each test provides detailed diagnostics.\n")
    
    # Test 1: System info
    test_system_info()
    
    # Test 2: Find binary
    bin_path = test_cloudflared_binary_location()
    if not bin_path:
        print("\n❌ CRITICAL: cloudflared binary not found. Cannot continue.")
        print("   Install cloudflared: brew install cloudflare/cloudflare/cloudflared")
        return 1
    
    # Test 3: Version check
    if not test_cloudflared_version():
        print("\n❌ CRITICAL: cloudflared cannot run. Cannot continue.")
        print("   Try: cloudflared update")
        print("   Or: brew reinstall cloudflare/cloudflare/cloudflared")
        return 1
    
    # Test 4: Port availability
    test_port = 8080
    port_available = test_port_availability(test_port)
    
    # If port 8080 is in use, find an available port
    if not port_available:
        print("\n  Finding available port...")
        test_port = find_available_port(18080)
        print(f"  Using port {test_port} for testing")
    
    # Test 5: Basic tunnel creation
    tunnel_url, tunnel_ok = test_cloudflared_tunnel_creation(test_port)
    
    if tunnel_url:
        # Test 6: DNS resolution
        from urllib.parse import urlparse
        parsed = urlparse(tunnel_url)
        hostname = parsed.hostname
        if hostname:
            await test_dns_resolution(hostname)
            
            # Test 8: HTTP connectivity
            await test_http_connectivity(tunnel_url)
    
    # Test 7: Tunnel with DNS verification
    await test_tunnel_with_dns_verification(test_port)
    
    # Test 8: Rate limit check
    await test_rate_limit_status()
    
    # Test 9: Manual command
    await test_manual_cloudflared_command(test_port)
    
    print_section("TEST SUMMARY")
    print("All tests completed. Review output above for failures.")
    print("\nIf tunnel creation fails:")
    print("  1. Check cloudflared version: cloudflared --version")
    print("  2. Try manual tunnel: cloudflared tunnel --url http://127.0.0.1:8080")
    print("  3. Update cloudflared: cloudflared update")
    print("  4. Reinstall: brew reinstall cloudflare/cloudflare/cloudflared")
    
    return 0


if __name__ == "__main__":
    import socket
    
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

