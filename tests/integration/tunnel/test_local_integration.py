#!/usr/bin/env python3
"""
Local-only integration test suite for synth-ai (SYNTH DEVELOPERS ONLY).

⚠️  WARNING: This test suite is designed EXCLUSIVELY for synth developers working
    on the synth-ai codebase. It tests local-only functionality without requiring
    Cloudflare tunnels or external services.

This script tests:
- Local HTTP server functionality
- Port availability and binding
- Localhost DNS resolution (should always work)
- HTTP connectivity to local services
- Task app health checks (local only)

DO NOT USE THIS FOR PRODUCTION TESTING OR CUSTOMER-FACING FUNCTIONALITY.
This is for internal development and debugging only.
"""

import asyncio
import http.server
import logging
import os
import platform
import socket
import socketserver
import sys
import threading
import time
from typing import Optional, Tuple

import httpx

# Setup logging
logging.basicConfig(
    level=logging.INFO,  # Less verbose than cloudflared tests
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Import synth-ai modules
try:
    from synth_ai.api.train.task_app import check_task_app_health
    from synth_ai.cloudflare import resolve_hostname_with_explicit_resolvers
except ImportError as e:
    logger.error(f"Failed to import synth_ai modules: {e}")
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
    print(f"User: {os.getenv('USER', 'unknown')}")


def find_available_port(start_port: int = 18000) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find available port starting from {start_port}")


def test_port_availability(port: int = 8080) -> bool:
    """Test 2: Check if a port is available."""
    print_section(f"TEST 2: Port {port} Availability")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))
            print_test("Port available", "PASS")
            print(f"  Port {port} is available for binding")
            return True
    except OSError as e:
        print_test("Port in use", "WARN")
        print(f"  Port {port} is already in use: {e}")
        return False


def test_local_http_server(port: int = 8080) -> Tuple[Optional[socketserver.TCPServer], Optional[threading.Thread]]:
    """Test 3: Start a local HTTP server."""
    print_section(f"TEST 3: Local HTTP Server (port {port})")
    
    server = None
    server_thread = None
    
    try:
        handler = http.server.SimpleHTTPRequestHandler
        server = socketserver.TCPServer(("127.0.0.1", port), handler)
        server.allow_reuse_address = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        time.sleep(0.5)  # Give server time to start
        
        # Test connectivity
        try:
            response = httpx.get(f"http://127.0.0.1:{port}", timeout=2.0)
            print_test("HTTP server started", "PASS")
            print(f"  Server responding on http://127.0.0.1:{port}")
            print(f"  Status: {response.status_code}")
            return server, server_thread
        except Exception as e:
            print_test("HTTP server failed", "FAIL")
            print(f"  Error: {e}")
            if server:
                server.shutdown()
                server.server_close()
            return None, None
    except Exception as e:
        print_test("Failed to start server", "FAIL")
        print(f"  Error: {e}")
        return None, None


async def test_localhost_dns_resolution():
    """Test 4: DNS resolution for localhost."""
    print_section("TEST 4: Localhost DNS Resolution")
    try:
        # Test localhost resolution
        ip = await resolve_hostname_with_explicit_resolvers("localhost")
        print_test("localhost DNS resolution", "PASS")
        print(f"  localhost -> {ip}")
        
        # Test 127.0.0.1 (should be immediate)
        ip2 = await resolve_hostname_with_explicit_resolvers("127.0.0.1")
        print_test("127.0.0.1 DNS resolution", "PASS")
        print(f"  127.0.0.1 -> {ip2}")
        
        return True
    except Exception as e:
        print_test("DNS resolution failed", "FAIL")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_local_http_connectivity(port: Optional[int] = None):
    """Test 5: HTTP connectivity to local server."""
    if port is None:
        port = find_available_port(18100)
    print_section(f"TEST 5: Local HTTP Connectivity (port {port})")
    
    # Start server
    server, server_thread = test_local_http_server(port)
    if not server:
        print_test("Skipping connectivity test", "SKIP")
        return False
    
    try:
        # Test various endpoints
        base_url = f"http://127.0.0.1:{port}"
        
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Test root
            response = await client.get(base_url)
            print_test("HTTP GET request", "PASS")
            print(f"  URL: {base_url}")
            print(f"  Status: {response.status_code}")
            print(f"  Headers: {dict(response.headers)}")
            
            # Test with localhost hostname
            localhost_url = f"http://localhost:{port}"
            response2 = await client.get(localhost_url)
            print_test("HTTP GET with localhost", "PASS")
            print(f"  URL: {localhost_url}")
            print(f"  Status: {response2.status_code}")
            
            return True
    except Exception as e:
        print_test("HTTP connectivity failed", "FAIL")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if server_thread:
            server_thread.join(timeout=2.0)


async def test_task_app_health_check_local(port: Optional[int] = None):
    """Test 6: Task app health check (local only)."""
    if port is None:
        port = find_available_port(18200)
    print_section(f"TEST 6: Task App Health Check (Local, port {port})")
    
    # Start a simple HTTP server that responds to /health and /health/task_info
    class HealthCheckHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            elif self.path == "/health/task_info":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"task_name": "test-task", "version": "1.0.0"}')
            else:
                self.send_response(404)
                self.end_headers()
    
    server = None
    server_thread = None
    
    try:
        server = socketserver.TCPServer(("127.0.0.1", port), HealthCheckHandler)
        server.allow_reuse_address = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        time.sleep(0.5)
        
        # Test health check
        health_url = f"http://127.0.0.1:{port}/health"
        api_key = "test-api-key-12345"
        
        try:
            # check_task_app_health is synchronous, not async
            health_result = check_task_app_health(health_url, api_key)
            print_test("Task app health check", "PASS")
            print(f"  URL: {health_url}")
            print(f"  OK: {health_result.ok}")
            print(f"  Health status: {health_result.health_status}")
            print(f"  Task info status: {health_result.task_info_status}")
            if health_result.detail:
                print(f"  Detail: {health_result.detail}")
            return health_result.ok
        except Exception as e:
            print_test("Task app health check failed", "FAIL")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if server_thread:
            server_thread.join(timeout=2.0)


async def test_multiple_local_servers():
    """Test 7: Multiple local servers on different ports."""
    print_section("TEST 7: Multiple Local Servers")
    
    ports = []
    servers = []
    threads = []
    
    try:
        # Start 3 servers on different ports
        for i in range(3):
            port = find_available_port(18000 + i * 10)
            ports.append(port)
            
            handler = http.server.SimpleHTTPRequestHandler
            server = socketserver.TCPServer(("127.0.0.1", port), handler)
            server.allow_reuse_address = True
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            servers.append(server)
            threads.append(thread)
            time.sleep(0.2)
        
        # Verify all servers are running
        async with httpx.AsyncClient(timeout=5.0) as client:
            all_ok = True
            for port in ports:
                try:
                    response = await client.get(f"http://127.0.0.1:{port}")
                    if response.status_code != 200:
                        all_ok = False
                except Exception:
                    all_ok = False
            
            if all_ok:
                print_test("Multiple servers running", "PASS")
                print(f"  Started {len(ports)} servers on ports: {ports}")
                return True
            else:
                print_test("Some servers failed", "FAIL")
                return False
    except Exception as e:
        print_test("Failed to start multiple servers", "FAIL")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        for server in servers:
            try:
                server.shutdown()
                server.server_close()
            except Exception:
                pass
        for thread in threads:
            thread.join(timeout=1.0)


async def main():
    """Run all local-only integration tests."""
    print("\n" + "=" * 80)
    print("  LOCAL-ONLY INTEGRATION TEST SUITE (SYNTH DEVELOPERS ONLY)")
    print("=" * 80 + "\n")
    print("⚠️  WARNING: This test suite is for synth developers only.")
    print("   It tests local functionality without Cloudflare tunnels.")
    print("   DO NOT USE FOR PRODUCTION TESTING.\n")
    
    # Test 1: System Information
    test_system_info()
    
    # Test 2: Port availability
    test_port = 8080
    port_available = test_port_availability(test_port)
    
    # If port 8080 is in use, find an available port
    if not port_available:
        print("\n  Finding available port...")
        test_port = find_available_port(18000)
        print(f"  Using port {test_port} for testing")
    
    # Test 3: Local HTTP server
    server, server_thread = test_local_http_server(test_port)
    if server:
        # Clean up
        server.shutdown()
        server.server_close()
        if server_thread:
            server_thread.join(timeout=2.0)
        # Give port time to be released
        time.sleep(0.5)
    
    # Test 4: Localhost DNS resolution
    await test_localhost_dns_resolution()
    
    # Test 5: HTTP connectivity (uses its own port)
    await test_local_http_connectivity()
    
    # Test 6: Task app health check (uses its own port)
    await test_task_app_health_check_local()
    
    # Test 7: Multiple servers
    await test_multiple_local_servers()
    
    print_section("TEST SUMMARY")
    print("All local-only tests completed.")
    print("\nThis test suite verifies:")
    print("  ✅ Local HTTP server functionality")
    print("  ✅ Port availability and binding")
    print("  ✅ Localhost DNS resolution")
    print("  ✅ HTTP connectivity to local services")
    print("  ✅ Task app health checks (local)")
    print("\nNote: These tests do NOT require Cloudflare tunnels or external services.")
    print("For Cloudflare tunnel testing, use test_cloudflared_tunnel_integration.py")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

