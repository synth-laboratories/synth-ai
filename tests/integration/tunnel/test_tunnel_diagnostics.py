#!/usr/bin/env python3
"""
Diagnostic script to test Cloudflare tunnel DNS hypotheses.

Tests:
1. Resolver path issue - compares system resolver vs explicit resolvers (1.1.1.1, 8.8.8.8)
2. Tunnel connection status - checks cloudflared logs for actual connection
3. DNS propagation timing - measures how long DNS takes to resolve
"""
import asyncio
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import httpx


def test_dns_resolution(hostname: str, resolver: Optional[str] = None) -> Tuple[bool, str, float]:
    """
    Test DNS resolution for a hostname.
    
    Args:
        hostname: Hostname to resolve
        resolver: Optional DNS server IP (e.g., "1.1.1.1", "8.8.8.8")
                  If None, uses system resolver
    
    Returns:
        Tuple of (success, ip_address_or_error, elapsed_time)
    """
    start = time.time()
    try:
        if resolver:
            # Try dig first, fallback to dnspython if available
            try:
                result = subprocess.run(
                    ["dig", f"@{resolver}", "+short", hostname],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if result.returncode == 0 and result.stdout.strip():
                    ip = result.stdout.strip().split()[0]
                    elapsed = time.time() - start
                    return True, ip, elapsed
                else:
                    elapsed = time.time() - start
                    return False, f"dig returned: {result.stderr or result.stdout}", elapsed
            except FileNotFoundError:
                # dig not available, try dnspython
                try:
                    import dns.resolver
                    resolver_obj = dns.resolver.Resolver()
                    resolver_obj.nameservers = [resolver]
                    answer = resolver_obj.resolve(hostname, 'A', lifetime=5.0)
                    ip = str(answer[0])
                    elapsed = time.time() - start
                    return True, ip, elapsed
                except ImportError:
                    elapsed = time.time() - start
                    return False, "dig not found and dnspython not installed", elapsed
                except Exception as e:
                    elapsed = time.time() - start
                    return False, f"dnspython error: {e}", elapsed
        else:
            # Use system resolver
            ip = socket.gethostbyname(hostname)
            elapsed = time.time() - start
            return True, ip, elapsed
    except socket.gaierror as e:
        elapsed = time.time() - start
        return False, str(e), elapsed
    except Exception as e:
        elapsed = time.time() - start
        return False, str(e), elapsed


def check_cloudflared_logs(process: subprocess.Popen, timeout: float = 30.0) -> dict:
    """
    Check cloudflared process logs for connection status.
    
    Returns dict with:
    - connected: bool - whether tunnel connected to edge
    - registered: bool - whether tunnel registered
    - error: Optional[str] - any error messages
    - logs: list[str] - relevant log lines
    """
    if process.stdout is None:
        return {"connected": False, "registered": False, "error": "No stdout", "logs": []}
    
    start = time.time()
    logs = []
    connected = False
    registered = False
    error = None
    
    # Read available output
    while time.time() - start < timeout:
        if process.poll() is not None:
            # Process exited
            remaining, _ = process.communicate()
            if remaining:
                logs.extend(remaining.splitlines())
            break
        
        try:
            line = process.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            
            line = line.strip()
            logs.append(line)
            
            # Check for connection indicators
            if "connected" in line.lower() or "connection established" in line.lower():
                connected = True
            if "registered" in line.lower() or "route registered" in line.lower():
                registered = True
            if ("error" in line.lower() or "failed" in line.lower()) and not error:
                error = line
            
            # Stop if we see both connection and registration
            if connected and registered:
                break
        except Exception:
            break
    
    return {
        "connected": connected,
        "registered": registered,
        "error": error,
        "logs": logs[-20:],  # Last 20 lines
    }


async def test_tunnel_diagnostics():
    """Run comprehensive tunnel diagnostics."""
    print("=" * 80)
    print("CLOUDFLARE TUNNEL DNS DIAGNOSTICS")
    print("=" * 80)
    print()
    
    # Step 1: Create a tunnel
    print("[1/4] Creating Cloudflare tunnel...")
    from synth_ai.cloudflare import open_quick_tunnel
    
    test_port = 9999
    # Start a dummy server on the port
    import http.server
    import socketserver
    import threading
    
    server = socketserver.TCPServer(("127.0.0.1", test_port), http.server.SimpleHTTPRequestHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    
    try:
        tunnel_url, tunnel_proc = open_quick_tunnel(test_port, wait_s=15.0)
        parsed = urlparse(tunnel_url)
        hostname = parsed.hostname
        
        print(f"✓ Tunnel created: {tunnel_url}")
        print(f"  Hostname: {hostname}")
        print(f"  Process PID: {tunnel_proc.pid}")
        print()
        
        # Step 2: Check cloudflared logs for connection status
        print("[2/4] Checking cloudflared connection status...")
        log_info = check_cloudflared_logs(tunnel_proc, timeout=10.0)
        print(f"  Connected to edge: {log_info['connected']}")
        print(f"  Route registered: {log_info['registered']}")
        if log_info['error']:
            print(f"  Error in logs: {log_info['error']}")
        if log_info['logs']:
            print(f"  Recent logs ({len(log_info['logs'])} lines):")
            for log_line in log_info['logs'][-5:]:
                print(f"    {log_line}")
        print()
        
        # Step 3: Test DNS resolution with different resolvers
        print("[3/4] Testing DNS resolution with different resolvers...")
        resolvers = {
            "System": None,
            "Cloudflare (1.1.1.1)": "1.1.1.1",
            "Google (8.8.8.8)": "8.8.8.8",
        }
        
        results = {}
        for name, resolver_ip in resolvers.items():
            success, result, elapsed = test_dns_resolution(hostname, resolver_ip)
            results[name] = (success, result, elapsed)
            status = "✓" if success else "✗"
            print(f"  {status} {name}: {result} ({elapsed:.3f}s)")
        
        print()
        
        # Step 4: Test HTTP connectivity
        print("[4/4] Testing HTTP connectivity...")
        max_wait = 60.0
        start_time = time.time()
        http_success = False
        
        while time.time() - start_time < max_wait:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{parsed.scheme}://{hostname}/", follow_redirects=True)
                    if resp.status_code in (200, 404, 405):
                        http_success = True
                        elapsed = time.time() - start_time
                        print(f"  ✓ HTTP connectivity successful after {elapsed:.1f}s")
                        print(f"    Status: {resp.status_code}")
                        break
            except Exception:
                await asyncio.sleep(1.0)
        
        if not http_success:
            elapsed = time.time() - start_time
            print(f"  ✗ HTTP connectivity failed after {elapsed:.1f}s")
        
        print()
        print("=" * 80)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 80)
        print()
        print(f"Tunnel URL: {tunnel_url}")
        print(f"Hostname: {hostname}")
        print()
        print("DNS Resolution Results:")
        for name, (success, result, elapsed) in results.items():
            status = "✓ RESOLVED" if success else "✗ FAILED"
            print(f"  {name:25s} {status:15s} ({elapsed:.3f}s) - {result}")
        print()
        print(f"Cloudflared Connection: {'✓ Connected' if log_info['connected'] else '✗ Not connected'}")
        print(f"Route Registered: {'✓ Registered' if log_info['registered'] else '✗ Not registered'}")
        print(f"HTTP Connectivity: {'✓ Working' if http_success else '✗ Failed'}")
        print()
        
        # Analysis
        print("HYPOTHESIS ANALYSIS:")
        print("-" * 80)
        
        system_resolved = results["System"][0]
        cloudflare_resolved = results["Cloudflare (1.1.1.1)"][0]
        google_resolved = results["Google (8.8.8.8)"][0]
        
        if not system_resolved and (cloudflare_resolved or google_resolved):
            print("✓ CONFIRMED: Resolver path issue")
            print("  System resolver cannot resolve, but explicit resolvers can.")
            print("  This suggests your system DNS (ISP/corporate/VPN) is slow or blocking.")
        elif not system_resolved and not cloudflare_resolved and not google_resolved:
            print("✓ CONFIRMED: Global DNS propagation delay")
            print("  No resolver can resolve the hostname yet.")
            print("  This is a Cloudflare-side delay, not a local resolver issue.")
        else:
            print("✓ DNS resolution working across all resolvers")
        
        if not log_info['connected'] or not log_info['registered']:
            print("✓ CONFIRMED: Tunnel not fully connected")
            print("  cloudflared process started but may not have connected to Cloudflare edge.")
            print("  This could explain why DNS never propagates.")
        else:
            print("✓ Tunnel connection confirmed in logs")
        
        if http_success:
            print("✓ HTTP connectivity working - tunnel is functional")
        else:
            print("✗ HTTP connectivity failed - tunnel may not be working even if DNS resolves")
        
        print()
        
        # Write results to file
        output_file = Path("tunnel_diagnostics_results.txt")
        with open(output_file, "w") as f:
            f.write("CLOUDFLARE TUNNEL DNS DIAGNOSTICS RESULTS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Tunnel URL: {tunnel_url}\n")
            f.write(f"Hostname: {hostname}\n\n")
            f.write("DNS Resolution Results:\n")
            for name, (success, result, elapsed) in results.items():
                status = "RESOLVED" if success else "FAILED"
                f.write(f"  {name:25s} {status:15s} ({elapsed:.3f}s) - {result}\n")
            f.write(f"\nCloudflared Connection: {'Connected' if log_info['connected'] else 'Not connected'}\n")
            f.write(f"Route Registered: {'Registered' if log_info['registered'] else 'Not registered'}\n")
            f.write(f"HTTP Connectivity: {'Working' if http_success else 'Failed'}\n")
            if log_info['logs']:
                f.write("\nRecent cloudflared logs:\n")
                for log_line in log_info['logs']:
                    f.write(f"  {log_line}\n")
        
        print(f"✓ Results written to: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during diagnostics: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'tunnel_proc' in locals() and tunnel_proc:
            tunnel_proc.terminate()
            try:
                tunnel_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                tunnel_proc.kill()
        if 'server' in locals():
            server.shutdown()
        print("✓ Cleanup complete")


if __name__ == "__main__":
    asyncio.run(test_tunnel_diagnostics())

