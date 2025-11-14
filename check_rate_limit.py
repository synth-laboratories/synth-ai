#!/usr/bin/env python3
"""
Check Cloudflare quick tunnel rate limit status.

This attempts to create a tunnel and checks for rate limit errors.
"""

import subprocess
import sys
import time
from pathlib import Path

from synth_ai.cloudflare import require_cloudflared


def check_rate_limit_status():
    """Check if we're currently rate limited."""
    print("🔍 Checking Cloudflare Quick Tunnel Rate Limit Status\n")
    
    bin_path = require_cloudflared()
    print(f"Using cloudflared: {bin_path}\n")
    
    # Start a dummy HTTP server on a high port
    import http.server
    import socketserver
    import threading
    
    test_port = 19999
    server = None
    server_thread = None
    
    try:
        handler = http.server.SimpleHTTPRequestHandler
        server = socketserver.TCPServer(("127.0.0.1", test_port), handler)
        server.allow_reuse_address = True
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        print(f"Started test server on port {test_port}")
        time.sleep(0.5)
        
        # Try to create a tunnel
        print("Attempting to create quick tunnel...")
        proc = subprocess.Popen(
            [str(bin_path), "tunnel", "--url", f"http://127.0.0.1:{test_port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Wait a few seconds to see if it succeeds or fails
        start = time.time()
        output_lines = []
        stderr_lines = []
        
        while time.time() - start < 5.0:
            if proc.poll() is not None:
                # Process exited
                stdout, stderr = proc.communicate()
                if stdout:
                    output_lines.extend(stdout.splitlines())
                if stderr:
                    stderr_lines.extend(stderr.splitlines())
                break
            
            # Try to read output
            if proc.stdout:
                try:
                    import fcntl
                    import os
                    flags = fcntl.fcntl(proc.stdout.fileno(), fcntl.F_GETFL)
                    fcntl.fcntl(proc.stdout.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
                    line = proc.stdout.readline()
                    if line:
                        output_lines.append(line.rstrip())
                        print(f"  Output: {line.rstrip()}")
                except:
                    pass
            
            if proc.stderr:
                try:
                    import fcntl
                    import os
                    flags = fcntl.fcntl(proc.stderr.fileno(), fcntl.F_GETFL)
                    fcntl.fcntl(proc.stderr.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
                    line = proc.stderr.readline()
                    if line:
                        stderr_lines.append(line.rstrip())
                except:
                    pass
            
            time.sleep(0.1)
        
        # Clean up
        proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except:
            proc.kill()
        
        # Analyze results
        print("\n" + "=" * 80)
        print("RATE LIMIT STATUS")
        print("=" * 80)
        
        all_output = "\n".join(stderr_lines + output_lines)
        
        # Check for rate limit indicators
        is_rate_limited = False
        rate_limit_info = {}
        
        if proc.returncode == 1:
            if "429" in all_output and "Too Many Requests" in all_output:
                is_rate_limited = True
                print("❌ RATE LIMITED: Cloudflare is currently rate-limiting quick tunnel creation")
                print("\nThis means:")
                print("  • Too many quick tunnels have been created recently")
                print("  • You need to wait before creating another tunnel")
                print("  • Typical reset time: 5-10 minutes")
            elif "rate limit" in all_output.lower():
                is_rate_limited = True
                print("❌ RATE LIMITED: Rate limit detected in output")
            else:
                print(f"⚠️  Tunnel creation failed (exit code {proc.returncode})")
                print("   This might be a rate limit or another issue")
        elif proc.returncode is None:
            # Process still running - might have succeeded
            print("✅ Tunnel creation appears to be in progress")
            print("   (Process is still running after 5 seconds)")
            print("   This suggests rate limits are NOT currently active")
            
            # Look for URL in output
            import re
            url_re = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)
            for line in output_lines + stderr_lines:
                match = url_re.search(line)
                if match:
                    print(f"   Found tunnel URL: {match.group(0)}")
                    break
        else:
            print(f"✅ Tunnel creation completed (exit code {proc.returncode})")
        
        if stderr_lines:
            print("\nSTDERR Output:")
            for line in stderr_lines:
                print(f"  {line}")
        
        if output_lines:
            print("\nSTDOUT Output:")
            for line in output_lines:
                print(f"  {line}")
        
        # Extract rate limit info if available
        if is_rate_limited:
            print("\n" + "=" * 80)
            print("RECOMMENDATIONS")
            print("=" * 80)
            print("1. Wait 5-10 minutes before trying again")
            print("2. Use a managed tunnel instead (requires Cloudflare account)")
            print("3. Set INTERCEPTOR_TUNNEL_URL env var to reuse an existing tunnel")
            print("4. Reduce the frequency of tunnel creation")
        
        return is_rate_limited
        
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if server_thread:
            server_thread.join(timeout=2.0)


if __name__ == "__main__":
    try:
        is_limited = check_rate_limit_status()
        sys.exit(1 if is_limited else 0)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

