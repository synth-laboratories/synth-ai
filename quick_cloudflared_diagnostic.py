#!/usr/bin/env python3
"""
Quick diagnostic for cloudflared issues.
Run this to get immediate feedback on what's wrong.
"""

import subprocess
import sys
from pathlib import Path

from synth_ai.cloudflare import get_cloudflared_path, require_cloudflared

print("🔍 Quick Cloudflared Diagnostic\n")

# 1. Find binary
print("1. Finding cloudflared binary...")
try:
    bin_path = require_cloudflared()
    print(f"   ✅ Found: {bin_path}")
except Exception as e:
    print(f"   ❌ Not found: {e}")
    sys.exit(1)

# 2. Test --version
print("\n2. Testing cloudflared --version...")
try:
    proc = subprocess.run(
        [str(bin_path), "--version"],
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if proc.returncode == 0:
        print(f"   ✅ Works! Version:\n{proc.stdout}")
    else:
        print(f"   ❌ Failed with exit code {proc.returncode}")
        print(f"   STDOUT: {proc.stdout}")
        print(f"   STDERR: {proc.stderr}")
        sys.exit(1)
except subprocess.TimeoutExpired:
    print("   ❌ Timed out - binary is hanging")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# 3. Test tunnel command with a real server
print("\n3. Testing tunnel command with real HTTP server...")
import http.server
import socketserver
import threading
import time

test_port = 18080  # Use a high port to avoid conflicts
server = None
server_thread = None

try:
    # Start a simple HTTP server
    handler = http.server.SimpleHTTPRequestHandler
    server = socketserver.TCPServer(("127.0.0.1", test_port), handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"   Started HTTP server on port {test_port}")
    time.sleep(0.5)  # Give server time to start
    
    # Now try tunnel
    proc = subprocess.Popen(
        [str(bin_path), "tunnel", "--url", f"http://127.0.0.1:{test_port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    import time
    import select
    
    output_lines = []
    start = time.time()
    while time.time() - start < 5.0:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            print(f"   ⚠️  Process exited early (code {proc.returncode})")
            if stdout:
                print(f"   STDOUT (full):\n{stdout}")
            if stderr:
                print(f"   STDERR (full):\n{stderr}")
            break
        
        # Try to read output
        if proc.stdout:
            import fcntl
            import os
            try:
                # Set non-blocking
                flags = fcntl.fcntl(proc.stdout.fileno(), fcntl.F_GETFL)
                fcntl.fcntl(proc.stdout.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
                line = proc.stdout.readline()
                if line:
                    output_lines.append(line)
                    print(f"   Output: {line.rstrip()}")
            except:
                pass
        
        time.sleep(0.1)
    
    proc.terminate()
    try:
        proc.wait(timeout=2.0)
    except:
        proc.kill()
    
    if output_lines:
        print(f"   ✅ cloudflared started and produced output ({len(output_lines)} lines)")
        # Look for URL
        import re
        url_re = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)
        for line in output_lines:
            match = url_re.search(line)
            if match:
                print(f"   ✅ Found tunnel URL: {match.group(0)}")
    else:
        print("   ⚠️  cloudflared started but no output in 5 seconds")
        print("   This might indicate an issue")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    if server:
        server.shutdown()
        server.server_close()
    if server_thread:
        server_thread.join(timeout=2.0)

print("\n✅ Basic checks passed! cloudflared appears to be working.")
print("   If tunnels still fail, run the full integration test:")
print("   python test_cloudflared_integration.py")

