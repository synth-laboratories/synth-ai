# Crafter Environment Slowness Fix

## Root Cause Analysis

The 5-7 second "environment time" is NOT due to the service being slow. Direct tests show the service responds in 15-30ms per request. The slowness is caused by:

1. **HTTP Retry Mechanism**: When requests fail, exponential backoff adds:
   - 1st retry: 1 second delay
   - 2nd retry: 2 seconds delay  
   - Total: ~3+ seconds of retry delays

2. **Intermittent Failures**: Something is causing occasional HTTP failures that trigger retries

3. **Large Response Payloads**: 62KB per response (mostly the observation_image field)

## Immediate Fixes

### 1. Reduce Retry Delays (Quick Fix)
In `test_crafter_react_agent_openai.py`, change:
```python
# Line 78-81
MAX_RETRIES = 1  # Was 3
BASE_DELAY = 0.1  # Was 1.0
MAX_DELAY = 0.5   # Was 10.0
HTTP_TIMEOUT = 5.0  # Was 30.0
```

### 2. Add Retry Logging
Add logging to see when retries happen:
```python
# In retry_http_request function, after line 101
if attempt > 0:
    print(f"    ⚠️  Retry {attempt} with {delay:.1f}s delay")
```

### 3. Diagnose Network Issues
Check for:
- Firewall/proxy interference
- Port conflicts (we found and fixed one already)
- Service overload with concurrent requests

## Long-term Fixes

### 1. Optimize Response Size
- Remove `observation_image` from responses (57KB)
- Use compression (gzip)
- Only send changed fields

### 2. Use Connection Pooling
- Reuse HTTP connections
- Consider using a persistent WebSocket

### 3. Fix Service Stability
- Add better error handling in the service
- Use proper async handling for concurrent requests
- Consider using multiple workers

## Verification

Run this to verify the fix works:
```bash
# Test with reduced retry delays
python -m synth_ai.environments.examples.crafter_classic.agent_demos.test_crafter_react_agent_openai \
    --model gpt-4o-mini \
    --episodes 1 \
    --max-turns 5
```

Watch for:
- Environment time should be <1s per step
- No retry messages in output
- Consistent fast performance