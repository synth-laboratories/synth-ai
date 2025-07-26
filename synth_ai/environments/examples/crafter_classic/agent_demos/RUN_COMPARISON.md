# Running Crafter Trace Comparison

This guide explains how to run the Crafter evaluation with both OpenAI direct API and LM class approaches to verify v2 tracing equivalence.

## Prerequisites

1. **Crafter service running** on port 8901:
   ```bash
   cd synth_ai/environments/service
   python app.py --port 8901
   ```

2. **OpenAI API key** set:
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

## Quick Test

### 1. Run the quick LM trace test (no Crafter needed):
```bash
cd synth_ai/environments/examples/crafter_classic/agent_demos
python quick_trace_test.py
```

This verifies that the LM class creates proper v2 traces.

### 2. Run the automated comparison:
```bash
python compare_traces.py
```

This runs both versions and compares the traces automatically.

## Manual Comparison

### 1. Run OpenAI version:
```bash
python test_crafter_react_agent_openai.py \
    --model gpt-3.5-turbo \
    --episodes 1 \
    --max-turns 2 \
    --verbose
```

Traces saved to: `./traces/`

### 2. Run LM version:
```bash
python test_crafter_react_agent_lm.py \
    --model gpt-3.5-turbo \
    --episodes 1 \
    --max-turns 2 \
    --verbose
```

Traces saved to: `./traces_v2_lm/`

### 3. Compare traces manually:
```bash
# View OpenAI trace
cat traces/trace_episode_0.json | jq '.event_history[] | select(.llm_call_records != null)'

# View LM trace  
cat traces_v2_lm/trace_episode_0.json | jq '.event_history[] | select(.llm_call_records != null)'
```

## What to Look For

### ✅ Both traces should have:
- **CAISEvents** with `llm_call_records` containing:
  - Model name
  - Messages sent
  - Response with tool calls
  - Token usage
- **Same number of events** (within reason - LLM responses may vary)
- **Same system_instance_id pattern** (`crafter-react-agent-*`)
- **Turn numbers** in metadata

### ⚠️  Expected differences:
- **Response content** - LLMs are non-deterministic
- **Exact tool calls** - Agent might choose different actions
- **Token counts** - May vary slightly between runs
- **Timestamps** - Will obviously differ

## Example Comparison Output

```
=============================================================
Comparing traces for episode 0
=============================================================

Comparing CAISEvent...

  Event 1:
    ✓ Model matches: gpt-3.5-turbo
    ✓ system_instance_id matches
    Tokens (OpenAI): {'prompt_tokens': 245, 'completion_tokens': 32, 'total_tokens': 277}
    Tokens (LM):     {'prompt_tokens': 245, 'completion_tokens': 32, 'total_tokens': 277}

Comparing messages...
✓ Same number of messages: 4

=============================================================
FINAL VERDICT
=============================================================
✅ Traces match! LM class produces equivalent v2 traces.
```

## Troubleshooting

### "No CAIS events found"
- Check that v2 tracing is enabled: `SYNTH_TRACING_MODE=v2` or `dual`
- Verify the LM class has `enable_v2_tracing=True`
- Check that a SessionTracer was provided

### "Different number of events"
- This is often due to LLM non-determinism
- Check if one agent terminated early
- Compare the actual decisions made

### "Module not found" errors
- Make sure you're in the correct directory
- Add the project root to PYTHONPATH if needed:
  ```bash
  export PYTHONPATH=/path/to/synth-ai:$PYTHONPATH
  ```

## Summary

The LM class with v2 tracing should produce traces that are structurally identical to the OpenAI direct API version. The main benefits:

1. **Cleaner code** - No manual event creation
2. **Automatic tracking** - Decorators handle all tracing
3. **Provider agnostic** - Works with any LLM provider
4. **Cost tracking** - Automatic cost calculation
5. **PII safety** - Built-in masking capabilities