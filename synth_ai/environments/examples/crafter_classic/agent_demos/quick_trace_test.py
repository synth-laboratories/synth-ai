#!/usr/bin/env python3
"""
Quick test to verify LM class produces proper v2 traces.
This is a minimal test that doesn't require the full Crafter environment.
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime

# Disable v1 tracing
os.environ["LANGFUSE_ENABLED"] = "false"
os.environ["SYNTH_LOGGING"] = "false"
os.environ["SYNTH_TRACING_MODE"] = "v2"

from synth_ai.lm.core.main_v2 import LM
from synth_ai.tracing_v2.session_tracer import SessionTracer


async def test_lm_tracing():
    """Test that LM class creates proper v2 traces."""
    print("🧪 Testing LM V2 Tracing\n")
    
    # Create tracer
    tracer = SessionTracer()
    
    # Create LM with v2 tracing
    lm = LM(
        model_name="gpt-3.5-turbo",
        formatting_model_name="gpt-3.5-turbo",
        temperature=0,
        synth_logging=False,  # Disable v1
        session_tracer=tracer,
        system_id="test_agent",
        enable_v2_tracing=True
    )
    
    # Run a simple session
    session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Start session (not async context manager)
    tracer.start_session(session_id)
    
    print("📝 Making LM call...")
    
    # Make a simple call
    response = await lm.respond_async(
        system_message="You are a helpful assistant. Answer in one sentence.",
        user_message="What is the capital of France?",
        turn_number=0
    )
    
    print(f"Response: {response.raw_response}")
    print()
    
    # End session (saves automatically)
    trace_path = tracer.end_session()
    print(f"💾 Saved trace to {trace_path}")
    
    # Use the saved trace file
    trace_file = trace_path if trace_path else Path("./traces") / f"{session_id}.json"
    
    # Load and inspect the trace
    with open(trace_file, 'r') as f:
        trace_data = json.load(f)
    
    # Check for CAISEvents
    events = trace_data.get('event_history', [])
    cais_events = [e for e in events if 'llm_call_records' in e]
    
    print(f"\n📊 Trace Analysis:")
    print(f"  Total events: {len(events)}")
    print(f"  CAIS events: {len(cais_events)}")
    
    if cais_events:
        print(f"\n✅ Found CAIS event with v2 tracing!")
        event = cais_events[0]
        
        # Check key fields
        print(f"  System ID: {event.get('system_instance_id', 'N/A')}")
        print(f"  Has LLM records: {'llm_call_records' in event}")
        
        if 'llm_call_records' in event and event['llm_call_records']:
            llm_record = event['llm_call_records'][0]
            print(f"  Model: {llm_record.get('model', 'N/A')}")
            
            # Check response
            if 'response' in llm_record:
                usage = llm_record['response'].get('usage', {})
                if usage:
                    print(f"  Tokens: {usage.get('total_tokens', 'N/A')}")
        
        # Check metadata
        metadata = event.get('metadata', {})
        if metadata:
            print(f"  Turn: {metadata.get('turn', 'N/A')}")
            print(f"  Model in metadata: {metadata.get('model_name', 'N/A')}")
        
        print(f"\n✅ LM class successfully creates v2 traces!")
    else:
        print(f"\n❌ No CAIS events found in trace")
    
    # Show trace structure
    print(f"\n📋 Trace Structure:")
    metadata = trace_data.get('session_metadata', {})
    if isinstance(metadata, dict):
        print(f"  - session_metadata: {list(metadata.keys())}")
    else:
        print(f"  - session_metadata: {type(metadata).__name__}")
    print(f"  - event_history: {len(trace_data.get('event_history', []))} events")
    print(f"  - message_history: {len(trace_data.get('message_history', []))} messages")
    
    # Clean up
    if trace_file.exists():
        trace_file.unlink()
        print(f"\n🧹 Cleaned up test trace file")


if __name__ == "__main__":
    print("="*60)
    print("LM V2 Tracing Quick Test")
    print("="*60)
    
    try:
        asyncio.run(test_lm_tracing())
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()