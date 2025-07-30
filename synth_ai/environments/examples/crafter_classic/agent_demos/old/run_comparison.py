#!/usr/bin/env python3
"""
Run comparison between OpenAI and LM implementations to verify trace equivalence.
"""

import asyncio
import subprocess
import json
from pathlib import Path
import sys

async def run_tests():
    """Run both tests and compare results."""
    print("🚀 Running Crafter v2 tracing comparison test")
    print("=" * 80)
    
    # Test parameters
    model = "gpt-4o-mini"
    episodes = 2
    max_turns = 5
    
    # Run OpenAI implementation
    print("\n📍 Running OpenAI implementation...")
    openai_cmd = [
        sys.executable,
        "synth_ai/environments/examples/crafter_classic/agent_demos/test_crafter_react_agent_openai.py",
        "--episodes", str(episodes),
        "--model", model,
        "--max-turns", str(max_turns)
    ]
    openai_result = subprocess.run(openai_cmd, capture_output=True, text=True)
    
    if openai_result.returncode != 0:
        print(f"❌ OpenAI test failed: {openai_result.stderr}")
        return
    
    print("✅ OpenAI test completed")
    
    # Run LM implementation
    print("\n📍 Running LM implementation...")
    lm_cmd = [
        sys.executable,
        "synth_ai/environments/examples/crafter_classic/agent_demos/test_crafter_react_agent_lm.py",
        "--episodes", str(episodes),
        "--model", model,
        "--max-turns", str(max_turns)
    ]
    lm_result = subprocess.run(lm_cmd, capture_output=True, text=True)
    
    if lm_result.returncode != 0:
        print(f"❌ LM test failed: {lm_result.stderr}")
        return
    
    print("✅ LM test completed")
    
    # Compare results
    print("\n📊 Comparing results...")
    
    # Load OpenAI results
    openai_results_path = Path("traces/results.json")
    if openai_results_path.exists():
        with open(openai_results_path) as f:
            openai_results = json.load(f)
        print(f"\nOpenAI Results:")
        print(f"  Episodes: {openai_results['summary']['successful']}/{episodes}")
        print(f"  Avg Reward: {openai_results['summary']['avg_reward']:.2f}")
        print(f"  Avg Steps: {openai_results['summary']['avg_steps']:.1f}")
    else:
        print("❌ OpenAI results not found")
        
    # Load LM results
    lm_results_path = Path("traces_v2_lm/results.json")
    if lm_results_path.exists():
        with open(lm_results_path) as f:
            lm_results = json.load(f)
        print(f"\nLM Results:")
        print(f"  Episodes: {lm_results['summary']['successful']}/{episodes}")
        print(f"  Avg Reward: {lm_results['summary']['avg_reward']:.2f}")
        print(f"  Avg Steps: {lm_results['summary']['avg_steps']:.1f}")
    else:
        print("❌ LM results not found")
    
    # Compare trace structures
    print("\n🔍 Comparing trace structures...")
    
    openai_trace = Path("traces/trace_episode_0.json")
    lm_trace = Path("traces_v2_lm/trace_episode_0.json")
    
    if openai_trace.exists() and lm_trace.exists():
        with open(openai_trace) as f:
            openai_data = json.load(f)
        with open(lm_trace) as f:
            lm_data = json.load(f)
            
        # Check key structures
        print(f"\nOpenAI trace:")
        print(f"  Messages: {len(openai_data.get('message_history', []))}")
        print(f"  Events: {len(openai_data.get('event_history', []))}")
        print(f"  Timesteps: {len(openai_data.get('session_time_steps', []))}")
        
        print(f"\nLM trace:")
        print(f"  Messages: {len(lm_data.get('message_history', []))}")
        print(f"  Events: {len(lm_data.get('event_history', []))}")
        print(f"  Timesteps: {len(lm_data.get('session_time_steps', []))}")
        
        # Check for AI events
        openai_ai_events = [e for e in openai_data.get('event_history', []) 
                           if 'gen_ai.request.model' in e.get('system_state_before', {})]
        lm_ai_events = [e for e in lm_data.get('event_history', [])
                       if 'gen_ai.request.model' in e.get('system_state_before', {})]
        
        print(f"\nAI Events:")
        print(f"  OpenAI: {len(openai_ai_events)}")
        print(f"  LM: {len(lm_ai_events)}")
        
        if len(openai_ai_events) == len(lm_ai_events):
            print("✅ Same number of AI events captured")
        else:
            print("⚠️  Different number of AI events")
            
    else:
        print("❌ Trace files not found")
    
    print("\n" + "=" * 80)
    print("✅ Comparison complete!")

if __name__ == "__main__":
    asyncio.run(run_tests())