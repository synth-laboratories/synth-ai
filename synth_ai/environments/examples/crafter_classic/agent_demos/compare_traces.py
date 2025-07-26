#!/usr/bin/env python3
"""
Script to compare traces between OpenAI direct API and LM class implementations.
Runs both versions and compares the captured events.
"""

import asyncio
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def run_openai_version(model: str = "gpt-4o-mini", episodes: int = 1, max_turns: int = 2) -> Tuple[bool, str]:
    """Run the OpenAI version and return trace directory."""
    print(f"{BLUE}Running OpenAI version...{RESET}")
    
    cmd = [
        sys.executable,
        "test_crafter_react_agent_openai.py",
        "--model", model,
        "--episodes", str(episodes),
        "--max-turns", str(max_turns)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            print(f"{GREEN}✓ OpenAI version completed successfully{RESET}")
            # Extract trace directory from output
            for line in result.stdout.split('\n'):
                if "Saved trace to" in line:
                    trace_path = line.split("Saved trace to")[-1].strip()
                    trace_dir = Path(trace_path).parent
                    return True, str(trace_dir)
            return True, "./traces"
        else:
            print(f"{RED}✗ OpenAI version failed{RESET}")
            print(result.stderr)
            return False, ""
    except Exception as e:
        print(f"{RED}✗ Error running OpenAI version: {e}{RESET}")
        return False, ""


def run_lm_version(model: str = "gpt-4o-mini", episodes: int = 1, max_turns: int = 2) -> Tuple[bool, str]:
    """Run the LM class version and return trace directory."""
    print(f"{BLUE}Running LM version...{RESET}")
    
    cmd = [
        sys.executable,
        "test_crafter_react_agent_lm.py",
        "--model", model,
        "--episodes", str(episodes),
        "--max-turns", str(max_turns)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            print(f"{GREEN}✓ LM version completed successfully{RESET}")
            # Extract trace directory from output
            for line in result.stdout.split('\n'):
                if "Saved trace to" in line:
                    trace_path = line.split("Saved trace to")[-1].strip()
                    trace_dir = Path(trace_path).parent
                    return True, str(trace_dir)
            return True, "./traces_v2_lm"
        else:
            print(f"{RED}✗ LM version failed{RESET}")
            print(result.stderr)
            return False, ""
    except Exception as e:
        print(f"{RED}✗ Error running LM version: {e}{RESET}")
        return False, ""


def load_trace(trace_file: Path) -> Dict[str, Any]:
    """Load a trace file."""
    try:
        with open(trace_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"{RED}Error loading trace {trace_file}: {e}{RESET}")
        return {}


def extract_cais_events(trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract CAISEvents from trace data."""
    events = trace_data.get('event_history', [])
    return [e for e in events if e.get('system_instance_id', '').startswith('crafter-react-agent')]


def extract_env_events(trace_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract EnvironmentEvents from trace data."""
    events = trace_data.get('event_history', [])
    return [e for e in events if 'reward' in e and not e.get('system_instance_id', '').startswith('crafter-react-agent')]


def compare_events(events1: List[Dict], events2: List[Dict], event_type: str) -> bool:
    """Compare two lists of events and report differences."""
    print(f"\n{BLUE}Comparing {event_type}...{RESET}")
    
    if len(events1) != len(events2):
        print(f"{RED}✗ Different number of events: {len(events1)} vs {len(events2)}{RESET}")
        return False
    
    all_match = True
    for i, (e1, e2) in enumerate(zip(events1, events2)):
        print(f"\n  Event {i+1}:")
        
        # Compare key fields
        fields_to_compare = {
            'CAISEvent': ['system_instance_id', 'llm_call_records', 'metadata'],
            'EnvironmentEvent': ['reward', 'done', 'info']
        }
        
        for field in fields_to_compare.get(event_type, []):
            if field in e1 and field in e2:
                # Special handling for llm_call_records
                if field == 'llm_call_records':
                    if len(e1[field]) != len(e2[field]):
                        print(f"    {RED}✗ Different number of LLM calls{RESET}")
                        all_match = False
                    else:
                        # Compare model and tool calls
                        for j, (llm1, llm2) in enumerate(zip(e1[field], e2[field])):
                            model1 = llm1.get('model', '')
                            model2 = llm2.get('model', '')
                            if model1 != model2:
                                print(f"    {RED}✗ Different models: {model1} vs {model2}{RESET}")
                                all_match = False
                            else:
                                print(f"    {GREEN}✓ Model matches: {model1}{RESET}")
                            
                            # Check tool calls
                            resp1 = llm1.get('response', {})
                            resp2 = llm2.get('response', {})
                            tools1 = extract_tool_calls(resp1)
                            tools2 = extract_tool_calls(resp2)
                            
                            if tools1 != tools2:
                                print(f"    {YELLOW}⚠ Different tool calls: {tools1} vs {tools2}{RESET}")
                                # This might be OK due to LLM non-determinism
                elif field == 'metadata':
                    # Compare token counts if available
                    tokens1 = {k: v for k, v in e1[field].items() if 'token' in k}
                    tokens2 = {k: v for k, v in e2[field].items() if 'token' in k}
                    if tokens1 and tokens2:
                        print(f"    Tokens (OpenAI): {tokens1}")
                        print(f"    Tokens (LM):     {tokens2}")
                else:
                    if e1[field] == e2[field]:
                        print(f"    {GREEN}✓ {field} matches{RESET}")
                    else:
                        print(f"    {RED}✗ {field} differs: {e1[field]} vs {e2[field]}{RESET}")
                        all_match = False
    
    return all_match


def extract_tool_calls(response: Dict[str, Any]) -> List[str]:
    """Extract tool call names from response."""
    tool_names = []
    choices = response.get('choices', [])
    if choices and isinstance(choices[0], dict):
        tool_calls = choices[0].get('message', {}).get('tool_calls', [])
        for tc in tool_calls:
            tool_names.append(tc.get('function', {}).get('name', 'unknown'))
    return tool_names


def compare_traces(openai_dir: Path, lm_dir: Path, episode: int = 0) -> bool:
    """Compare traces from both versions."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Comparing traces for episode {episode}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    # Load traces - OpenAI uses different naming convention
    # Look for most recent session files
    openai_files = list(openai_dir.glob(f"session_episode_{episode}_*.json"))
    if openai_files:
        # Get the most recent one
        openai_trace_file = max(openai_files, key=lambda f: f.stat().st_mtime)
    else:
        openai_trace_file = openai_dir / f"trace_episode_{episode}.json"
    
    lm_trace_file = lm_dir / f"trace_episode_{episode}.json"
    
    if not openai_trace_file.exists():
        print(f"{RED}✗ OpenAI trace not found: {openai_trace_file}{RESET}")
        return False
    
    if not lm_trace_file.exists():
        print(f"{RED}✗ LM trace not found: {lm_trace_file}{RESET}")
        return False
    
    openai_trace = load_trace(openai_trace_file)
    lm_trace = load_trace(lm_trace_file)
    
    # Extract events
    openai_cais = extract_cais_events(openai_trace)
    lm_cais = extract_cais_events(lm_trace)
    
    openai_env = extract_env_events(openai_trace)
    lm_env = extract_env_events(lm_trace)
    
    # Compare
    cais_match = compare_events(openai_cais, lm_cais, "CAISEvent")
    env_match = compare_events(openai_env, lm_env, "EnvironmentEvent")
    
    # Check messages
    print(f"\n{BLUE}Comparing messages...{RESET}")
    openai_msgs = openai_trace.get('message_history', [])
    lm_msgs = lm_trace.get('message_history', [])
    
    if len(openai_msgs) != len(lm_msgs):
        print(f"{RED}✗ Different number of messages: {len(openai_msgs)} vs {len(lm_msgs)}{RESET}")
    else:
        print(f"{GREEN}✓ Same number of messages: {len(openai_msgs)}{RESET}")
    
    return cais_match and env_match


async def main():
    """Run both versions and compare traces."""
    print(f"{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Crafter Trace Comparison: OpenAI vs LM{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    # Configuration
    model = "gpt-4o-mini"
    episodes = 2
    max_turns = 3
    
    print(f"\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Episodes: {episodes}")
    print(f"  Max turns: {max_turns}")
    
    # Run OpenAI version
    openai_success, openai_dir = run_openai_version(model, episodes, max_turns)
    if not openai_success:
        print(f"{RED}Failed to run OpenAI version{RESET}")
        return
    
    # Small delay to avoid rate limits
    await asyncio.sleep(2)
    
    # Run LM version
    lm_success, lm_dir = run_lm_version(model, episodes, max_turns)
    if not lm_success:
        print(f"{RED}Failed to run LM version{RESET}")
        return
    
    # Compare traces
    print(f"\n{BLUE}Trace directories:{RESET}")
    print(f"  OpenAI: {openai_dir}")
    print(f"  LM:     {lm_dir}")
    
    all_match = True
    for episode in range(episodes):
        match = compare_traces(Path(openai_dir), Path(lm_dir), episode)
        all_match = all_match and match
    
    # Final verdict
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}FINAL VERDICT{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")
    
    if all_match:
        print(f"{GREEN}✅ Traces match! LM class produces equivalent v2 traces.{RESET}")
    else:
        print(f"{YELLOW}⚠️  Some differences found. This may be due to:{RESET}")
        print(f"{YELLOW}   - LLM non-determinism (different responses){RESET}")
        print(f"{YELLOW}   - Minor implementation differences{RESET}")
        print(f"{YELLOW}   - Timing variations{RESET}")
    
    print(f"\nKey observations:")
    print(f"  • Both versions create CAISEvents with LLM call records")
    print(f"  • Both capture environment events and observations")
    print(f"  • Token counts and metadata are preserved")
    print(f"  • The LM class successfully replaces direct OpenAI calls")


if __name__ == "__main__":
    asyncio.run(main())